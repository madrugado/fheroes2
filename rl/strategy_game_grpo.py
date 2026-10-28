"""Whole-game GRPO for the strategic answers (user decision 2026-10-20: a single answer barely moves the end of a
game and the value cannot tell answers apart a few days later, so the reward is the END of the game and the group is a
set of games of the same seed).

  collect — per seed one color is ours; a group of `--group` games of that seed (every player on `--plan`, per-turn
            random streams so an answer changes the dice only until the next turn); in every game our answers are
            SAMPLED from the net's tempered distribution softmax(log p / T) (on-policy for the update below; the first
            version explored with uniform random answers and the update drifted: 2026-10-20). Reward of a game =
            minus the hero equivalent of our strongest hero at the end (strategy_games.equivalent_of) against ONE
            reference per group — the strongest rival hero of game 0 — so every game of a group is measured on the
            same scale. The model file is re-read when it changes (on-policy with a training run that saves it).
  The training side is `train_strategy_net --game-grpo FILES`: per group A_k = (R_k - mean) / (std + eps), and every
  answer of game k is pushed by A_k: -A log pi_T(answer), pi_T the same tempered distribution, with a KL leash.
  Idea (a) (2026-10-27): --kind-temperature / --kind-epsilon explore per query kind (more in army/build, where the v2
  regression found effects, less in targets/hire, where it found none); every answer records the T (or the "eps" mark)
  it was drawn with and the update tempers per answer (epsilon answers get no policy term).
  values  — (actor-critic, user decision 2026-10-24) a frozen critic's value of our player's state at the start of
            every day of every game (the strategic value head, hero-equivalent units); with them an answer given on day
            d is credited only with what happened AFTER d: A = [(R_k - R_mean) - (V_kd - V_mean_d)] / (std + eps) for
            lambda 1, and lambda < 1 trusts the critic for the later days too (GAE over the days, see group_records).
  deviations — (idea 3 of 2026-10-28) independent replays at a deviation point: per game, for sampled answers that
            deviate from the built-in AI's, the paired experiment of the sampled answer vs the built-in one under
            --salts lucks (salt 0 = the plain replay, FHEROES2_RESEED from the day after the query), everything else
            answered by the same sampled policy; per salt the hero parts of the deviation against the same-salt
            baseline (strategy_games.hero_state / hero_components). Writes pg-format rows (salt_scores / salt_parts,
            two scored options: the sampled answer and the built-in one) for `train_strategy_net pg` — the paired
            difference removes the salt's luck, so a deviation's effect is measurable at all; clean but expensive
            (2 x salts full-game replays per deviation).
  aux     — (idea 4 of 2026-10-28) every game also stores the intermediate events of our color per day ("aux":
            cumulative hero battles won/lost, own castles, the strongest hero's level); the training side
            (train_strategy_net --game-grpo-aux-weight, also on the JOLT path) turns them into a GAE advantage over
            the days and ADDS it to the game advantage — auxiliary only, never the main reward.

  rl/.venv/bin/python rl/strategy_game_grpo.py collect --model rl/models/unified_200m_joint12.pt --seeds 6001-6500 \\
      --out rl/data/game_grpo_c1.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

from harvest_battles import parse_seeds  # noqa: E402
from strategy_games import ( COLOR_LETTER, HERO_LOG2_RANGE, PolicyBranch, combine_hero_parts, equivalent_of,  # noqa: E402
                             hero_components, hero_state, play, strongest_rival_hero )
from strategy_net import answer_of, builtin_option, full_index, query_options  # noqa: E402
from strategy_rollout import same_query  # noqa: E402

COLORS = ( "Blue", "Green", "Red", "Yellow", "Orange", "Purple" )


def per_kind( spec: str, what: str ) -> dict[str, float]:
    """'target=1,build=3' -> {"target": 1.0, "build": 3.0}; empty spec -> {}. Kinds not listed fall
    back to the global value (idea a, 2026-10-27: explore where the effects are — the v2 regression
    found them for the army budget and building, none for hero targets and hire)."""
    table: dict[str, float] = {}
    for item in ( spec or "" ).split( "," ):
        if not item.strip():
            continue
        kind, sep, value = item.partition( "=" )
        if not sep:
            raise SystemExit( f"{what}: entry '{item}' is not kind=value" )
        table[kind.strip()] = float( value )
    return table


class SampleBranch( PolicyBranch ):
    """PolicyBranch whose policy SAMPLES our color's answers from its tempered distribution softmax(log p / T) — the
    policy the GRPO update trains (on-policy: an unbiased gradient; exploration ~ T). The game is kept once: our
    player's days (turn contexts) and answers with the index of the day they were given on, from which the training
    side rebuilds the history of every answer. `kind_temperature` / `kind_epsilon` (idea a) override T per query kind
    and add a uniform-random answer with that probability (an "eps" answer is off-policy: the training side gives it
    no policy term, its reward feeds the pooled advantage head); every answer stores the T it was drawn with and its
    global query index ("n"). `forced` = {global query index: choice} overrides that one answer — a replay branch of
    the deviations experiment (idea 3): the rng draws happen exactly as in the base game (so its sampled answers
    stay reproducible), only the override's answer differs; `expected` marks divergence when the query changed.
    `greedy_after` = n: every later answer is the policy's most probable option, no draws — the two replays of a
    deviation then differ by that one answer (and the salt's luck), not by the draws landing on other queries."""

    def __init__( self, policy, color: str, temperature: float, rng: random.Random,
                  kind_temperature: dict[str, float] | None = None, kind_epsilon: dict[str, float] | None = None,
                  forced: dict[int, int] | None = None, expected: dict | None = None, greedy_after: int | None = None ):
        super().__init__( policy, color )
        self.temperature = temperature
        self.kind_temperature = kind_temperature or {}
        self.kind_epsilon = kind_epsilon or {}
        self.forced = forced
        self.expected = expected
        self.greedy_after = greedy_after
        self.rng = rng
        self.our_days: list[dict] = []
        self.answers: list[dict] = []
        self.greedy_matches = 0

    def observe_turn( self, turn_context: dict ) -> None:
        super().observe_turn( turn_context )
        if turn_context.get( "p" ) == self.color:
            self.our_days.append( turn_context )

    def _answer( self, kind: str, event: dict ):
        if event.get( "p" ) != self.color:
            return super()._answer( kind, event )
        self.count += 1
        options = query_options( kind, event, getattr( self.policy, "view", "full" ) )
        choice = None
        if len( options ) >= 2:
            probs = self.policy.probabilities( kind, event )
            temperature = self.kind_temperature.get( kind, self.temperature )
            epsilon = self.kind_epsilon.get( kind, 0.0 )
            drawn = {"kind": kind, "event": event, "ctx": len( self.our_days ) - 1, "T": temperature, "n": self.count - 1}
            if self.greedy_after is not None and self.count - 1 > self.greedy_after:
                choice = max( range( len( options ) ), key=lambda i: probs[i] )
                drawn["greedy"] = 1
            elif epsilon > 0 and self.rng.random() < epsilon:
                choice = self.rng.randrange( len( options ) )  # uniform exploration: off-policy for the update
                drawn["eps"] = 1
            else:
                weights = [max( p, 1e-12 ) ** ( 1.0 / temperature ) for p in probs]
                choice = self.rng.choices( range( len( options ) ), weights=weights )[0]
                drawn["p"] = round( weights[choice] / sum( weights ), 5 )
            if self.forced is not None and ( self.count - 1 ) in self.forced:
                # The draws already happened (the base game's rng stream is preserved), only the answer is overridden.
                if self.expected is not None and not same_query( event, self.expected ):
                    self.diverged = True
                else:
                    choice = self.forced[self.count - 1]
            self.greedy_matches += choice == max( range( len( options ) ), key=lambda i: probs[i] )
            self.answers.append( {**drawn, "answer": choice} )
            self._remember( kind, event, full_index( kind, choice, getattr( self.policy, "view", "full" ) ) )  # value data: full view
        self.policy.record( kind, event, choice )
        answer = None if choice is None else answer_of( options[choice] )
        if kind == "target" and answer is not None and answer is ( event.get( "cands" ) or [None] )[0]:
            answer = None  # the top candidate is the built-in choice
        return answer


def collect( args ) -> None:
    from engine_bridge import BattleEnv
    from strategy_net import NetStrategyPolicy

    rng = random.Random( args.rng_seed )
    colors = args.colors.split( "," )
    kind_temperature = per_kind( args.kind_temperature, "--kind-temperature" )
    kind_epsilon = per_kind( args.kind_epsilon, "--kind-epsilon" )
    if kind_temperature or kind_epsilon:
        print( f"per-kind exploration: temperature {kind_temperature or '{}'}, epsilon {kind_epsilon or '{}'} "
               f"(the rest at T {args.temperature})", flush=True )
    duel_env = BattleEnv( map_name=args.map )
    policy, loaded = None, None
    started = time.time()
    try:
        with open( args.out, "a" ) as out:
            for number, seed in enumerate( parse_seeds( args.seeds ) ):
                stamp = os.path.getmtime( args.model )
                if stamp != loaded:  # a training run saved a newer model: stay on-policy
                    try:
                        policy, loaded = NetStrategyPolicy( args.model, threads=args.threads, bf16=args.bf16 ), stamp
                    except Exception as error:  # keep the previous model and try again with the next seed
                        print( f"model reload failed ({error})", flush=True )
                        if policy is None:
                            raise
                color = colors[number % len( colors )]
                games = []
                reference = None
                # JOLT (rl/jolt.py): the group is `--group` games of the privileged teacher (the built-in AI's answers as
                # hints) and as many of the student (no access to the built-in AI); one network, two views.
                roles = ["teacher"] * args.group + ["student"] * args.group if args.jolt else [None] * args.group
                for k, role in enumerate( roles ):
                    if role is not None:
                        policy.view = role
                    branch = SampleBranch( policy, color, args.temperature, random.Random( rng.random() ),
                                           kind_temperature, kind_epsilon )
                    try:
                        _, results, events, _ = play( args, seed, args.days, branch )
                    except ( TimeoutError, RuntimeError ) as error:
                        print( f"seed {seed} game {k}: failed ({error})", flush=True )
                        continue
                    if k == 0:
                        reference = strongest_rival_hero( results, color )
                    reward = -equivalent_of( duel_env, results, color, reference, seed )
                    state = str( ( results.get( color ) or {} ).get( "s", "" ) )
                    game = {"k": k, "reward": round( reward, 4 ), "greedy": branch.greedy_matches, "days": branch.our_days,
                            "answers": branch.answers, "outcome": 1 if state == "0" else ( -1 if state == "1" else 0 ),  # won / lost / open
                            "aux": aux_trajectory( branch, events, color )}
                    if role is not None:
                        game["role"] = role
                    games.append( game )
                if len( games ) >= 2:
                    out.write( json.dumps( {"seed": seed, "map": args.map, "color": color, "model_time": loaded, "hero_range": HERO_LOG2_RANGE,
                                            **( {"jolt": 1} if args.jolt else {} ), "games": games},
                                           separators=( ",", ":" ) ) + "\n" )
                    out.flush()
                rewards = [g["reward"] for g in games]
                answers = sum( len( g["answers"] ) for g in games )
                greedy = sum( g["greedy"] for g in games )
                print( f"seed {seed} {color}: rewards {rewards} (sd {statistics.pstdev( rewards ) if rewards else 0:.2f}), "
                       f"{answers} answers, {greedy / max( answers, 1 ):.0%} = the greedy one, {time.time() - started:.0f}s", flush=True )
    finally:
        duel_env.close()


# --- deviations (idea 3): independent replays at a deviation point ---------------------------------


def deviation_candidates( branch: SampleBranch, view: str, per_game: int, rng: random.Random ) -> list[int]:
    """Indexes of the game's sampled answers that DEVIATE from the built-in AI's answer (the v2-validated reference:
    real effects only for army budget and building) and are not epsilon draws; up to `per_game` of them, in query
    order."""
    picked = []
    for i, answer in enumerate( branch.answers ):
        if answer.get( "eps" ):
            continue
        kind, event = answer["kind"], answer["event"]
        options = query_options( kind, event, view )
        builtin = builtin_option( kind, options, event )
        if builtin is None or answer["answer"] == builtin:
            continue
        picked.append( i )
    return sorted( rng.sample( picked, min( per_game, len( picked ) ) ) )


def deviation_row( args, policy, view: str, branch: SampleBranch, branch_seed: int, seed: int, color: str, answer: dict,
                   previous: list[dict], duel_env, salts: int, hero_rule: str, rest: str = "greedy" ) -> dict | None:
    """The paired multi-luck experiment at one deviation (idea 3): the sampled answer vs the built-in AI's at that
    query, both replayed under `salts` lucks (salt 0 = the plain replay, FHEROES2_RESEED from the day after the
    query); before the query both replays reproduce the base game exactly (they share its rng stream), after it
    `rest` = "greedy" answers with the policy's most probable option in both (the default: otherwise the draws land
    on other queries once the games diverge and the replays differ by dozens of random answers, not by one) or
    "sampled" keeps drawing. Per salt the hero parts of the deviation against the same-salt
    baseline (strategy_games.hero_state / hero_components) — the paired difference removes that salt's luck.
    Returns a pg-format row (salt_scores / salt_parts keyed by the FULL view's option indexes) or None."""
    kind, event = answer["kind"], answer["event"]
    options = query_options( kind, event, view )
    builtin = builtin_option( kind, options, event )
    ours_full, builtin_full = full_index( kind, answer["answer"], view ), full_index( kind, builtin, view )
    if ours_full == builtin_full:
        return None
    day = event["t"]
    diffs, hero_parts, army_parts = [], [], []
    for salt in range( salts ):
        reseed = ( day + 1, salt ) if salt else None
        forced_base = {answer["n"]: builtin}
        forced_ours = {answer["n"]: answer["answer"]}
        after = answer["n"] if rest == "greedy" else None
        try:
            base = SampleBranch( policy, color, args.temperature, random.Random( branch_seed ),
                                 branch.kind_temperature, branch.kind_epsilon, forced_base, event, after )
            _, base_results, _, _ = play( args, seed, args.days, base, None, reseed )
            if base.diverged:
                return None
            alt = SampleBranch( policy, color, args.temperature, random.Random( branch_seed ),
                                branch.kind_temperature, branch.kind_epsilon, forced_ours, event, after )
            _, alt_results, _, _ = play( args, seed, args.days, alt, None, reseed )
            if alt.diverged:
                return None
        except ( TimeoutError, RuntimeError ) as error:
            print( f"seed {seed} query {answer['n']} salt {salt}: failed ({error})", flush=True )
            continue
        reference = hero_state( duel_env, base_results, color, seed )
        hero, army = hero_components( duel_env, reference, alt_results, color, seed )
        hero_parts.append( hero )
        army_parts.append( army )
        diffs.append( combine_hero_parts( hero, army, hero_rule ) )
    if len( diffs ) < 2:
        return None  # fewer than two lucks: no standard error
    mean = statistics.fmean( diffs )
    se = statistics.stdev( diffs ) / math.sqrt( len( diffs ) )
    ctx = answer["ctx"]
    return {"kind": kind, "event": event, "context": branch.our_days[ctx] if ctx >= 0 else None,
            "history": {"days": branch.our_days[:max( ctx, 0 )], "decisions": list( previous )},
            "seed": seed, "map": args.map, "color": color, "n": answer["n"], "salts": salts, "hero_rule": hero_rule, "rest": rest,
            "own": ours_full, "builtin": builtin_full,
            "salt_scores": {str( ours_full ): diffs, str( builtin_full ): [0.0] * len( diffs )},
            "salt_parts": {str( ours_full ): {"hero": hero_parts, "army": army_parts},
                           str( builtin_full ): {"hero": [0.0] * len( diffs ), "army": [0.0] * len( diffs )}},
            "gap": [round( mean, 4 ), round( se, 4 )]}


def collect_deviations( args ) -> None:
    """Per seed one sampled game (as `collect`), then for up to --per-game deviating answers the paired multi-luck
    experiment (deviation_row, idea 3). pg-format rows for `train_strategy_net pg --data <out>`."""
    from engine_bridge import BattleEnv
    from strategy_net import NetStrategyPolicy

    rng = random.Random( args.rng_seed )
    colors = args.colors.split( "," )
    kind_temperature = per_kind( args.kind_temperature, "--kind-temperature" )
    kind_epsilon = per_kind( args.kind_epsilon, "--kind-epsilon" )
    duel_env = BattleEnv( map_name=args.map )
    policy, loaded, total, started = None, None, 0, time.time()
    try:
        with open( args.out, "a" ) as out:
            for number, seed in enumerate( parse_seeds( args.seeds ) ):
                stamp = os.path.getmtime( args.model )
                if stamp != loaded:  # a training run saved a newer model: stay on-policy
                    try:
                        policy, loaded = NetStrategyPolicy( args.model, threads=args.threads, bf16=args.bf16 ), stamp
                    except Exception as error:
                        print( f"model reload failed ({error})", flush=True )
                        if policy is None:
                            raise
                color = colors[number % len( colors )]
                view = getattr( policy, "view", "full" )
                branch_seed = rng.getrandbits( 32 )
                branch = SampleBranch( policy, color, args.temperature, random.Random( branch_seed ),
                                       kind_temperature, kind_epsilon )
                try:
                    play( args, seed, args.days, branch )
                except ( TimeoutError, RuntimeError ) as error:
                    print( f"seed {seed}: game failed ({error})", flush=True )
                    continue
                candidates = set( deviation_candidates( branch, view, args.per_game, random.Random( rng.getrandbits( 32 ) ) ) )
                rows = 0
                for i, answer in enumerate( branch.answers ):
                    if i not in candidates:
                        continue
                    previous = [{"kind": a["kind"], "event": a["event"],
                                 "context": branch.our_days[a["ctx"]] if a["ctx"] >= 0 else None, "answer": a["answer"]}
                                for a in branch.answers[:i]]
                    row = deviation_row( args, policy, view, branch, branch_seed, seed, color, answer, previous, duel_env,
                                         args.salts, args.hero_rule, args.rest )
                    if row is None:
                        continue
                    out.write( json.dumps( row, separators=( ",", ":" ) ) + "\n" )
                    out.flush()
                    rows += 1
                    total += 1
                print( f"seed {seed} {color}: {len( candidates )} deviations tried, {rows} rows (total {total}, "
                       f"{time.time() - started:.0f}s)", flush=True )
    finally:
        duel_env.close()


def day_advantages( rewards: list[float], values: list[list[float]], lam: float ) -> list[list[float]]:
    """Per game and day the advantage of the answers given that day (not yet divided by the group's spread): with the
    critic's values V_kd of the state at the start of day d, both centred over the group's games (the reward by its
    group mean, a day's values by the mean over the games that reached that day — the group shares one reference, the
    critic predicts against its own), the terminal "value" being the centred reward:
        A_kd = sum_{j >= d} lam^(j - d) (V~_k,j+1 - V~_kj),  V~_k,D = R_k - R_mean
    lam 1: R~_k - V~_kd (what the game gained after day d over what its state at d promised); lam 0: one day's change."""
    mean = statistics.fmean( rewards )
    days = max( len( v ) for v in values )
    day_mean = []
    for d in range( days ):
        reached = [v[d] for v in values if len( v ) > d]
        day_mean.append( statistics.fmean( reached ) )
    advantages = []
    for reward, game_values in zip( rewards, values ):
        centred = [v - day_mean[d] for d, v in enumerate( game_values )] + [reward - mean]
        adv = [0.0] * len( game_values )
        running = 0.0
        for d in range( len( game_values ) - 1, -1, -1 ):
            running = ( centred[d + 1] - centred[d] ) + lam * running
            adv[d] = running
        advantages.append( adv )
    return advantages


def aux_trajectory( branch: SampleBranch, events: list[dict], color: str ) -> list[dict]:
    """The intermediate events of our color per its own day (idea 4, stored per game as "aux"): cumulative
    hero-vs-hero battles won / lost ("w"/"l", a hero on both sides, as strategy_games.real_hero_battles), other
    battles won / lost ("mw"/"ml": neutral monsters, garrisons without a hero), the number of own castles and the
    strongest hero's level (the turn contexts). One entry per our day, aligned with branch.our_days."""
    letter = COLOR_LETTER[color]
    starts: dict = {}
    wins: dict[tuple[bool, int], int] = {}
    losses: dict[tuple[bool, int], int] = {}
    for ev in events:
        if ev.get( "ev" ) == "battle_start":
            starts[ev.get( "bid" )] = ev
        elif ev.get( "ev" ) == "battle_end":
            start = starts.get( ev.get( "bid" ) ) or {}
            if start.get( "att", {} ).get( "c" ) == letter:
                ours, theirs = "att", "def"
            elif start.get( "def", {} ).get( "c" ) == letter:
                ours, theirs = "def", "att"
            else:
                continue
            winner, day = ev.get( "winner" ), ev.get( "t", 0 )
            key = ( "hero" in start.get( "att", {} ) and "hero" in start.get( "def", {} ), day )
            if winner == letter:
                wins[key] = wins.get( key, 0 ) + 1
            elif winner == start[theirs].get( "c" ):
                losses[key] = losses.get( key, 0 ) + 1
    def upto( table: dict, heroes: bool, day: int ) -> int:
        return sum( count for ( hero_battle, when ), count in table.items() if hero_battle == heroes and when <= day )

    trajectory = []
    for ctx in branch.our_days:
        day = ctx.get( "t", 0 )
        trajectory.append( {"d": day, "w": upto( wins, True, day ), "l": upto( losses, True, day ),
                            "mw": upto( wins, False, day ), "ml": upto( losses, False, day ),
                            "k": len( ctx.get( "castles" ) or [] ),
                            "v": max( ( h.get( "lvl", 0 ) for h in ctx.get( "heroes" ) or [] ), default=0 )} )
    return trajectory


def aux_score( entry: dict, weights: dict[str, float] ) -> float:
    """The intermediate-events score of one day (idea 4): net hero-vs-hero battles won ("battles"), net other battles
    won ("monsters", off unless given), own castles, the strongest hero's level."""
    return ( weights.get( "battles", 0.0 ) * ( entry.get( "w", 0 ) - entry.get( "l", 0 ) )
             + weights.get( "monsters", 0.0 ) * ( entry.get( "mw", 0 ) - entry.get( "ml", 0 ) )
             + weights.get( "castles", 0.0 ) * entry.get( "k", 0 ) + weights.get( "level", 0.0 ) * entry.get( "v", 0 ) )


def aux_advantages( games: list[dict], weights: dict[str, float], lam: float, eps: float ) -> list[list[float]]:
    """Per game and day the AUXILIARY advantage of the intermediate-events trajectory (idea 4: denser than the game
    reward, never the main reward): the aux scores centred per day over the group's games (the final score centred
    over the games' finals as the terminal), GAE with `lam` over the days (day_advantages' shape), normalized by
    (the pooled std of the raw advantages + eps) so the scale matches the game advantage's. Games without a
    trajectory (old data) get none."""
    scores = [[aux_score( entry, weights ) for entry in game.get( "aux" ) or []] for game in games]
    if not scores or any( not s for s in scores ):
        return [[] for _ in games]
    days = max( len( s ) for s in scores )
    day_mean = [statistics.fmean( [s[d] for s in scores if len( s ) > d] ) for d in range( days )]
    final_mean = statistics.fmean( s[-1] for s in scores )
    raw = []
    for s in scores:
        centred = [v - day_mean[d] for d, v in enumerate( s )] + [s[-1] - final_mean]
        adv = [0.0] * len( s )
        running = 0.0
        for d in range( len( s ) - 1, -1, -1 ):
            running = ( centred[d + 1] - centred[d] ) + lam * running
            adv[d] = running
        raw.append( adv )
    pooled = statistics.pstdev( [v for row in raw for v in row] )
    if pooled + eps <= 0:
        return [[] for _ in games]
    return [[v / ( pooled + eps ) for v in row] for row in raw]


def group_records( group: dict, eps: float = 0.3, values: list[list[float]] | None = None, lam: float = 1.0,
                   aux_weight: float = 0.0, aux: dict[str, float] | None = None ) -> list[dict]:
    """The GRPO records of one group: every sampled answer of game k with A_k = (R_k - mean) / (std + eps) and the
    history it was given in (`train_strategy_net` pg format: "adv" {option: advantage}). With `values` (per game the
    critic's value at the start of every day, `values` subcommand) an answer gets the advantage of its day instead
    (day_advantages / (std + eps)). With `aux` weights and `aux_weight` > 0 the intermediate-events advantage
    (aux_advantages, idea 4) is ADDED to every non-eps answer's advantage — auxiliary only, never the main reward;
    baked together here, so a weight change takes effect at the next refresh and the records carry one "adv"."""
    rewards = [g["reward"] for g in group["games"]]
    if len( rewards ) < 2:
        return []
    mean, spread = statistics.fmean( rewards ), statistics.pstdev( rewards )
    if spread < 1e-9:
        return []  # every game of the group scored the same: no signal
    by_day = day_advantages( rewards, values, lam ) if values is not None else None
    aux_adv = aux_advantages( group["games"], aux, lam, eps ) if aux and aux_weight > 0 else None
    records = []
    for number, game in enumerate( group["games"] ):
        advantage = ( game["reward"] - mean ) / ( spread + eps )
        days = game["days"]
        previous: list[dict] = []
        for answer in game["answers"]:
            ctx = answer["ctx"]
            if answer.get( "eps" ):
                # The collector's uniform exploration: off-policy for the update, no policy term.
                value = 0.0
            else:
                if by_day is not None and by_day[number]:
                    value = by_day[number][min( max( ctx, 0 ), len( by_day[number] ) - 1 )] / ( spread + eps )
                else:
                    value = advantage
                if aux_adv is not None and aux_adv[number]:
                    value = value + aux_weight * aux_adv[number][min( max( ctx, 0 ), len( aux_adv[number] ) - 1 )]
            records.append( {"kind": answer["kind"], "event": answer["event"], "context": days[ctx] if ctx >= 0 else None,
                             "history": {"days": days[:max( ctx, 0 )], "decisions": list( previous )},
                             "seed": group["seed"], "map": group["map"], "adv": {answer["answer"]: value},
                             **( {"T": answer["T"]} if "T" in answer else {} ),
                             **( {"eps": 1} if answer.get( "eps" ) else {} )} )
            previous.append( {"kind": answer["kind"], "event": answer["event"], "context": days[ctx] if ctx >= 0 else None,
                              "answer": answer["answer"]} )
    return records


def group_key( group: dict ) -> tuple:
    return ( group["map"], group["seed"], group["color"] )


def game_day_values( model, games: list[dict], obj_vocab: list[int], scale: float, batch: int = 32 ) -> list[list[float]]:
    """The critic's value of our player's state at the start of every day of every game (hero-equivalent units: the
    value head's output x `scale`, the training run's --value-scale)."""
    import torch

    from strategy_value import values

    trajectories = [{"days": g["days"], "decisions": [{"kind": a["kind"], "event": a["event"], "ctx": a["ctx"], "answer": a["answer"]}
                                                      for a in g["answers"]]} for g in games]
    states = sorted( ( ( t, d ) for t in trajectories for d in range( len( t["days"] ) ) ), key=lambda s: s[1] )  # similar lengths
    out: dict[tuple[int, int], float] = {}
    with torch.no_grad():
        for start in range( 0, len( states ), batch ):
            chunk = states[start:start + batch]
            for ( trajectory, day ), value in zip( chunk, values( model, chunk, obj_vocab ).float().tolist() ):
                out[( id( trajectory ), day )] = value * scale
    return [[out[( id( t ), d )] for d in range( len( t["days"] ) )] for t in trajectories]


def compute_values( args ) -> None:

    import torch

    from precision import autocast_for, for_inference
    from transformer_model import load_checkpoint

    torch.set_num_threads( args.threads )
    model = for_inference( load_checkpoint( args.model ), "cpu", args.bf16 )
    obj_vocab = list( model.config.get( "obj_vocab", [] ) )
    precision = autocast_for( model )
    done = set()
    if os.path.exists( args.out ):  # resume
        with open( args.out ) as f:
            done = {tuple( json.loads( line )["key"] ) for line in f if line.strip()}
    started, count = time.time(), 0
    with open( args.out, "a" ) as out, precision:
        for path in args.data:
            with open( path ) as f:
                for line in f:
                    if not line.strip():
                        continue
                    group = json.loads( line )
                    key = group_key( group )
                    if key in done:
                        continue
                    day_values = game_day_values( model, group["games"], obj_vocab, args.scale, args.batch )
                    out.write( json.dumps( {"key": list( key ), "values": [[round( v, 4 ) for v in g] for g in day_values]},
                                           separators=( ",", ":" ) ) + "\n" )
                    out.flush()
                    done.add( key )
                    count += 1
                    print( f"{path} {key}: {sum( len( v ) for v in day_values )} states, {count} groups, {time.time() - started:.0f}s", flush=True )


def load_values( paths: list[str] ) -> dict:
    """{(map, seed, color): per game the day values} of `values` files."""
    table = {}
    for path in paths:
        with open( path ) as f:
            for line in f:
                if line.strip():
                    row = json.loads( line )
                    table[tuple( row["key"] )] = row["values"]
    return table


def main() -> None:
    parser = argparse.ArgumentParser( description="Whole-game GRPO data for the strategic answers" )
    sub = parser.add_subparsers( dest="cmd", required=True )
    c = sub.add_parser( "collect" )
    c.add_argument( "--binary", default="./fheroes2" )
    c.add_argument( "--map", default="Battlefi.mp2" )
    c.add_argument( "--days", type=int, default=30 )
    c.add_argument( "--seeds", required=True )
    c.add_argument( "--colors", default=",".join( COLORS ) )
    c.add_argument( "--plan", default="champion=1,split_singles=1,max_heroes=3" )
    c.add_argument( "--model", required=True, help="strategic net (re-read when the file changes)" )
    c.add_argument( "--group", type=int, default=8, help="games per seed" )
    c.add_argument( "--temperature", type=float, default=2.0, help="answers sampled from softmax(log p / T)" )
    c.add_argument( "--kind-temperature", default="", help="T per query kind, 'target=1,build=3,army=4' (idea a: explore "
                    "where the regression found effects — army/build — and keep the kinds without effects near greedy); "
                    "kinds not listed use --temperature. The answer records the T it was drawn with, the update tempers "
                    "per answer" )
    c.add_argument( "--kind-epsilon", default="", help="per kind, this probability of a UNIFORM-random answer (covers rare "
                    "options for the pooled advantage head; the answer is marked 'eps' and gets no policy term in the update)" )
    c.add_argument( "--rng-seed", type=int, default=1 )
    c.add_argument( "--threads", type=int, default=1, help="torch threads of the strategic net (one per collector core)" )
    c.add_argument( "--bf16", action="store_true", help="bf16 inference (CPUs with avx512_bf16)" )
    c.add_argument( "--no-rng-streams", dest="rng_streams", action="store_false" )
    c.add_argument( "--jolt", action="store_true", help="JOLT groups: --group teacher games + --group student games per seed "
                    "(train_strategy_net --jolt-games); use --temperature 1 (the JOLT update is on-policy for the plain policy)" )
    c.add_argument( "--out", required=True )
    v = sub.add_parser( "values" )
    v.add_argument( "--model", required=True, help="the frozen critic (a unified checkpoint with a trained strategic value head)" )
    v.add_argument( "--data", nargs="+", required=True, help="collect files" )
    v.add_argument( "--scale", type=float, required=True, help="value head output -> hero-equivalent units (the --value-scale it was "
                    "trained with: 3 for critics of the old hero range, 6 for the current one)" )
    v.add_argument( "--batch", type=int, default=32 )
    v.add_argument( "--threads", type=int, default=8 )
    v.add_argument( "--bf16", action="store_true" )
    v.add_argument( "--out", required=True )
    d = sub.add_parser( "deviations", help="paired multi-luck replays at deviation points (idea 3): pg rows for "
                        "train_strategy_net pg" )
    d.add_argument( "--binary", default="./fheroes2" )
    d.add_argument( "--map", default="Battlefi.mp2" )
    d.add_argument( "--days", type=int, default=30 )
    d.add_argument( "--seeds", required=True )
    d.add_argument( "--colors", default=",".join( COLORS ) )
    d.add_argument( "--plan", default="champion=1,split_singles=1,max_heroes=3" )
    d.add_argument( "--model", required=True, help="strategic net (re-read when the file changes)" )
    d.add_argument( "--per-game", type=int, default=4, help="deviating answers experimented with per game" )
    d.add_argument( "--salts", type=int, default=4, help="lucks per answer (2 x --salts full-game replays per deviation)" )
    d.add_argument( "--hero-rule", choices=["hero", "army", "mean"], default="mean",
                    help="how the hero parts of --label hero make the per-salt score" )
    d.add_argument( "--temperature", type=float, default=2.0 )
    d.add_argument( "--rest", choices=["greedy", "sampled"], default="greedy", help="answers AFTER the deviation in both "
                    "replays: the policy's most probable option (default; the replays differ by the one answer) or drawn" )
    d.add_argument( "--kind-temperature", default="" )
    d.add_argument( "--kind-epsilon", default="" )
    d.add_argument( "--rng-seed", type=int, default=1 )
    d.add_argument( "--threads", type=int, default=1 )
    d.add_argument( "--bf16", action="store_true" )
    d.add_argument( "--no-rng-streams", dest="rng_streams", action="store_false" )
    d.add_argument( "--out", required=True )
    args = parser.parse_args()
    if args.cmd == "values":
        compute_values( args )
    elif args.cmd == "deviations":
        collect_deviations( args )
    else:
        collect( args )


if __name__ == "__main__":
    main()
