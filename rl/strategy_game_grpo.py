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

  rl/.venv/bin/python rl/strategy_game_grpo.py collect --model rl/models/unified_200m_joint12.pt --seeds 6001-6500 \\
      --out rl/data/game_grpo_c1.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

from harvest_battles import parse_seeds  # noqa: E402
from strategy_games import HERO_LOG2_RANGE, PolicyBranch, equivalent_of, play, strongest_rival_hero  # noqa: E402
from strategy_net import answer_of, full_index, query_options  # noqa: E402

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
    no policy term, its reward feeds the pooled advantage head); every answer stores the T it was drawn with."""

    def __init__( self, policy, color: str, temperature: float, rng: random.Random,
                  kind_temperature: dict[str, float] | None = None, kind_epsilon: dict[str, float] | None = None ):
        super().__init__( policy, color )
        self.temperature = temperature
        self.kind_temperature = kind_temperature or {}
        self.kind_epsilon = kind_epsilon or {}
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
            drawn = {"kind": kind, "event": event, "ctx": len( self.our_days ) - 1, "T": temperature}
            if epsilon > 0 and self.rng.random() < epsilon:
                choice = self.rng.randrange( len( options ) )  # uniform exploration: off-policy for the update
                drawn["eps"] = 1
            else:
                weights = [max( p, 1e-12 ) ** ( 1.0 / temperature ) for p in probs]
                choice = self.rng.choices( range( len( options ) ), weights=weights )[0]
                drawn["p"] = round( weights[choice] / sum( weights ), 5 )
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
                        _, results, _, _ = play( args, seed, args.days, branch )
                    except ( TimeoutError, RuntimeError ) as error:
                        print( f"seed {seed} game {k}: failed ({error})", flush=True )
                        continue
                    if k == 0:
                        reference = strongest_rival_hero( results, color )
                    reward = -equivalent_of( duel_env, results, color, reference, seed )
                    state = str( ( results.get( color ) or {} ).get( "s", "" ) )
                    game = {"k": k, "reward": round( reward, 4 ), "greedy": branch.greedy_matches, "days": branch.our_days,
                            "answers": branch.answers, "outcome": 1 if state == "0" else ( -1 if state == "1" else 0 )}  # won / lost / open
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


def group_records( group: dict, eps: float = 0.3, values: list[list[float]] | None = None, lam: float = 1.0 ) -> list[dict]:
    """The GRPO records of one group: every sampled answer of game k with A_k = (R_k - mean) / (std + eps) and the
    history it was given in (`train_strategy_net` pg format: "adv" {option: advantage}). With `values` (per game the
    critic's value at the start of every day, `values` subcommand) an answer gets the advantage of its day instead
    (day_advantages / (std + eps))."""
    rewards = [g["reward"] for g in group["games"]]
    if len( rewards ) < 2:
        return []
    mean, spread = statistics.fmean( rewards ), statistics.pstdev( rewards )
    if spread < 1e-9:
        return []  # every game of the group scored the same: no signal
    by_day = day_advantages( rewards, values, lam ) if values is not None else None
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
            elif by_day is not None and by_day[number]:
                value = by_day[number][min( max( ctx, 0 ), len( by_day[number] ) - 1 )] / ( spread + eps )
            else:
                value = advantage
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
    args = parser.parse_args()
    if args.cmd == "values":
        compute_values( args )
    else:
        collect( args )


if __name__ == "__main__":
    main()
