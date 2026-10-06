"""DPO preference pairs from our own games against the built-in AI (on-policy).

One color of a seeded game is played by the unified transformer's strategic output
(strategy_net.NetStrategyPolicy, deterministic argmax), every other player by the built-in AI.
Games are deterministic per (seed, answers), so for a query n of our color (day t) the game can
be replayed exactly up to n and branched:

  baseline branch  — our policy answers everything (= the played game cut at day t + H);
  option branch j  — identical, but query n is answered with option j; our policy continues.

`--label hero` (user design 2026-10-06, for "more strength for the final duel"): both parts are
measured at the end of the game, per luck, positive = the branch is stronger, in log2 units:
  hero — the branch's strongest hero against a fixed reference (the strongest rival hero of the
         baseline of the same luck): how much less army it needs to win half of the duels
         (hero_equivalent: army, skills, artifacts and magic in one number);
  army — log2 of the ratio of the strongest heroes' army strengths.
Both are stored per luck in the pairs (`salt_parts`); the label is `--hero-rule` of them.

The label of option j is the difference of our player's score between its branch and the
baseline (the policy's own answer has label 0), averaged over the horizons (`--horizons 7,14`:
one week and two weeks after the query; one replay per branch to the last horizon, the earlier
states come from the engine's "day_report" events, FHEROES2_REPORT_DAYS). The score at a horizon
(`--label duel`, default):
  - if our player fought a hero-vs-hero battle between the query and the horizon (the branch's
    AI log), the battle's result: +1 win / -1 loss + share of own army strength left - share of
    the enemy's;
  - otherwise a DUEL in the battle server: our strongest hero against the strongest hero of the
    other players (both from the report's "top", save-game serializations with army, skills and
    artifacts), played by the built-in AI on both sides attacking and defending, each with 3
    battle seeds; the mean of the six battle scores.
A military outcome is a much less noisy label than the army/castle statistics after a week
(`--label stats`: strategy_model.label_score), which a reshuffled random stream dominates.
Candidates per query: the policy's answer, the
built-in AI's answer (target: top candidate, hire: `bi`, army: 100%; unknown for build) and
`--random` random other options. The best and the worst candidate form a pair when they differ
by more than --margin — so the pairs are about the states OUR policy reaches, and a DPO step moves
it towards the answers that actually worked better from there.

Runs ONE engine at a time in this process (the model is loaded once); see the machine load rule
in AGENTS.md.

Usage:
    rl/.venv/bin/python rl/strategy_games.py --model rl/models/unified_sft.pt --map 2kings.mp2 \\
        --days 30 --seeds 1-20 --color Blue --out rl/data/strategy_game_prefs.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
import time

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

from battle_prefs import move_score  # noqa: E402
from engine_bridge import BattleEnv  # noqa: E402
from harvest_battles import parse_seeds  # noqa: E402
from strategy_bench import player_stats  # noqa: E402
from strategy_env import StrategyEnv  # noqa: E402
from strategy_model import label_score  # noqa: E402
from strategy_net import BUILTIN, NetStrategyPolicy, answer_of, query_options  # noqa: E402,F401
from strategy_rollout import same_query  # noqa: E402

KIND_OF_EVENT = {"decision": "target", "build": "build", "hire": "hire", "army": "army"}


class PolicyBranch:
    """Strategic policy for one replay: `policy` answers `color`'s queries, the built-in AI all
    others; the query with global index `pick_at` is answered with `answer` instead (checked
    against `expected`). Records every query of `color` with its context and the answer given."""

    def __init__( self, policy, color: str, pick_at: int | None = None, answer_index: int | None = None, expected: dict | None = None ):
        self.policy = policy
        policy.reset()  # a new game: the policy's history starts empty
        self.color = color
        self.pick_at = pick_at
        self.answer_index = answer_index
        self.expected = expected
        self.count = 0
        self.queries: list[dict] = []
        self.contexts: dict[str, dict] = {}
        self.diverged = False
        # Every player's game in the value-data format (strategy_value): days and answers (option index;
        # the day as an index into the player's days). The other players answer as the built-in AI.
        self.days: dict[str, list[dict]] = {}
        self.decisions: dict[str, list[dict]] = {}

    def observe_turn( self, turn_context: dict ) -> None:
        self.contexts[turn_context.get( "p" )] = turn_context
        self.days.setdefault( turn_context.get( "p" ), [] ).append( turn_context )
        if turn_context.get( "p" ) == self.color:
            self.policy.observe_turn( turn_context )

    def _answer( self, kind: str, event: dict ):
        index = self.count
        self.count += 1
        if event.get( "p" ) != self.color:
            options = query_options( kind, event )
            if len( options ) >= 2:
                self._remember( kind, event, builtin_index( kind, options, event ) )
            return None
        # The history the policy saw before this query (stored with the query for DPO pairs).
        history = self.policy.history( self.color )
        snapshot = {"days": list( history["days"] ), "decisions": list( history["decisions"] )}
        options = query_options( kind, event )
        if index == self.pick_at and self.expected is not None and not same_query( event, self.expected ):
            self.diverged = True
        if index == self.pick_at and not self.diverged:
            choice = self.answer_index
        else:
            choice = self.policy.decide( kind, event )
        self.policy.record( kind, event, choice )
        if len( options ) >= 2 and choice is not None:
            self._remember( kind, event, choice )
        answer = None if choice is None else answer_of( options[choice] )
        if kind == "target" and answer is not None and answer is ( event.get( "cands" ) or [None] )[0]:
            answer = None  # the top candidate is the built-in choice
        self.queries.append( {"n": index, "kind": kind, "event": event, "context": self.contexts.get( self.color ),
                              "answer": answer, "answer_index": choice, "history": snapshot} )
        return answer

    def _remember( self, kind: str, event: dict, index: int | None ) -> None:
        color = event.get( "p" )
        self.decisions.setdefault( color, [] ).append( {"kind": kind, "event": event, "ctx": len( self.days.get( color, [] ) ) - 1, "answer": index} )

    def trajectory( self, color: str ) -> dict:
        """The player's game as a strategic value record (without the label)."""
        return {"color": color, "days": self.days.get( color, [] ), "decisions": self.decisions.get( color, [] )}

    def __call__( self, decision: dict ):
        return self._answer( "target", decision )

    def build( self, event: dict ):
        return self._answer( "build", event )

    def hire( self, event: dict ):
        return self._answer( "hire", event )

    def army( self, event: dict ):
        return self._answer( "army", event )


COLOR_LETTER = {"Blue": "B", "Green": "G", "Red": "R", "Yellow": "Y", "Orange": "O", "Purple": "P"}


def play( args, seed: int, days: int, branch: PolicyBranch, report_days: list[int] | None = None,
          reseed: tuple[int, int] | None = None ) -> tuple[dict, dict, list[dict], dict]:
    """One seeded game (one engine, under nice); `reseed` = (day, salt) gives the game other luck
    from that day on. Returns (stats of the branch's color, game_end results by color, the game's
    AI log events, {day: results by color} of the day reports)."""
    handle, log_path = tempfile.mkstemp( prefix="strategy_branch_", suffix=".jsonl" )
    os.close( handle )
    env = StrategyEnv( binary=args.binary, map_name=args.map, days=days, playthroughs=1, seed=seed, ai_log=log_path,
                       report_days=report_days, reseed=reseed )
    try:
        summaries = env.run( branch )
    finally:
        env.close()
    try:
        with open( log_path ) as f:
            events = [json.loads( line ) for line in f if line.strip()]
    except ( OSError, json.JSONDecodeError ):
        events = []
    finally:
        os.remove( log_path )
    if not summaries:
        raise RuntimeError( f"seed {seed}: no game_end" )
    results = {r["c"]: r for r in summaries[-1].get( "results" ) or []}
    reports = {ev["t"]: {r["c"]: r for r in ev.get( "results" ) or []} for ev in env.day_reports}
    return player_stats( summaries[-1] )[branch.color], results, events, reports


def battle_score( outcome: float, own0: float, own1: float, enemy0: float, enemy1: float ) -> float:
    """outcome (+1 / 0 / -1) + share of own strength left - share of enemy strength left (the
    battle_prefs.move_score scale)."""
    own = own1 / own0 if own0 > 0 else 0.0
    enemy = enemy1 / enemy0 if enemy0 > 0 else 0.0
    return outcome + own - enemy


def real_hero_battles( events: list[dict], color: str, first_day: int, last_day: int, rival: str | None = None ) -> list[float]:
    """Scores of the hero-vs-hero battles of `color` on days [first_day, last_day] (AI log); only
    those against the `rival` color when given."""
    letter = COLOR_LETTER[color]
    rival_letter = COLOR_LETTER.get( rival ) if rival else None
    starts = {}
    scores = []
    for ev in events:
        if ev.get( "ev" ) == "battle_start":
            starts[ev.get( "bid" )] = ev
        elif ev.get( "ev" ) == "battle_end" and first_day <= ev.get( "t", 0 ) <= last_day:
            start = starts.get( ev.get( "bid" ) )
            if start is None or "hero" not in start.get( "att", {} ) or "hero" not in start.get( "def", {} ):
                continue
            if start["att"].get( "c" ) == letter:
                ours, theirs = "att", "def"
            elif start["def"].get( "c" ) == letter:
                ours, theirs = "def", "att"
            else:
                continue
            if rival_letter is not None and start[theirs].get( "c" ) != rival_letter:
                continue
            winner = ev.get( "winner" )
            outcome = 1.0 if winner == letter else ( 0.0 if winner == "N" else -1.0 )
            scores.append( battle_score( outcome, ev.get( f"{ours}0", 0 ), ev.get( f"{ours}1", 0 ), ev.get( f"{theirs}0", 0 ), ev.get( f"{theirs}1", 0 ) ) )
    return scores


DUEL_SEEDS = 5  # every duel orientation is fought with this many battle seeds (user requests: less battle luck; 3 -> 5 on 2026-10-03)


def duel_between( duel_env: BattleEnv, ours: dict, theirs: dict, seed: int, seeds: int = DUEL_SEEDS ) -> float:
    """Two heroes ("top" records: hid + serialization) fight to the end with the built-in AI on both
    sides, ours attacking and defending, each with `seeds` battle seeds; the mean battle score."""
    scores = []
    for battle_seed in range( seed, seed + seeds ):
        for our_side, heroes in ( ( "att", ( ours, theirs ) ), ( "def", ( theirs, ours ) ) ):
            root = duel_env.new_battle( seed=battle_seed, attacker="", defender="", hero_att=( heroes[0]["hid"], heroes[0]["hero"] ),
                                        hero_def=( heroes[1]["hid"], heroes[1]["hero"] ) )
            if root is None or root.get( "ev" ) != "state":
                continue
            if root.get( "result" ):
                final = root
            else:
                duel_env.snapshot_save( 1 )
                final = duel_env.snapshot_restore( 1, rollout=True )
            scores.append( move_score( root, final, our_side ) )
    return sum( scores ) / len( scores ) if scores else 0.0


FINAL_DUEL_STEPS = 4  # bisection steps of the handicap search (log2 of the army ratio in [-1, 1])


def duel_wins( duel_env: BattleEnv, ours: dict, theirs: dict, seed: int, seeds: int, our_scale: int, their_scale: int ) -> tuple[int, int]:
    """(battles won by our hero, battles fought): ours attacking and defending, `seeds` battle seeds
    each, every stack of a side at its scale (percent of its count), built-in AI on both sides."""
    wins = fought = 0
    for battle_seed in range( seed, seed + seeds ):
        for our_side, heroes, scales in ( ( "att", ( ours, theirs ), ( our_scale, their_scale ) ), ( "def", ( theirs, ours ), ( their_scale, our_scale ) ) ):
            root = duel_env.new_battle( seed=battle_seed, attacker="", defender="", hero_att=( heroes[0]["hid"], heroes[0]["hero"] ),
                                        hero_def=( heroes[1]["hid"], heroes[1]["hero"] ), att_scale=scales[0], def_scale=scales[1] )
            if root is None or root.get( "ev" ) != "state":
                continue
            if root.get( "result" ):
                final = root
            else:
                duel_env.snapshot_save( 1 )
                final = duel_env.snapshot_restore( 1, rollout=True )
            fought += 1
            wins += ( final or {} ).get( "result" ) == our_side
    return wins, fought


def final_duel_label( duel_env: BattleEnv, ours: dict, theirs: dict, seed: int, seeds: int = DUEL_SEEDS, steps: int = FINAL_DUEL_STEPS ) -> float:
    """The final duel (user design 2026-10-01): our strongest hero against the rival's, attacking and
    defending with `seeds` battle seeds each. Won every battle: +1, lost every battle: -1. A mixed
    result is no clear victory: forces are added to the side that won less and the duel is fought
    again — a bisection for the army ratio at which the sides are even (half the battles won); the
    label is -log2 of that ratio (ours / theirs) in [-1, 1]: +1 = the rival needs twice our army."""
    wins, fought = duel_wins( duel_env, ours, theirs, seed, seeds, 100, 100 )
    if fought == 0:
        return 0.0
    if wins == fought:
        return 1.0
    if wins == 0:
        return -1.0
    # Search log2(ours / theirs) for the even point: below 0 the rival gets the extra forces.
    low, high = ( -1.0, 0.0 ) if wins * 2 > fought else ( 0.0, 1.0 )
    if wins * 2 == fought:
        return 0.0
    for _ in range( steps ):
        middle = ( low + high ) / 2
        our_scale, their_scale = ( round( 100 * 2 ** middle ), 100 ) if middle >= 0 else ( 100, round( 100 * 2 ** -middle ) )
        wins, fought = duel_wins( duel_env, ours, theirs, seed, seeds, our_scale, their_scale )
        if fought == 0 or wins * 2 == fought:
            low = high = middle
            break
        if wins * 2 > fought:
            high = middle
        else:
            low = middle
    return max( -1.0, min( 1.0, -( low + high ) / 2 ) )


def final_label( duel_env: BattleEnv, results: dict, color: str, seed: int, seeds: int = DUEL_SEEDS ) -> float:
    """The `--label final` score of a player at the end of a game: +1 / -1 for a won / lost game, else
    final_duel_label of our strongest hero against the strongest hero of the strongest active rival
    (by total army strength). The strategic value network predicts the same number."""
    state = str( ( results.get( color ) or {} ).get( "s", "" ) )
    if state == "0":
        return 1.0
    if state == "1":
        return -1.0
    ours = ( results.get( color ) or {} ).get( "top" )
    if ours is None:
        return -1.0  # no hero left: we cannot fight at all
    rivals = {c: r for c, r in active_rivals( results, color ).items() if r.get( "top" )}
    if not rivals:
        return 1.0
    strongest = max( rivals.values(), key=lambda r: r.get( "str", 0 ) )
    return final_duel_label( duel_env, ours, strongest["top"], seed, seeds )


HERO_LOG2_RANGE = 3.0  # hero equivalents are searched for army factors 1/8 .. 8 (log2 in [-3, 3])
HERO_STEPS = 6  # bisection steps of hero_equivalent: a resolution of 6 / 2^6 ~ 0.1 in log2
HERO_LABEL_RULES = ( "hero", "army", "mean", "agree" )


def hero_equivalent( duel_env: BattleEnv, ours: dict, reference: dict, seed: int, seeds: int = DUEL_SEEDS, steps: int = HERO_STEPS ) -> float:
    """Strength of our hero ("top" record: army, skills, artifacts, magic) in units of its own army:
    log2 of the factor every stack of OUR army is scaled by so that it wins half of the duels against
    `reference` (attacking and defending, `seeds` battle seeds each, built-in AI on both sides).
    Lower = stronger: -1 means half of the army is enough. Clipped to [-HERO_LOG2_RANGE, HERO_LOG2_RANGE]."""
    low, high = -HERO_LOG2_RANGE, HERO_LOG2_RANGE
    for _ in range( steps ):
        middle = ( low + high ) / 2
        wins, fought = duel_wins( duel_env, ours, reference, seed, seeds, max( 1, round( 100 * 2 ** middle ) ), 100 )
        if fought == 0:
            return 0.0
        if wins * 2 == fought:
            return middle
        if wins * 2 > fought:
            high = middle
        else:
            low = middle
    return ( low + high ) / 2


def strongest_rival_hero( results: dict, color: str ) -> dict | None:
    """The "top" hero of the strongest active rival (by total army strength) that has a hero."""
    rivals = {c: r for c, r in active_rivals( results, color ).items() if r.get( "top" )}
    return max( rivals.values(), key=lambda r: r.get( "str", 0 ) )["top"] if rivals else None


def equivalent_of( duel_env: BattleEnv, results: dict, color: str, reference: dict | None, seed: int ) -> float:
    """hero_equivalent of our strongest hero at the end of a game, with the game's outcome on the ends
    of the scale: a won game is the strongest (-HERO_LOG2_RANGE), a lost game or no hero left the
    weakest (+HERO_LOG2_RANGE); no rival hero to fight (reference None) counts as the strongest."""
    mine = results.get( color ) or {}
    state = str( mine.get( "s", "" ) )
    if state == "0":
        return -HERO_LOG2_RANGE
    if state == "1" or mine.get( "top" ) is None:
        return HERO_LOG2_RANGE
    if reference is None:
        return -HERO_LOG2_RANGE
    return hero_equivalent( duel_env, mine["top"], reference, seed )


def hero_state( duel_env: BattleEnv, results: dict, color: str, seed: int ) -> dict:
    """The baseline side of the `--label hero` comparison (user design 2026-10-06): the reference hero
    (the strongest hero of the strongest active rival in THIS game), our strongest hero's equivalent
    against it (equivalent_of) and its army strength. A branch under the same luck is measured against
    the same reference, so the rival's luck in the branch does not enter the label."""
    reference = strongest_rival_hero( results, color )
    ours = ( results.get( color ) or {} ).get( "top" )
    return {"ref": reference, "m": equivalent_of( duel_env, results, color, reference, seed ), "str": float( ( ours or {} ).get( "str", 0 ) )}


def hero_components( duel_env: BattleEnv, base: dict, results: dict, color: str, seed: int ) -> tuple[float, float]:
    """(hero, army) differences of a branch against its baseline (hero_state of the same luck), both in
    log2 units, positive = the branch is stronger: the hero = how much less army the branch's strongest
    hero needs against the reference (the baseline's; the branch's own strongest rival hero when the
    baseline had none), the army = log2 of the ratio of the strongest heroes' army strengths
    (Army::GetStrength: troops with the hero's attack/defense, morale, luck), clipped to the range."""
    reference = base["ref"] if base["ref"] is not None else strongest_rival_hero( results, color )
    hero = base["m"] - equivalent_of( duel_env, results, color, reference, seed )
    ours = ( results.get( color ) or {} ).get( "top" )
    strength = float( ( ours or {} ).get( "str", 0 ) )
    army = max( -HERO_LOG2_RANGE, min( HERO_LOG2_RANGE, math.log2( ( strength + 1 ) / ( base["str"] + 1 ) ) ) )
    return hero, army


def combine_hero_parts( hero: float | None, army: float, rule: str ) -> float:
    """One label value from the two parts: "hero", "army", or their mean ("mean"/"agree"; "agree"
    additionally keeps only pairs whose parts point the same way, see label_game)."""
    if rule == "army" or hero is None:
        return army
    if rule == "hero":
        return hero
    return ( hero + army ) / 2


def active_rivals( results: dict, color: str ) -> dict:
    """The other players still in the game (a player that lost has state "1")."""
    return {c: r for c, r in results.items() if c != color and str( r.get( "s", "" ) ) != "1"}


def duel_score( duel_env: BattleEnv, results: dict, color: str, seed: int, seeds: int = DUEL_SEEDS ) -> float:
    """Our strongest hero against the strongest hero of the other players (mean over sides/seeds)."""
    ours = ( results.get( color ) or {} ).get( "top" )
    rivals = [r["top"] for c, r in results.items() if c != color and r.get( "top" )]
    if ours is None:
        return -2.0  # no hero left: we cannot fight at all
    if not rivals:
        return 2.0
    return duel_between( duel_env, ours, max( rivals, key=lambda top: top.get( "str", 0 ) ), seed, seeds )


def duel_all( duel_env: BattleEnv, results: dict, color: str, seed: int, seeds: int = DUEL_SEEDS ) -> float:
    """Our strongest hero against the strongest hero of EVERY active rival; the mean."""
    ours = ( results.get( color ) or {} ).get( "top" )
    rivals = [r["top"] for r in active_rivals( results, color ).values() if r.get( "top" )]
    if ours is None:
        return -2.0
    if not rivals:
        return 2.0
    return sum( duel_between( duel_env, ours, theirs, seed, seeds ) for theirs in rivals ) / len( rivals )


def war_score( duel_env: BattleEnv, state: dict, events: list[dict], color: str, first_day: int, last_day: int, seed: int ) -> float:
    """The `--label war` score of our player at a horizon (user design, see the module doc):
    +2 / -2 for a won / lost game; else the battles against the strongest rival (by total army
    strength) in the window; else duels of our strongest hero against every rival's."""
    state_of_ours = str( ( state.get( color ) or {} ).get( "s", "" ) )
    if state_of_ours == "0":
        return 2.0
    if state_of_ours == "1":
        return -2.0
    rivals = active_rivals( state, color )
    if rivals:
        strongest = max( rivals, key=lambda c: rivals[c].get( "str", 0 ) )
        real = real_hero_battles( events, color, first_day, last_day, strongest )
        if real:
            return sum( real ) / len( real )
    return duel_all( duel_env, state, color, seed )


def horizons_of( args ) -> list[int]:
    return sorted( int( h ) for h in str( args.horizons ).split( "," ) if h )


def branch_scores( args, duel_env, played: tuple, first_day: int, seed: int ) -> list[float]:
    """The label scores of a branch at every horizon (see the module doc): the state at the end
    of day first_day + h comes from the day report of the next day, the last horizon from
    game_end (the branch is played exactly to it)."""
    stats, results, events, reports = played
    if args.label == "final":
        return [final_label( duel_env, results, args.color, seed )]  # the branch is played to the end of the game
    horizons = horizons_of( args )
    scores = []
    for h in horizons:
        last_day = first_day + h
        # A game that ended before the report day (a player won) has its final state in game_end.
        state = results if h == horizons[-1] else reports.get( last_day + 1, results )
        if args.label == "stats":
            scores.append( label_score( player_stats( {"results": list( state.values() )} )[args.color] ) )
            continue
        if args.label == "war":
            scores.append( war_score( duel_env, state, events, args.color, first_day, last_day, seed ) )
            continue
        real = real_hero_battles( events, args.color, first_day, last_day )
        scores.append( sum( real ) / len( real ) if real else duel_score( duel_env, state, args.color, seed ) )
    return scores


def option_index( kind: str, options: list, answer, event: dict ) -> int | None:
    """Index of an answer among the options (None = the built-in AI decided)."""
    if answer is None:
        if kind == "target":
            return 0
        if kind == "army":
            return options.index( 100 )
        if kind == "hire":
            bi = event.get( "bi", -1 )
            return bi if bi >= 0 else len( options ) - 1
        return options.index( BUILTIN )  # build: "let the built-in AI decide"
    return options.index( answer ) if answer in options else None


def builtin_index( kind: str, options: list, event: dict ) -> int | None:
    return option_index( kind, options, None, event )


def salts_of( args ) -> int:
    return max( 1, int( getattr( args, "salts", 1 ) or 1 ) )


def query_scores( args, policy, seed: int, query: dict, others: list[int], duel_env, baselines: dict, until: int, report_days: list[int] ):
    """Scores of the answers `others` to one query against the policy's own answer (0), each the mean
    over `args.salts` replays with different luck (user design 2026-10-03: salt 0 is the plain replay,
    salt k re-seeds the game from the day after the query, FHEROES2_RESEED) of (branch - baseline of
    the same luck); the baseline replays are cached per (day, salt) in `baselines`. Returns
    ({index: mean}, {str(index): [per-salt differences]}, {str(index): {"hero": [...], "army": [...]}}
    (the parts of `--label hero`, else empty)); a failed replay drops that salt only."""
    first_day = query["event"]["t"]
    hero_label = getattr( args, "label", None ) == "hero"

    def replay( branch: PolicyBranch, salt: int ):
        played = play( args, seed, until, branch, report_days, ( first_day + 1, salt ) if salt else None )
        if branch.diverged:
            raise RuntimeError( "replay diverged" )
        if hero_label:
            return played[1]  # game_end results by color
        return branch_scores( args, duel_env, played, first_day, seed )

    scores: dict[int, float] = {}
    salt_scores: dict[str, list[float]] = {}
    salt_parts: dict[str, dict[str, list]] = {}
    for index in others:
        differences = []
        parts: dict[str, list] = {"hero": [], "army": []}
        for salt in range( salts_of( args ) ):
            key = ( first_day, salt )
            if key not in baselines:
                try:
                    baseline = replay( PolicyBranch( policy, args.color ), salt )
                    baselines[key] = hero_state( duel_env, baseline, args.color, seed ) if hero_label else baseline
                except ( TimeoutError, RuntimeError ) as error:
                    print( f"seed {seed} query {query['n']} salt {salt}: baseline failed ({error})", flush=True )
                    baselines[key] = None
            if baselines[key] is None:
                continue
            branch = PolicyBranch( policy, args.color, pick_at=query["n"], answer_index=index, expected=query["event"] )
            try:
                values = replay( branch, salt )
            except ( TimeoutError, RuntimeError ) as error:
                print( f"seed {seed} query {query['n']} salt {salt}: branch {index} failed ({error})", flush=True )
                continue
            if hero_label:
                hero, army = hero_components( duel_env, baselines[key], values, args.color, seed )
                parts["hero"].append( hero )
                parts["army"].append( army )
                differences.append( combine_hero_parts( hero, army, getattr( args, "hero_rule", "mean" ) ) )
                continue
            # The mean over the horizons of (branch - baseline) under the same luck.
            differences.append( sum( b - a for a, b in zip( baselines[key], values ) ) / len( values ) )
        if differences:
            scores[index] = sum( differences ) / len( differences )
            salt_scores[str( index )] = differences
            if hero_label:
                salt_parts[str( index )] = parts
    return scores, salt_scores, salt_parts


def parts_agree( salt_parts: dict, chosen: int, rejected: int ) -> bool:
    """`--hero-rule agree`: the hero part and the army part both prefer `chosen` (means over lucks;
    a missing hero part — no reference hero — does not veto)."""
    for part in ( "hero", "army" ):
        gaps = [c - r for c, r in zip( salt_parts[str( chosen )][part], salt_parts[str( rejected )][part] ) if c is not None and r is not None]
        if gaps and sum( gaps ) <= 0:
            return False
    return True


def value_trajectories( args, branch: PolicyBranch, played: tuple, seed: int, duel_env ) -> list[dict]:
    """Strategic value records of every player of a played game: its days, answers and the final
    label (final_label) — the value network's training data (strategy_value.py format)."""
    _, results, _, _ = played
    end_day = max( ( d.get( "t", 0 ) for days in branch.days.values() for d in days ), default=0 )
    return [dict( branch.trajectory( color ), seed=seed, map=args.map, end_day=end_day, final=final_label( duel_env, results, color, seed ),
                  policy=getattr( args, "policy_tag", None ) )
            for color in sorted( branch.days ) if color in results]


def label_game( args, policy, seed: int, rng: random.Random, duel_env: BattleEnv | None = None, trajectories: list | None = None ) -> list[dict]:
    """DPO pairs of one seeded game; with `trajectories` the base game's value records (every player,
    final label) are appended to it."""
    base = PolicyBranch( policy, args.color )
    base_played = play( args, seed, args.days, base )
    if trajectories is not None and duel_env is not None:
        trajectories.extend( value_trajectories( args, base, base_played, seed, duel_env ) )

    horizons = horizons_of( args )
    eligible = [q for q in base.queries
                if 1 <= q["event"].get( "t", 0 ) <= args.days - horizons[-1] and len( query_options( q["kind"], q["event"] ) ) >= 2]
    chosen = sorted( rng.sample( eligible, min( args.per_game, len( eligible ) ) ), key=lambda q: q["n"] )

    pairs = []
    baselines: dict[tuple[int, int], list[float] | None] = {}  # (day, salt) -> baseline scores (query_scores)
    for query in chosen:
        kind, event = query["kind"], query["event"]
        options = query_options( kind, event )
        own = query["answer_index"]
        if own is None:
            continue
        candidates = {own}
        builtin = builtin_index( kind, options, event )
        if builtin is not None:
            candidates.add( builtin )
        others = [i for i in range( len( options ) ) if i not in candidates]
        candidates.update( rng.sample( others, min( args.random, len( others ) ) ) )

        first_day = event["t"]
        # The final label is taken at the end of the game; the others at the horizons after the query.
        at_end = args.label in ( "final", "hero" )
        until = args.days if at_end else first_day + horizons[-1]
        report_days = [] if at_end else [first_day + h + 1 for h in horizons[:-1]]
        scores, salt_scores, salt_parts = query_scores( args, policy, seed, query, sorted( candidates - {own} ), duel_env, baselines, until,
                                                        report_days )
        scores[own] = 0.0
        salt_scores[str( own )] = [0.0] * len( next( iter( salt_scores.values() ), [] ) )
        if salt_parts:
            # The own answer is the baseline itself: 0 in both parts (None where the hero part was missing).
            size = len( next( iter( salt_parts.values() ) )["hero"] )
            salt_parts[str( own )] = {"hero": [0.0] * size, "army": [0.0] * size}

        best = max( scores, key=scores.get )
        worst = min( scores, key=scores.get )
        # --label hero keeps every query with two scored options: the reliability rule of the DPO step
        # (train_strategy_net --reliable-rule) picks and orients the pair from the per-luck parts.
        if len( scores ) >= 2 and ( salt_parts or scores[best] - scores[worst] > args.margin ):
            if salt_parts and getattr( args, "hero_rule", "mean" ) == "agree" and not parts_agree( salt_parts, best, worst ):
                continue
            pair = {"kind": kind, "event": event, "context": query["context"], "history": query["history"],
                    "chosen": best, "rejected": worst,
                    "own": own, "builtin": builtin, "scores": {str( i ): s for i, s in scores.items()},
                    "seed": seed, "map": args.map, "n": query["n"], "horizons": horizons,
                    "salts": salts_of( args ), "salt_scores": salt_scores}
            if salt_parts:
                pair["label"], pair["hero_rule"], pair["salt_parts"] = "hero", getattr( args, "hero_rule", "mean" ), salt_parts
            pairs.append( pair )
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser( description="DPO pairs from our games against the built-in AI" )
    parser.add_argument( "--model", required=True, help="unified transformer checkpoint (strategic output)" )
    parser.add_argument( "--binary", default="./fheroes2" )
    parser.add_argument( "--map", default="2kings.mp2" )
    parser.add_argument( "--days", type=int, default=45, help="base game length: queries up to days - the last horizon" )
    parser.add_argument( "--horizons", default="21", help="label horizons in days (the label is the mean over them)" )
    parser.add_argument( "--seeds", default="1-10" )
    parser.add_argument( "--color", default="Blue" )
    parser.add_argument( "--per-game", type=int, default=6, help="queries branched per game" )
    parser.add_argument( "--random", type=int, default=1, help="random extra options per query" )
    parser.add_argument( "--label", choices=["final", "hero", "war", "duel", "stats"], default="final",
                         help="hero: our strongest hero's strength (duel equivalent) and army at the end of the game; "
                              "final: the final duel at the end of the game with a handicap search (final_label); "
                              "duel: the week's real hero battle or a duel of the strongest heroes; stats: army/castle score" )
    parser.add_argument( "--margin", type=float, default=0.1, help="minimal label gap of a pair (duel scale ~[-2, 2])" )
    parser.add_argument( "--salts", type=int, default=1, help="replays with different luck per answer; the label is their mean" )
    parser.add_argument( "--hero-rule", choices=HERO_LABEL_RULES, default="mean",
                         help="--label hero: the hero part, the army part, their mean, or the mean with both parts agreeing" )
    parser.add_argument( "--device", default="cpu" )
    parser.add_argument( "--out", required=True )
    args = parser.parse_args()

    policy = NetStrategyPolicy( args.model, args.device )
    # One battle-server engine for all duels (the same map as the games), next to the one game at a time.
    duel_env = BattleEnv( binary=args.binary, map_name=args.map ) if args.label in ( "final", "hero", "war", "duel" ) else None
    total = 0
    t0 = time.time()
    with open( args.out, "w" ) as out:
        for seed in parse_seeds( args.seeds ):
            pairs = label_game( args, policy, seed, random.Random( seed ), duel_env )
            for pair in pairs:
                out.write( json.dumps( pair, separators=( ",", ":" ) ) + "\n" )
            total += len( pairs )
            print( f"seed {seed}: {len( pairs )} pairs (total {total}, {time.time() - t0:.0f}s)", flush=True )
    if duel_env is not None:
        duel_env.close()
    print( f"done: {total} pairs -> {args.out}" )


if __name__ == "__main__":
    main()
