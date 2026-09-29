"""DPO preference pairs from our own games against the built-in AI (on-policy).

One color of a seeded game is played by the unified transformer's strategic output
(strategy_net.NetStrategyPolicy, deterministic argmax), every other player by the built-in AI.
Games are deterministic per (seed, answers), so for a query n of our color (day t) the game can
be replayed exactly up to n and branched:

  baseline branch  — our policy answers everything (= the played game cut at day t + H);
  option branch j  — identical, but query n is answered with option j; our policy continues.

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

    def observe_turn( self, turn_context: dict ) -> None:
        self.contexts[turn_context.get( "p" )] = turn_context
        if turn_context.get( "p" ) == self.color:
            self.policy.observe_turn( turn_context )

    def _answer( self, kind: str, event: dict ):
        index = self.count
        self.count += 1
        if event.get( "p" ) != self.color:
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
        answer = None if choice is None else answer_of( options[choice] )
        if kind == "target" and answer is not None and answer is ( event.get( "cands" ) or [None] )[0]:
            answer = None  # the top candidate is the built-in choice
        self.queries.append( {"n": index, "kind": kind, "event": event, "context": self.contexts.get( self.color ),
                              "answer": answer, "answer_index": choice, "history": snapshot} )
        return answer

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


DUEL_SEEDS = 3  # every duel orientation is fought with this many battle seeds (user request: less battle luck)


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


def label_game( args, policy, seed: int, rng: random.Random, duel_env: BattleEnv | None = None ) -> list[dict]:
    base = PolicyBranch( policy, args.color )
    play( args, seed, args.days, base )

    horizons = horizons_of( args )
    eligible = [q for q in base.queries
                if 1 <= q["event"].get( "t", 0 ) <= args.days - horizons[-1] and len( query_options( q["kind"], q["event"] ) ) >= 2]
    chosen = sorted( rng.sample( eligible, min( args.per_game, len( eligible ) ) ), key=lambda q: q["n"] )

    pairs = []
    baselines: dict[int, list[float]] = {}
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
        until = first_day + horizons[-1]
        report_days = [first_day + h + 1 for h in horizons[:-1]]
        if first_day not in baselines:
            try:
                played = play( args, seed, until, PolicyBranch( policy, args.color ), report_days )
                baselines[first_day] = branch_scores( args, duel_env, played, first_day, seed )
            except ( TimeoutError, RuntimeError ) as error:
                print( f"seed {seed} query {query['n']}: baseline failed ({error}), skipped", flush=True )
                continue
        scores = {own: 0.0}
        for index in sorted( candidates - {own} ):
            branch = PolicyBranch( policy, args.color, pick_at=query["n"], answer_index=index, expected=event )
            try:
                played = play( args, seed, until, branch, report_days )
            except ( TimeoutError, RuntimeError ) as error:
                print( f"seed {seed} query {query['n']}: branch {index} failed ({error})", flush=True )
                continue
            if branch.diverged:
                print( f"seed {seed} query {query['n']}: replay diverged, skipped", flush=True )
                continue
            branch_values = branch_scores( args, duel_env, played, first_day, seed )
            # The mean over the horizons of (branch - baseline).
            scores[index] = sum( b - a for a, b in zip( baselines[first_day], branch_values ) ) / len( horizons )

        best = max( scores, key=scores.get )
        worst = min( scores, key=scores.get )
        if len( scores ) >= 2 and scores[best] - scores[worst] > args.margin:
            pairs.append( {"kind": kind, "event": event, "context": query["context"], "history": query["history"],
                           "chosen": best, "rejected": worst,
                           "own": own, "builtin": builtin, "scores": {str( i ): s for i, s in scores.items()},
                           "seed": seed, "map": args.map, "n": query["n"], "horizons": horizons} )
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
    parser.add_argument( "--label", choices=["war", "duel", "stats"], default="war",
                         help="duel: the week's real hero battle or a duel of the strongest heroes; stats: army/castle score" )
    parser.add_argument( "--margin", type=float, default=0.1, help="minimal label gap of a pair (duel scale ~[-2, 2])" )
    parser.add_argument( "--device", default="cpu" )
    parser.add_argument( "--out", required=True )
    args = parser.parse_args()

    policy = NetStrategyPolicy( args.model, args.device )
    # One battle-server engine for all duels (the same map as the games), next to the one game at a time.
    duel_env = BattleEnv( binary=args.binary, map_name=args.map ) if args.label in ( "war", "duel" ) else None
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
