"""DPO preference pairs from our own games against the built-in AI (on-policy).

One color of a seeded game is played by the unified transformer's strategic output
(strategy_net.NetStrategyPolicy, deterministic argmax), every other player by the built-in AI.
Games are deterministic per (seed, answers), so for a query n of our color (day t) the game can
be replayed exactly up to n and branched:

  baseline branch  — our policy answers everything (= the played game cut at day t + H);
  option branch j  — identical, but query n is answered with option j; our policy continues.

The label of option j is the difference of our player's score between its branch and the
baseline (the policy's own answer has label 0), with the score of a branch played for a week
(H = 7 days) after the query (`--label duel`, default):
  - if our player fought a hero-vs-hero battle on days [t, t + H] (the branch's AI log), the
    battle's result: +1 win / -1 loss + share of own army strength left - share of the enemy's;
  - otherwise a DUEL in the battle server: our strongest hero against the strongest hero of the
    other players (both from game_end "top", save-game serializations with army, skills and
    artifacts), played by the built-in AI on both sides once attacking and once defending; the
    mean of the two battle scores.
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
    az/.venv/bin/python az/strategy_games.py --model az/models/unified_sft.pt --map 2kings.mp2 \\
        --days 30 --seeds 1-20 --color Blue --out az/data/strategy_game_prefs.jsonl
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
from strategy_net import BUILTIN, NetStrategyPolicy, answer_of, query_options  # noqa: E402
from strategy_rollout import same_query  # noqa: E402

KIND_OF_EVENT = {"decision": "target", "build": "build", "hire": "hire", "army": "army"}


class PolicyBranch:
    """Strategic policy for one replay: `policy` answers `color`'s queries, the built-in AI all
    others; the query with global index `pick_at` is answered with `answer` instead (checked
    against `expected`). Records every query of `color` with its context and the answer given."""

    def __init__( self, policy, color: str, pick_at: int | None = None, answer=None, expected: dict | None = None ):
        self.policy = policy
        self.color = color
        self.pick_at = pick_at
        self.answer = answer
        self.expected = expected
        self.count = 0
        self.queries: list[dict] = []
        self.contexts: dict[str, dict] = {}
        self.diverged = False

    def observe_turn( self, turn_context: dict ) -> None:
        self.contexts[turn_context.get( "p" )] = turn_context
        if turn_context.get( "p" ) == self.color:
            self.policy.observe_turn( turn_context )

    def _answer( self, kind: str, event: dict, method ):
        index = self.count
        self.count += 1
        if event.get( "p" ) != self.color:
            return None
        if index == self.pick_at:
            if self.expected is not None and not same_query( event, self.expected ):
                self.diverged = True
            else:
                self.queries.append( {"n": index, "kind": kind, "event": event, "context": self.contexts.get( self.color ), "answer": self.answer} )
                return self.answer
        answer = method( event )
        self.queries.append( {"n": index, "kind": kind, "event": event, "context": self.contexts.get( self.color ), "answer": answer} )
        return answer

    def __call__( self, decision: dict ):
        return self._answer( "target", decision, self.policy )

    def build( self, event: dict ):
        return self._answer( "build", event, self.policy.build )

    def hire( self, event: dict ):
        return self._answer( "hire", event, self.policy.hire )

    def army( self, event: dict ):
        return self._answer( "army", event, self.policy.army )


COLOR_LETTER = {"Blue": "B", "Green": "G", "Red": "R", "Yellow": "Y", "Orange": "O", "Purple": "P"}


def play( args, seed: int, days: int, branch: PolicyBranch ) -> tuple[dict, dict, list[dict]]:
    """One seeded game (one engine, under nice). Returns (stats of the branch's color, game_end
    results by color, the game's AI log events)."""
    handle, log_path = tempfile.mkstemp( prefix="strategy_branch_", suffix=".jsonl" )
    os.close( handle )
    env = StrategyEnv( binary=args.binary, map_name=args.map, days=days, playthroughs=1, seed=seed, ai_log=log_path )
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
    return player_stats( summaries[-1] )[branch.color], results, events


def battle_score( outcome: float, own0: float, own1: float, enemy0: float, enemy1: float ) -> float:
    """outcome (+1 / 0 / -1) + share of own strength left - share of enemy strength left (the
    battle_prefs.move_score scale)."""
    own = own1 / own0 if own0 > 0 else 0.0
    enemy = enemy1 / enemy0 if enemy0 > 0 else 0.0
    return outcome + own - enemy


def real_hero_battles( events: list[dict], color: str, first_day: int, last_day: int ) -> list[float]:
    """Scores of the hero-vs-hero battles of `color` on days [first_day, last_day] (AI log)."""
    letter = COLOR_LETTER[color]
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
            winner = ev.get( "winner" )
            outcome = 1.0 if winner == letter else ( 0.0 if winner == "N" else -1.0 )
            scores.append( battle_score( outcome, ev.get( f"{ours}0", 0 ), ev.get( f"{ours}1", 0 ), ev.get( f"{theirs}0", 0 ), ev.get( f"{theirs}1", 0 ) ) )
    return scores


def duel_score( duel_env: BattleEnv, results: dict, color: str, seed: int ) -> float:
    """Our strongest hero against the strongest hero of the other players, played to the end by
    the built-in AI on both sides, once attacking and once defending; the mean battle score."""
    ours = ( results.get( color ) or {} ).get( "top" )
    rivals = [r["top"] for c, r in results.items() if c != color and r.get( "top" )]
    if ours is None:
        return -2.0  # no hero left: we cannot fight at all
    if not rivals:
        return 2.0
    theirs = max( rivals, key=lambda top: top.get( "str", 0 ) )
    scores = []
    for our_side, heroes in ( ( "att", ( ours, theirs ) ), ( "def", ( theirs, ours ) ) ):
        root = duel_env.new_battle( seed=seed, attacker="", defender="", hero_att=( heroes[0]["hid"], heroes[0]["hero"] ),
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


def branch_score( args, duel_env, stats: dict, results: dict, events: list[dict], first_day: int, last_day: int, seed: int ) -> float:
    """The label score of a branch played to day `last_day` (see the module doc)."""
    if args.label == "stats":
        return label_score( stats )
    real = real_hero_battles( events, args.color, first_day, last_day )
    if real:
        return sum( real ) / len( real )
    return duel_score( duel_env, results, args.color, seed )


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

    eligible = [q for q in base.queries
                if 1 <= q["event"].get( "t", 0 ) <= args.days - args.horizon and len( query_options( q["kind"], q["event"] ) ) >= 2]
    chosen = sorted( rng.sample( eligible, min( args.per_game, len( eligible ) ) ), key=lambda q: q["n"] )

    pairs = []
    baselines: dict[int, float] = {}
    for query in chosen:
        kind, event = query["kind"], query["event"]
        options = query_options( kind, event )
        own = option_index( kind, options, query["answer"], event )
        if own is None:
            continue
        candidates = {own}
        builtin = builtin_index( kind, options, event )
        if builtin is not None:
            candidates.add( builtin )
        others = [i for i in range( len( options ) ) if i not in candidates]
        candidates.update( rng.sample( others, min( args.random, len( others ) ) ) )

        until = event["t"] + args.horizon
        if until not in baselines:
            played = play( args, seed, until, PolicyBranch( policy, args.color ) )
            baselines[until] = branch_score( args, duel_env, *played, event["t"], until, seed )
        scores = {own: 0.0}
        for index in sorted( candidates - {own} ):
            branch = PolicyBranch( policy, args.color, pick_at=query["n"], answer=answer_of( options[index] ), expected=event )
            try:
                played = play( args, seed, until, branch )
            except ( TimeoutError, RuntimeError ) as error:
                print( f"seed {seed} query {query['n']}: branch {index} failed ({error})", flush=True )
                continue
            if branch.diverged:
                print( f"seed {seed} query {query['n']}: replay diverged, skipped", flush=True )
                continue
            scores[index] = branch_score( args, duel_env, *played, event["t"], until, seed ) - baselines[until]

        best = max( scores, key=scores.get )
        worst = min( scores, key=scores.get )
        if len( scores ) >= 2 and scores[best] - scores[worst] > args.margin:
            pairs.append( {"kind": kind, "event": event, "context": query["context"], "chosen": best, "rejected": worst,
                           "own": own, "builtin": builtin, "scores": {str( i ): s for i, s in scores.items()},
                           "seed": seed, "map": args.map, "n": query["n"], "horizon": args.horizon} )
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser( description="DPO pairs from our games against the built-in AI" )
    parser.add_argument( "--model", required=True, help="unified transformer checkpoint (strategic output)" )
    parser.add_argument( "--binary", default="./fheroes2" )
    parser.add_argument( "--map", default="2kings.mp2" )
    parser.add_argument( "--days", type=int, default=30 )
    parser.add_argument( "--horizon", type=int, default=7 )
    parser.add_argument( "--seeds", default="1-10" )
    parser.add_argument( "--color", default="Blue" )
    parser.add_argument( "--per-game", type=int, default=6, help="queries branched per game" )
    parser.add_argument( "--random", type=int, default=1, help="random extra options per query" )
    parser.add_argument( "--label", choices=["duel", "stats"], default="duel",
                         help="duel: the week's real hero battle or a duel of the strongest heroes; stats: army/castle score" )
    parser.add_argument( "--margin", type=float, default=0.1, help="minimal label gap of a pair (duel scale ~[-2, 2])" )
    parser.add_argument( "--device", default="cpu" )
    parser.add_argument( "--out", required=True )
    args = parser.parse_args()

    policy = NetStrategyPolicy( args.model, args.device )
    # One battle-server engine for all duels (the same map as the games), next to the one game at a time.
    duel_env = BattleEnv( binary=args.binary, map_name=args.map ) if args.label == "duel" else None
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
