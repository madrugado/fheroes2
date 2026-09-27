"""Counterfactual rollout labels for strategic hero-target decisions.

The built-in AI always takes the top-value candidate, so plain games never show what another
choice would have led to. Seeded playthroughs are deterministic (FHEROES2_AUTO_PLAYTEST_SEED),
which makes exact counterfactuals cheap:

1. a base game with the given seed where everybody plays the built-in AI enumerates the
   decisions (the n-th decision event is the same in every replay of that seed);
2. for a sampled decision n of color p on day t, the BASELINE branch replays the game to day
   t + H with the built-in choices, and each ALTERNATIVE branch replays it identically but picks
   candidate j at decision n (built-in AI afterwards);
3. the label of candidate j is the difference of p's kingdom stats (castles, heroes, army
   strength, gold, outcome) at day t + H between the alternative and the baseline branch.

The day limit only ends the game (the AI does not know it), so a branch cut at t + H is an exact
prefix of a longer game (verified: identical AI logs). Every alternative branch re-checks that
decision n is the expected one before picking (determinism guard).

Usage:
    az/.venv/bin/python az/strategy_rollout.py --map Battlefi.mp2 --seeds 101-120 --per-seed 25 --top 4 --horizon 7 --jobs 2
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

from strategy_bench import player_stats  # noqa: E402
from strategy_env import StrategyEnv  # noqa: E402

STAT_KEYS = ( "outcome", "k", "h", "str", "g" )


class Branch:
    """Policy for one seeded game: built-in choices everywhere, except an optional pick of
    `pick_tile` at the global decision index `pick_at`. Records every decision together with the
    last turn_context of the deciding color."""

    def __init__( self, pick_at: int | None = None, pick_tile: int | None = None, expected: dict | None = None ):
        self.pick_at = pick_at
        self.pick_tile = pick_tile
        self.expected = expected
        self.decisions: list[dict] = []
        self.contexts: dict[str, dict] = {}
        self.diverged = False
        self.picked = False

    def observe_turn( self, turn_context: dict ) -> None:
        self.contexts[turn_context.get( "p" )] = turn_context

    def __call__( self, decision: dict ) -> dict | None:
        index = len( self.decisions )
        self.decisions.append( {"n": index, "decision": decision, "context": self.contexts.get( decision.get( "p" ) )} )

        if index != self.pick_at:
            return None

        if self.expected is not None and not same_decision( decision, self.expected ):
            self.diverged = True
            return None

        for cand in decision.get( "cands" ) or []:
            if cand["i"] == self.pick_tile:
                self.picked = True
                return cand
        self.diverged = True
        return None


def same_decision( a: dict, b: dict ) -> bool:
    return all( a.get( key ) == b.get( key ) for key in ( "t", "p", "h", "from", "cands" ) )


def play( binary: str, map_name: str, days: int, seed: int, branch: Branch ) -> dict:
    """Plays one seeded game with the branch policy; returns the game_end event."""
    env = StrategyEnv( binary=binary, map_name=map_name, days=days, playthroughs=1, seed=seed )
    try:
        summaries = env.run( branch )
    finally:
        env.close()
    if not summaries:
        raise RuntimeError( f"seed {seed}, days {days}: no game_end" )
    return summaries[-1]


def select_decisions( decisions: list[dict], per_seed: int, max_day: int, rng: random.Random ) -> list[dict]:
    """Decisions worth labeling: at least two candidates, early enough for the horizon."""
    eligible = [d for d in decisions if len( d["decision"].get( "cands" ) or [] ) >= 2 and 1 <= d["decision"].get( "t", 0 ) <= max_day]
    return sorted( rng.sample( eligible, min( per_seed, len( eligible ) ) ), key=lambda d: d["n"] )


def stat_delta( alt: dict, base: dict ) -> dict:
    return {key: alt[key] - base[key] for key in STAT_KEYS}


def label_seed( pool: ThreadPoolExecutor, args, seed: int ) -> list[dict]:
    rng = random.Random( seed )
    max_day = args.base_days - args.horizon

    base = Branch()
    play( args.binary, args.map, args.base_days, seed, base )
    chosen = select_decisions( base.decisions, args.per_seed, max_day, rng )

    # Baselines are shared by all decisions of the same day.
    days_needed = sorted( {d["decision"]["t"] + args.horizon for d in chosen} )
    baselines = dict( zip( days_needed, pool.map( lambda days: play( args.binary, args.map, days, seed, Branch() ), days_needed ) ) )

    jobs = []
    for item in chosen:
        decision = item["decision"]
        for j, cand in enumerate( decision["cands"][: args.top] ):
            if j == 0:
                continue  # the top-value candidate IS the built-in choice: the baseline branch
            jobs.append( ( item, j, cand ) )

    def run_alternative( job ):
        item, j, cand = job
        days = item["decision"]["t"] + args.horizon
        branch = Branch( pick_at=item["n"], pick_tile=cand["i"], expected=item["decision"] )
        try:
            game_end = play( args.binary, args.map, days, seed, branch )
        except ( TimeoutError, RuntimeError ) as error:
            # A stuck branch must not kill a long labeling run; report it for investigation.
            print( f"seed {seed}: decision {item['n']} candidate {j} (tile {cand['i']}, days {days}) failed: {error}", flush=True )
            return job, branch, None
        return job, branch, game_end

    records: list[dict] = []
    for item in chosen:
        decision = item["decision"]
        base_stats = player_stats( baselines[decision["t"] + args.horizon] )[decision["p"]]
        records.append( make_record( seed, args, item, 0, decision["cands"][0], {key: 0 for key in STAT_KEYS}, base_stats ) )

    for ( item, j, cand ), branch, game_end in pool.map( run_alternative, jobs ):
        if game_end is None:
            continue
        if branch.diverged or not branch.picked:
            print( f"seed {seed}: decision {item['n']} diverged in replay; skipped", flush=True )
            continue
        decision = item["decision"]
        base_stats = player_stats( baselines[decision["t"] + args.horizon] )[decision["p"]]
        alt_stats = player_stats( game_end )[decision["p"]]
        records.append( make_record( seed, args, item, j, cand, stat_delta( alt_stats, base_stats ), base_stats ) )

    return records


def make_record( seed: int, args, item: dict, j: int, cand: dict, delta: dict, base_stats: dict ) -> dict:
    return {
        "seed": seed,
        "map": args.map,
        "horizon": args.horizon,
        "n": item["n"],
        "j": j,
        "decision": item["decision"],
        "context": item["context"],
        "cand": cand,
        "delta": delta,
        "base": base_stats,
    }


def parse_seeds( text: str ) -> list[int]:
    if "-" in text:
        first, last = text.split( "-" )
        return list( range( int( first ), int( last ) + 1 ) )
    return [int( s ) for s in text.split( "," )]


def main() -> None:
    parser = argparse.ArgumentParser( description="Counterfactual rollout labels for strategic decisions" )
    parser.add_argument( "--binary", type=str, default="./fheroes2" )
    parser.add_argument( "--map", type=str, default="Battlefi.mp2" )
    parser.add_argument( "--seeds", type=str, default="101-120", help="range 'a-b' or list 'a,b,c'" )
    parser.add_argument( "--per-seed", type=int, default=25, help="decisions labeled per seed" )
    parser.add_argument( "--top", type=int, default=4, help="candidates per decision (by built-in value)" )
    parser.add_argument( "--horizon", type=int, default=7, help="days simulated after the decision" )
    parser.add_argument( "--base-days", type=int, default=21, help="decisions are sampled up to base-days - horizon" )
    parser.add_argument( "--jobs", type=int, default=2, help="engine processes in parallel (keep low on a laptop)" )
    parser.add_argument( "--out", type=str, default="az/data" )
    args = parser.parse_args()

    os.makedirs( args.out, exist_ok=True )
    out_path = os.path.join( args.out, f"strategy_rollouts_{os.path.splitext( args.map )[0]}_h{args.horizon}.jsonl" )

    t0 = time.time()
    total = 0
    with ThreadPoolExecutor( max_workers=args.jobs ) as pool, open( out_path, "a" ) as out:
        for seed in parse_seeds( args.seeds ):
            records = label_seed( pool, args, seed )
            for record in records:
                out.write( json.dumps( record ) + "\n" )
            out.flush()
            total += len( records )
            print( f"seed {seed}: {len( records )} records ({time.time() - t0:.0f}s)", flush=True )

    print( f"done: {total} records -> {out_path} in {time.time() - t0:.0f}s" )


if __name__ == "__main__":
    main()
