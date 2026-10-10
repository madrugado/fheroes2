"""Plan racing: tests rule candidates (plan keys) against a reference plan on fresh seeds (step 2 of the rule
mechanism, after rule_mining.py).

Every round plays one block of new seeds for every candidate still in the race (play_vs_builtin.py: paired games
against the built-in AI for every color of the map) and compares each candidate with the reference
paired by (seed, color) on all seeds so far: a candidate whose metric difference has its 95% bootstrap CI below 0
is dropped. The race ends when the rounds are used up or only the reference is left. Racing selects on noisy
estimates too: confirm a winner on fresh seeds before adopting it (as rl/data/confirm did for the factorial).

  rl/.venv/bin/python rl/plan_race.py --map Battlefi.mp2 --days 30 --first-seed 261 --block 10 --rounds 6 \\
      --reference "base=champion=1,split_singles=1" \\
      --candidate "dwell=champion=1,split_singles=1,dwellings_first=1" --out rl/data/race
"""

import argparse
import glob
import json
import os
import random
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname( os.path.abspath( __file__ ) )


def parse_candidate( text: str ) -> tuple[str, str]:
    name, _, plan = text.partition( "=" )
    if not name or not plan:
        raise SystemExit( f"candidate must be NAME=PLAN: {text}" )
    return name, plan


def play_block( name: str, plan: str, seeds: tuple[int, int], args ) -> None:
    tag = f"_{name}_{seeds[0]}-{seeds[1]}"
    if glob.glob( os.path.join( args.out, f"*{tag}.json" ) ):
        return  # already played (a resumed race)
    log = open( os.path.join( args.out, f"{name}_{seeds[0]}-{seeds[1]}.log" ), "w" )
    subprocess.run( [sys.executable, os.path.join( HERE, "play_vs_builtin.py" ), "--map", args.map, "--days", str( args.days ),
                     "--seeds", f"{seeds[0]}-{seeds[1]}", "--strategy", "rule", "--rule", "", "--plan", plan, "--battle", "planner",
                     "--duel", "--tag", tag, "--out", args.out], stdout=log, stderr=subprocess.STDOUT, check=False )


def load_pairs( out: str, name: str ) -> dict:
    pairs = {}
    for path in glob.glob( os.path.join( out, f"*_{name}_*.json" ) ):
        for pair in json.load( open( path ) ).get( "pairs", [] ):
            pairs[( pair["seed"], pair["color"] )] = pair
    return pairs


def metric_of( pair: dict, metric: str ) -> float:
    return float( pair.get( metric, 0.0 ) )


def bootstrap( values: list[float], n: int = 4000 ) -> tuple[float, float]:
    rng = random.Random( 0 )
    means = sorted( statistics.fmean( rng.choices( values, k=len( values ) ) ) for _ in range( n ) )
    return means[int( 0.025 * n )], means[int( 0.975 * n )]


def main() -> None:
    parser = argparse.ArgumentParser( description="Race plan candidates against a reference plan on fresh seeds" )
    parser.add_argument( "--map", required=True )
    parser.add_argument( "--days", type=int, default=30 )
    parser.add_argument( "--first-seed", type=int, required=True )
    parser.add_argument( "--block", type=int, default=10, help="seeds per round" )
    parser.add_argument( "--rounds", type=int, default=6 )
    parser.add_argument( "--jobs", type=int, default=4 )
    parser.add_argument( "--metric", default="duel", help="pair field: duel (outcast), k (castles), str, ..." )
    parser.add_argument( "--reference", required=True, help="NAME=PLAN" )
    parser.add_argument( "--candidate", action="append", default=[], help="NAME=PLAN (repeatable)" )
    parser.add_argument( "--out", required=True )
    args = parser.parse_args()

    os.makedirs( args.out, exist_ok=True )
    reference = parse_candidate( args.reference )
    alive = dict( [reference] + [parse_candidate( c ) for c in args.candidate] )
    dropped: dict[str, str] = {}
    progress = open( os.path.join( args.out, "race.jsonl" ), "a" )

    for round_index in range( args.rounds ):
        seeds = ( args.first_seed + round_index * args.block, args.first_seed + ( round_index + 1 ) * args.block - 1 )
        with ThreadPoolExecutor( max_workers=args.jobs ) as pool:
            list( pool.map( lambda item: play_block( item[0], item[1], seeds, args ), alive.items() ) )

        base = load_pairs( args.out, reference[0] )
        report = {"round": round_index + 1, "seeds": [args.first_seed, seeds[1]], "metric": args.metric,
                  "reference": statistics.fmean( metric_of( p, args.metric ) for p in base.values() ) if base else None, "candidates": {}}
        for name in list( alive ):
            if name == reference[0]:
                continue
            pairs = load_pairs( args.out, name )
            common = sorted( set( pairs ) & set( base ) )
            if len( common ) < 10:
                continue
            diff = [metric_of( pairs[k], args.metric ) - metric_of( base[k], args.metric ) for k in common]
            low, high = bootstrap( diff )
            report["candidates"][name] = {"n": len( common ), "vs_builtin": statistics.fmean( metric_of( pairs[k], args.metric ) for k in common ),
                                          "diff": statistics.fmean( diff ), "low": low, "high": high}
            if high < 0:
                dropped[name] = f"round {round_index + 1}: {statistics.fmean( diff ):+.3f} [{low:+.3f}, {high:+.3f}]"
                del alive[name]
        report["dropped"] = dict( dropped )
        progress.write( json.dumps( report ) + "\n" )
        progress.flush()
        print( json.dumps( report ), flush=True )
        if len( alive ) == 1:
            break

    print( "alive:", sorted( alive ), "| clearly better than the reference (confirm on fresh seeds):",
           sorted( n for n, c in report["candidates"].items() if c["low"] > 0 ) )


if __name__ == "__main__":
    main()
