"""Plan racing: tests rule candidates (plan keys) against a reference plan on fresh seeds (step 2 of the rule
mechanism, after rule_mining.py), on one or several maps at once.

Every round plays one block of new seeds per map for every candidate still in the race (play_vs_builtin.py:
paired games against the built-in AI for every color of the map) and compares each candidate with the reference
paired by (map, seed, color) on all seeds so far. The combined score is the MEAN OF THE PER-MAP MEANS (every map
weighs the same, whatever its number of players); its 95% CI comes from a bootstrap that resamples seeds within
each map (the colors of one seed share the game). A candidate whose combined CI is below 0 is dropped. Without
--reference the candidates are compared with the built-in AI itself (the pair metrics already are "plan - built-in").
The race ends when the rounds are used up or only the reference is left. Racing selects on noisy estimates too:
confirm a winner on fresh seeds before adopting it.

  rl/.venv/bin/python rl/plan_race.py --maps Battlefi.mp2:10,Thechaos.mp2:10,2kings.mp2:30 --days 30 \\
      --first-seed 801 --rounds 4 --reference "core=champion=1,split_singles=1,max_heroes=3" \\
      --candidate "xp=champion=1,split_singles=1,max_heroes=3,champion_xp=1" --out rl/data/race
"""

import argparse
import glob
import json
import os
import random
import statistics
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname( os.path.abspath( __file__ ) )


def parse_candidate( text: str ) -> tuple[str, str]:
    name, _, plan = text.partition( "=" )
    if not name or not plan:
        raise SystemExit( f"candidate must be NAME=PLAN: {text}" )
    return name, plan


def parse_maps( text: str, block: int ) -> list[tuple[str, int]]:
    """"A.mp2:10,B.mp2" -> [(map, seeds per round)]; a map without a count uses --block."""
    maps = []
    for item in text.split( "," ):
        name, _, count = item.partition( ":" )
        maps.append( ( name, int( count ) if count else block ) )
    return maps


def map_stem( map_name: str ) -> str:
    return os.path.splitext( map_name )[0]


def play_block( name: str, plan: str, map_name: str, seeds: tuple[int, int], args ) -> None:
    tag = f"_{name}_{map_stem( map_name )}_{seeds[0]}-{seeds[1]}"
    if glob.glob( os.path.join( args.out, f"*{tag}.json" ) ):
        return  # already played (a resumed race)
    log = open( os.path.join( args.out, f"{name}_{map_stem( map_name )}_{seeds[0]}-{seeds[1]}.log" ), "w" )
    subprocess.run( [sys.executable, os.path.join( HERE, "play_vs_builtin.py" ), "--map", map_name, "--days", str( args.days ),
                     "--seeds", f"{seeds[0]}-{seeds[1]}", "--strategy", "rule", "--rule", "", "--plan", plan, "--battle", "planner",
                     "--duel", *( ["--hero"] if args.metric == "hero" else [] ), "--tag", tag, "--out", args.out],
                    stdout=log, stderr=subprocess.STDOUT, check=False )


def load_pairs( out: str, name: str, maps: list[tuple[str, int]] ) -> dict:
    """(map stem, seed, color) -> pair."""
    pairs = {}
    for map_name, _ in maps:
        stem = map_stem( map_name )
        for path in glob.glob( os.path.join( out, f"*_{name}_{stem}_*.json" ) ):
            for pair in json.load( open( path ) ).get( "pairs", [] ):
                pairs[( stem, pair["seed"], pair["color"] )] = pair
    return pairs


def metric_of( pair: dict, metric: str ) -> float:
    return float( pair.get( metric, 0.0 ) )


def combined( diffs: dict, n: int = 2000 ) -> dict:
    """diffs: (map, seed, color) -> value. Mean of the per-map means and its CI (seeds resampled within maps), plus
    every map's own mean and CI."""
    by_map: dict[str, dict[int, list[float]]] = defaultdict( lambda: defaultdict( list ) )
    for ( stem, seed, _ ), value in diffs.items():
        by_map[stem][seed].append( value )
    rng = random.Random( 0 )

    def mean_of( groups: dict[int, list[float]], seeds: list[int] ) -> float:
        values = [v for seed in seeds for v in groups[seed]]
        return statistics.fmean( values ) if values else 0.0

    point = {stem: mean_of( groups, list( groups ) ) for stem, groups in by_map.items()}
    samples = defaultdict( list )
    for _ in range( n ):
        means = {}
        for stem, groups in by_map.items():
            seeds = list( groups )
            means[stem] = mean_of( groups, rng.choices( seeds, k=len( seeds ) ) )
            samples[stem].append( means[stem] )
        samples["all"].append( statistics.fmean( means.values() ) )
    result = {}
    for key, values in samples.items():
        values.sort()
        result[key] = {"mean": statistics.fmean( point.values() ) if key == "all" else point[key],
                       "low": values[int( 0.025 * n )], "high": values[int( 0.975 * n ) - 1]}
    result["n"] = len( diffs )
    return result


def main() -> None:
    parser = argparse.ArgumentParser( description="Race plan candidates against a reference plan on fresh seeds, on one or several maps" )
    parser.add_argument( "--maps", default="", help="MAP[:seeds per round],... (equal weight per map in the combined score)" )
    parser.add_argument( "--map", default="", help="one map (the older form of --maps)" )
    parser.add_argument( "--days", type=int, default=30 )
    parser.add_argument( "--first-seed", type=int, required=True )
    parser.add_argument( "--block", type=int, default=10, help="seeds per round for a map without its own count" )
    parser.add_argument( "--rounds", type=int, default=6 )
    parser.add_argument( "--jobs", type=int, default=4 )
    parser.add_argument( "--metric", default="duel", help="pair field: duel (outcast), hero (continuous hero label, play_vs_builtin "
                         "--hero), k (castles), str, ..." )
    parser.add_argument( "--reference", default="", help="NAME=PLAN; absent = compare with the built-in AI" )
    parser.add_argument( "--candidate", action="append", default=[], help="NAME=PLAN (repeatable)" )
    parser.add_argument( "--out", required=True )
    args = parser.parse_args()

    maps = parse_maps( args.maps or args.map, args.block )
    os.makedirs( args.out, exist_ok=True )
    reference = parse_candidate( args.reference ) if args.reference else None
    alive = dict( ( [reference] if reference else [] ) + [parse_candidate( c ) for c in args.candidate] )
    dropped: dict[str, str] = {}
    progress = open( os.path.join( args.out, "race.jsonl" ), "a" )
    starts = {map_name: args.first_seed for map_name, _ in maps}
    report: dict = {"candidates": {}}

    for round_index in range( args.rounds ):
        jobs = []
        for map_name, block in maps:
            seeds = ( starts[map_name], starts[map_name] + block - 1 )
            starts[map_name] += block
            jobs += [( name, plan, map_name, seeds ) for name, plan in alive.items()]
        with ThreadPoolExecutor( max_workers=args.jobs ) as pool:
            list( pool.map( lambda job: play_block( *job, args ), jobs ) )

        base = load_pairs( args.out, reference[0], maps ) if reference else None
        report = {"round": round_index + 1, "metric": args.metric, "maps": [m for m, _ in maps], "candidates": {}}
        if base:
            report["reference"] = combined( {k: metric_of( p, args.metric ) for k, p in base.items()} )
        for name in list( alive ):
            if reference and name == reference[0]:
                continue
            pairs = load_pairs( args.out, name, maps )
            keys = sorted( set( pairs ) & set( base ) ) if base is not None else sorted( pairs )
            if len( keys ) < 10:
                continue
            diffs = {k: metric_of( pairs[k], args.metric ) - ( metric_of( base[k], args.metric ) if base is not None else 0.0 ) for k in keys}
            report["candidates"][name] = combined( diffs )
            total = report["candidates"][name]["all"]
            if total["high"] < 0:
                dropped[name] = f"round {round_index + 1}: {total['mean']:+.3f} [{total['low']:+.3f}, {total['high']:+.3f}]"
                del alive[name]
        report["dropped"] = dict( dropped )
        progress.write( json.dumps( report ) + "\n" )
        progress.flush()
        print( json.dumps( report ), flush=True )
        if reference and len( alive ) == 1:
            break

    print( "alive:", sorted( alive ), "| clearly better than the", "reference" if reference else "built-in AI",
           "(confirm on fresh seeds):", sorted( n for n, c in report["candidates"].items() if c["all"]["low"] > 0 ) )


if __name__ == "__main__":
    main()
