"""Real-game comparison of two arms (user decision 2026-10-24: the only metric that counts is the quality in real
games, and every change is accepted only by paired games on enough fresh seeds).

Each arm is a set of play_vs_builtin.py result files (`--out` directories or json files); a pair there is
"our color with the arm's settings - the all-built-in control game" for one (seed, color). Two arms played on the
same seeds share the control game, so their difference per (map, seed, color) is the arm effect with the luck of
the game paired out. Printed per metric (hero, duel, army strength, castles, outcome): each arm vs the built-in AI
and B - A, means with 95% bootstrap intervals that resample whole seeds (the colors of one seed share a game).

Usage:
    rl/.venv/bin/python rl/game_compare.py --a rl/data/eval1/reference --b rl/data/eval1/net_strategy
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from collections import defaultdict

METRICS = ( "hero", "duel", "str", "k", "outcome" )


def load_arm( paths: list[str] ) -> dict:
    """{(map, seed, color): pair} of every play_vs_builtin result file under `paths`."""
    pairs = {}
    for path in paths:
        files = glob.glob( os.path.join( path, "*.json" ) ) if os.path.isdir( path ) else glob.glob( path )
        for name in files:
            with open( name ) as f:
                result = json.load( f )
            map_name = ( result.get( "summary" ) or result ).get( "map", "" )
            for pair in result.get( "pairs", [] ):
                pairs[( map_name, pair["seed"], pair["color"] )] = pair
    return pairs


def cluster_ci( values: dict, iterations: int = 2000, seed: int = 0 ) -> tuple[float, float, float, int]:
    """(mean, low, high, n) of the per-key values; the bootstrap resamples (map, seed) clusters."""
    clusters = defaultdict( list )
    for ( map_name, game_seed, _ ), value in values.items():
        clusters[( map_name, game_seed )].append( value )
    groups = list( clusters.values() )
    if not groups:
        return 0.0, 0.0, 0.0, 0
    n = sum( len( g ) for g in groups )
    mean = sum( sum( g ) for g in groups ) / n
    rng = random.Random( seed )
    means = []
    for _ in range( iterations ):
        sample = rng.choices( groups, k=len( groups ) )
        means.append( sum( sum( g ) for g in sample ) / sum( len( g ) for g in sample ) )
    means.sort()
    return mean, means[int( 0.025 * iterations )], means[int( 0.975 * iterations ) - 1], n


def compare( a: dict, b: dict ) -> dict:
    """Per metric: arm A vs built-in, arm B vs built-in and B - A on the shared (map, seed, color) keys."""
    keys = sorted( set( a ) & set( b ) )
    report = {"pairs": len( keys )}
    for metric in METRICS:
        report[metric] = {
            "a": cluster_ci( {k: float( a[k].get( metric, 0.0 ) ) for k in keys} ),
            "b": cluster_ci( {k: float( b[k].get( metric, 0.0 ) ) for k in keys} ),
            "b-a": cluster_ci( {k: float( b[k].get( metric, 0.0 ) ) - float( a[k].get( metric, 0.0 ) ) for k in keys} ),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser( description="Paired real-game comparison of two arms" )
    parser.add_argument( "--a", nargs="+", required=True, help="result dirs/files of arm A (the reference)" )
    parser.add_argument( "--b", nargs="+", required=True, help="result dirs/files of arm B" )
    args = parser.parse_args()
    report = compare( load_arm( args.a ), load_arm( args.b ) )
    print( f"{report['pairs']} shared pairs" )
    for metric in METRICS:
        cells = []
        for arm in ( "a", "b", "b-a" ):
            mean, low, high, _ = report[metric][arm]
            cells.append( f"{arm} {mean:+.3f} [{low:+.3f}, {high:+.3f}]" )
        print( f"  {metric:8s} " + "   ".join( cells ) )


if __name__ == "__main__":
    main()
