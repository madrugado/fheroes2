"""Main effects and pairwise interactions of plan rules from a factorial sweep (user remark
2026-10-09: "some rules only work together").

Every cell of the sweep is one combination of binary plan keys, run through play_vs_builtin.py with
`--tag _<cell>_<seeds>`, where <cell> is "f" followed by one letter per enabled key (see FACTORS).
Each pair (seed, color) gives the treatment-minus-control differences of the game_end metrics. A
least-squares fit per metric, with factors coded -1/+1, of

    metric ~ 1 + sum_i b_i x_i + sum_{i<j} b_ij x_i x_j

gives each rule's main effect (2 b_i = the average change from switching the rule on) and each
interaction (2 b_ij = how much the effect of rule i changes when rule j is on). Confidence intervals
by a bootstrap over seeds (all cells of a seed move together: the same map and luck).

    rl/.venv/bin/python rl/plan_factorial.py --data rl/data/factorial
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import random
import re
from collections import defaultdict

import numpy as np

FACTORS = {"S": "champion_skills", "R": "secondaries", "M": "secondary_min=2", "E": "secondary_skills", "G": "garrison_slowest"}
METRICS = ( "duel", "str", "k", "outcome", "h", "g" )


def load( directory: str ) -> list[dict]:
    """One row per pair: the cell's factor levels (-1/+1), the seed and the metrics."""
    rows = []
    for path in glob.glob( os.path.join( directory, "*.json" ) ):
        match = re.search( r"_f([A-Z]*)_\d+-\d+\.json$", os.path.basename( path ) )
        if match is None:
            continue
        cell = match.group( 1 )
        levels = {letter: ( 1.0 if letter in cell else -1.0 ) for letter in FACTORS}
        with open( path ) as f:
            for pair in json.load( f )["pairs"]:
                rows.append( {"cell": cell, "levels": levels, "seed": pair["seed"],
                              **{metric: float( pair.get( metric ) or 0.0 ) for metric in METRICS}} )
    return rows


def design( rows: list[dict] ) -> tuple[np.ndarray, list[str]]:
    letters = list( FACTORS )
    names = ["mean"] + letters + [a + b for a, b in itertools.combinations( letters, 2 )]
    matrix = []
    for row in rows:
        x = [row["levels"][letter] for letter in letters]
        matrix.append( [1.0] + x + [x[i] * x[j] for i, j in itertools.combinations( range( len( letters ) ), 2 )] )
    return np.array( matrix ), names


def fit( rows: list[dict], metric: str ) -> np.ndarray:
    matrix, _ = design( rows )
    target = np.array( [row[metric] for row in rows] )
    coefficients, *_ = np.linalg.lstsq( matrix, target, rcond=None )
    return coefficients


def analyze( rows: list[dict], metric: str, draws: int = 1000 ) -> list[tuple[str, float, float, float]]:
    """(term, effect, ci low, ci high): effect = 2 x coefficient (the change from -1 to +1); the
    "mean" term is the average treatment-minus-control difference over all cells."""
    _, names = design( rows )
    point = fit( rows, metric )
    by_seed = defaultdict( list )
    for row in rows:
        by_seed[row["seed"]].append( row )
    seeds = sorted( by_seed )
    rng = random.Random( 0 )
    samples = []
    for _ in range( draws ):
        sample = [row for seed in rng.choices( seeds, k=len( seeds ) ) for row in by_seed[seed]]
        samples.append( fit( sample, metric ) )
    samples = np.array( samples )
    result = []
    for index, name in enumerate( names ):
        scale = 1.0 if name == "mean" else 2.0
        low, high = np.percentile( samples[:, index] * scale, [2.5, 97.5] )
        result.append( ( name, point[index] * scale, low, high ) )
    return result


def main() -> None:
    parser = argparse.ArgumentParser( description=__doc__.splitlines()[0] )
    parser.add_argument( "--data", default="rl/data/factorial" )
    parser.add_argument( "--metrics", default="duel,str,k,outcome" )
    parser.add_argument( "--draws", type=int, default=1000 )
    args = parser.parse_args()

    rows = load( args.data )
    cells = sorted( {row["cell"] for row in rows} )
    print( f"{len( rows )} pairs in {len( cells )} cells, {len( {row['seed'] for row in rows} )} seeds" )
    print( "factors: " + ", ".join( f"{letter} = {name}" for letter, name in FACTORS.items() ) )
    for metric in args.metrics.split( "," ):
        print( f"\n{metric}: effect [95% CI] (* = the interval excludes 0)" )
        for name, effect, low, high in analyze( rows, metric, args.draws ):
            mark = "*" if low > 0 or high < 0 else " "
            print( f"  {name:5s} {effect:+9.3f} [{low:+9.3f}, {high:+9.3f}] {mark}" )
    # Cell means of the final duel, best first.
    means = defaultdict( list )
    for row in rows:
        means[row["cell"]].append( row["duel"] )
    print( "\ncells by mean duel difference:" )
    for cell, values in sorted( means.items(), key=lambda kv: -np.mean( kv[1] ) ):
        print( f"  f{cell:6s} n {len( values ):4d} duel {np.mean( values ):+.3f}" )


if __name__ == "__main__":
    main()
