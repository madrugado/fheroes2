"""Is there a learnable signal in whole-game GRPO groups (strategy_game_grpo.py collect)? A check without the net.

Per game: how many answers of each query kind left the built-in answer (the collecting model imitates the plan, so the
built-in answer is its greedy one), split for hero targets by the menu position taken. Within a group (same seed and
color, 8 games) the counts and the reward are centred, and the reward is regressed on the counts: the effect of one
more deviation of that kind on the end-of-game hero equivalent (log2 units, + = stronger). Seed-cluster bootstrap CIs;
also the same fit on held-out halves of the groups (does the sign repeat?).

  rl/.venv/bin/python rl/game_grpo_signal.py --data 'rl/data/game_grpo_v2_c*.jsonl'
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

import numpy as np

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

from strategy_games import builtin_index  # noqa: E402
from strategy_net import query_options  # noqa: E402

KINDS = ( "target", "build", "hire", "army" )


def game_features( game: dict ) -> dict[str, float]:
    counts = {f"dev_{kind}": 0.0 for kind in KINDS}
    counts.update( {"target_pos1": 0.0, "target_pos2": 0.0, "target_pos3+": 0.0, "army_below": 0.0, "hire_none": 0.0} )
    for answer in game["answers"]:
        kind, event = answer["kind"], answer["event"]
        options = query_options( kind, event )
        builtin = builtin_index( kind, options, event )
        if builtin is None or answer["answer"] == builtin:
            continue
        counts[f"dev_{kind}"] += 1
        if kind == "target":
            counts["target_pos1" if answer["answer"] == 1 else "target_pos2" if answer["answer"] == 2 else "target_pos3+"] += 1
        elif kind == "army" and options[answer["answer"]] < options[builtin]:
            counts["army_below"] += 1
        elif kind == "hire" and answer["answer"] == len( options ) - 1:
            counts["hire_none"] += 1
    return counts


def fit( rows: list[tuple[int, dict, float]], names: list[str] ) -> np.ndarray:
    x = np.array( [[features[n] for n in names] for _, features, _ in rows] )
    y = np.array( [reward for _, _, reward in rows] )
    return np.linalg.lstsq( x, y, rcond=None )[0]


def main() -> None:
    parser = argparse.ArgumentParser( description="Learnable signal in whole-game GRPO groups" )
    parser.add_argument( "--data", required=True, help="glob of collect files" )
    parser.add_argument( "--boot", type=int, default=500 )
    args = parser.parse_args()

    groups = []
    for path in sorted( glob.glob( args.data ) ):
        with open( path ) as f:
            groups += [json.loads( line ) for line in f if line.strip()]
    rows = []  # (group id, centred features, centred reward)
    flat = 0
    for gid, group in enumerate( groups ):
        games = group["games"]
        rewards = np.array( [g["reward"] for g in games] )
        if len( games ) < 2 or rewards.std() < 1e-9:
            flat += 1
            continue
        features = [game_features( g ) for g in games]
        means = {k: np.mean( [f[k] for f in features] ) for k in features[0]}
        for f, r in zip( features, rewards ):
            rows.append( ( gid, {k: f[k] - means[k] for k in f}, float( r - rewards.mean() ) ) )
    print( f"{len( groups )} groups ({flat} without any reward spread), {len( rows )} games; reward sd within groups "
           f"{np.std( [r for _, _, r in rows] ):.3f}" )
    totals = {k: np.mean( [abs( f[k] ) for _, f, _ in rows] ) for k in rows[0][1]}

    for title, names in ( ( "per kind", [f"dev_{k}" for k in KINDS] ),
                          ( "details", ["target_pos1", "target_pos2", "target_pos3+", "dev_build", "hire_none", "army_below"] ) ):
        point = fit( rows, names )
        ids = sorted( {g for g, _, _ in rows} )
        by_group: dict[int, list] = {}
        for row in rows:
            by_group.setdefault( row[0], [] ).append( row )
        rng = random.Random( 0 )
        samples = np.sort( np.array( [fit( [r for g in rng.choices( ids, k=len( ids ) ) for r in by_group[g]], names )
                                      for _ in range( args.boot )] ), axis=0 )
        half = len( ids ) // 2
        shuffled = ids[:]
        random.Random( 1 ).shuffle( shuffled )
        first = fit( [r for g in shuffled[:half] for r in by_group[g]], names )
        second = fit( [r for g in shuffled[half:] for r in by_group[g]], names )
        print( f"== {title}: effect of ONE more deviation on the reward (hero equivalent, log2), 95% CI, halves" )
        for i, name in enumerate( names ):
            low, high = samples[int( 0.025 * args.boot ), i], samples[int( 0.975 * args.boot ) - 1, i]
            flag = " *" if low > 0 or high < 0 else ""
            print( f"   {name:14s} {point[i]:+.4f} [{low:+.4f}, {high:+.4f}]{flag}   halves {first[i]:+.4f} / {second[i]:+.4f}"
                   f"   (mean |centred count| {totals.get( name, 0 ):.2f})" )


if __name__ == "__main__":
    main()
