"""Paired head-to-head benchmark of a strategic policy against the built-in AI.

For every seed the engine plays one CONTROL game (every player on the built-in strategic AI)
and, for every player color, one TREATMENT game where only that color picks hero targets with
the tested policy. FHEROES2_AUTO_PLAYTEST_SEED makes the games reproducible, so a treatment game
diverges from its control only through the policy's choices — the comparison is paired per
(seed, color), which removes most of the map/luck variance.

Metrics per player come from the "game_end" event: the outcome state, castles (k), heroes (h),
total army strength (str) and gold (g). A (seed, color) pair is "better" / "worse" / "equal" by
the lexicographic key (outcome, castles, army strength). The summary adds an exact two-sided sign test over the
better/worse verdicts and a 95% bootstrap interval of the mean army-strength difference.

Usage:
    az/.venv/bin/python az/strategy_bench.py --policy tempo --map 2kings.mp2 --days 14 --seeds 8 --jobs 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

from strategy_env import StrategyEnv  # noqa: E402
from strategy_policies import STRATEGY_POLICIES, ForColor, builtin_policy, make_strategy_policy  # noqa: E402

# AutoPlaytest::PlayerState (game_auto_playtest.h) as sent in "game_end": WINNER, LOSER,
# TIME_LIMIT, INTERRUPTED -> outcome score.
OUTCOME_SCORE = {"0": 1, "1": -1, "2": 0, "3": 0}


def player_stats( game_end: dict ) -> dict[str, dict]:
    """color -> {"outcome", "k", "h", "str", "g"} from a game_end event."""
    stats = {}
    for result in game_end.get( "results" ) or []:
        stats[result["c"]] = {
            "outcome": OUTCOME_SCORE.get( str( result.get( "s" ) ), 0 ),
            "k": result.get( "k", 0 ),
            "h": result.get( "h", 0 ),
            "str": result.get( "str", 0 ),
            "g": result.get( "g", 0 ),
        }
    return stats


def rank_key( stats: dict ) -> tuple:
    return ( stats["outcome"], stats["k"], stats["str"] )


def compare( control: dict, treatment: dict ) -> dict:
    """Paired comparison of one color: treatment minus control."""
    diff = {key: treatment[key] - control[key] for key in ( "outcome", "k", "h", "str", "g" )}
    key_c, key_t = rank_key( control ), rank_key( treatment )
    diff["verdict"] = "better" if key_t > key_c else ( "worse" if key_t < key_c else "equal" )
    return diff


def sign_test_p( better: int, worse: int ) -> float:
    """Two-sided exact sign test (ties dropped): the probability of a split at least this
    lopsided if the policy made no difference."""
    n = better + worse
    if n == 0:
        return 1.0
    k = min( better, worse )
    tail = sum( math.comb( n, i ) for i in range( k + 1 ) ) / 2**n
    return min( 1.0, 2 * tail )


def bootstrap_ci( values: list[float], iterations: int = 2000, seed: int = 0 ) -> tuple[float, float]:
    """95% percentile bootstrap interval of the mean (deterministic)."""
    if not values:
        return ( 0.0, 0.0 )
    rng = random.Random( seed )
    means = sorted( sum( rng.choices( values, k=len( values ) ) ) / len( values ) for _ in range( iterations ) )
    return ( round( means[int( 0.025 * iterations )], 3 ), round( means[int( 0.975 * iterations ) - 1], 3 ) )


def summarize( pairs: list[dict] ) -> dict:
    """Aggregates paired comparisons: verdict counts and mean differences."""
    summary: dict = {"pairs": len( pairs )}
    for verdict in ( "better", "equal", "worse" ):
        summary[verdict] = sum( 1 for p in pairs if p["verdict"] == verdict )
    for key in ( "outcome", "k", "h", "str", "g" ):
        summary[f"mean_d_{key}"] = round( sum( p[key] for p in pairs ) / len( pairs ), 3 ) if pairs else 0.0
    # Share of pairs where the policy actually changed a decision (0 = identical to builtin).
    summary["changed_games"] = sum( 1 for p in pairs if p.get( "changed" ) )
    summary["sign_test_p"] = round( sign_test_p( summary["better"], summary["worse"] ), 4 )
    summary["ci95_d_str"] = bootstrap_ci( [p["str"] for p in pairs] )
    return summary


def play( binary: str, map_name: str, days: int, seed: int, policy ) -> tuple[dict, int]:
    """One seeded game; returns (game_end event, number of decisions the policy overrode)."""
    env = StrategyEnv( binary=binary, map_name=map_name, days=days, playthroughs=1, seed=seed )
    overrides = 0

    def counting( decision: dict ):
        nonlocal overrides
        chosen = policy( decision )
        cands = decision.get( "cands" ) or []
        # The built-in choice is the top-value candidate (the engine sorts them by value).
        if chosen is not None and cands and chosen["i"] != cands[0]["i"]:
            overrides += 1
        return chosen

    if hasattr( policy, "observe_turn" ):
        counting.observe_turn = policy.observe_turn  # type: ignore[attr-defined]

    try:
        summaries = env.run( counting )
    finally:
        env.close()

    if not summaries:
        raise RuntimeError( f"seed {seed}: the game did not report game_end" )
    return summaries[-1], overrides


def main() -> None:
    parser = argparse.ArgumentParser( description="Paired benchmark of a strategic policy vs the built-in AI" )
    parser.add_argument( "--policy", choices=[p for p in STRATEGY_POLICIES if p != "builtin"], default="tempo" )
    parser.add_argument( "--binary", type=str, default="./fheroes2" )
    parser.add_argument( "--map", type=str, default="2kings.mp2" )
    parser.add_argument( "--days", type=int, default=14 )
    parser.add_argument( "--seeds", type=int, default=8 )
    parser.add_argument( "--first-seed", type=int, default=1 )
    parser.add_argument( "--jobs", type=int, default=4, help="engine processes in parallel" )
    parser.add_argument( "--out", type=str, default="az/data" )
    args = parser.parse_args()

    seeds = list( range( args.first_seed, args.first_seed + args.seeds ) )
    t0 = time.time()

    with ThreadPoolExecutor( max_workers=args.jobs ) as pool:
        controls = dict( zip( seeds, pool.map( lambda s: play( args.binary, args.map, args.days, s, builtin_policy )[0], seeds ) ) )

        colors = sorted( player_stats( next( iter( controls.values() ) ) ) )
        jobs = [( s, c ) for s in seeds for c in colors]

        def treatment( job ):
            seed, color = job
            policy = ForColor( make_strategy_policy( args.policy, random.Random( seed ) ), color )
            return play( args.binary, args.map, args.days, seed, policy )

        treatments = dict( zip( jobs, pool.map( treatment, jobs ) ) )

    pairs = []
    for ( seed, color ), ( game_end, overrides ) in treatments.items():
        pair = compare( player_stats( controls[seed] )[color], player_stats( game_end )[color] )
        pair.update( seed=seed, color=color, overrides=overrides, changed=overrides > 0 )
        pairs.append( pair )

    report = {
        "policy": args.policy,
        "map": args.map,
        "days": args.days,
        "seeds": seeds,
        "summary": summarize( pairs ),
        "pairs": pairs,
        "seconds": round( time.time() - t0, 1 ),
    }

    os.makedirs( args.out, exist_ok=True )
    out_path = os.path.join( args.out, f"bench_{args.policy}_{os.path.splitext( args.map )[0]}_{args.days}d.json" )
    with open( out_path, "w" ) as out:
        json.dump( report, out, indent=1 )

    print( json.dumps( report["summary"] ) )
    print( f"{len( pairs )} pairs in {report['seconds']}s -> {out_path}" )


if __name__ == "__main__":
    main()
