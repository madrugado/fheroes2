"""How much of the strategic reward's noise is the final duel's own randomness (idea 2 of 2026-10-28).

The whole-game GRPO / JOLT reward of a game is -equivalent_of(...): the duel equivalent of our strongest hero
against one reference (strategy_games.hero_equivalent, DUEL_SEEDS battle seeds per side; `--label final`:
final_label instead). Its spread within a group of games was 0.77 in the jolt2 data. Part of that spread is the
DUEL's own randomness: the same game end state, fought on other battle seeds, gets another label. This measures
it: built-in games are played once (--plan as in the collectors), then every game's label is recomputed on
--sets different duel-seed sets — the GAME is not replayed (duels cost ~0 next to a game).

  per game  v_n = variance of the label over the sets  = the duel noise of one measurement
  duel share = mean_n(v_n) / --group-spread^2          — the duel's share of the within-group variance
  the rest of the spread is real differences between the games (signal + game luck)

Won / lost games (or a game without heroes / without a rival hero) have a FIXED label — no duel, no duel noise;
their share is reported and they are left out of the estimate. The bisection's own resolution is negligible
(12 / 2^HERO_STEPS ~ 0.09 in log2). If the duel share is large, more duel seeds remove it almost for free:
measure again with --duel-seeds K and compare.

Only the built-in AI plays (a game takes seconds). Strictly one engine at a time (plus the duel server):
    rl/.venv/bin/python rl/duel_noise.py --seeds 1-24 --sets 8 --out rl/data/duel_noise.jsonl
    rl/.venv/bin/python rl/duel_noise.py --summary-only --out rl/data/duel_noise.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import types
from collections import defaultdict

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

import strategy_games  # noqa: E402
from engine_bridge import BattleEnv  # noqa: E402
from harvest_battles import parse_seeds  # noqa: E402
from label_noise import BuiltinBranchPolicy  # noqa: E402


def fixed_label( args, results: dict, state: str, reference: dict | None ) -> float | None:
    """The label of a game that never fights the duel (won / lost / no hero of ours / for `hero` also no rival
    hero to fight): the same values equivalent_of / final_label return, or None when the duel is fought."""
    top = ( results.get( args.color ) or {} ).get( "top" )
    if args.label == "final":
        if state == "0":
            return 1.0
        if state == "1" or top is None:
            return -1.0
        return 1.0 if reference is None else None
    if state == "0":
        return -strategy_games.HERO_LOG2_RANGE
    if state == "1" or top is None or reference is None:
        return strategy_games.HERO_LOG2_RANGE
    return None


def measure_game( args, seed: int, duel_env: BattleEnv, out ) -> int:
    """Plays one seeded built-in game and writes one row per duel-seed set: the game's label on that set."""
    policy = BuiltinBranchPolicy()
    base = strategy_games.PolicyBranch( policy, args.color )
    _, results, _, _ = strategy_games.play( args, seed, args.days, base )
    state = str( ( results.get( args.color ) or {} ).get( "s", "" ) )
    reference = strategy_games.strongest_rival_hero( results, args.color )
    fixed = fixed_label( args, results, state, reference )
    rows = 0
    for m in range( args.sets ):
        duel_seed = 700001 + 1000 * m + seed  # outside the game seeds: every set its own battle seeds
        if fixed is not None:
            value = fixed
        elif args.label == "final":
            value = strategy_games.final_label( duel_env, results, args.color, duel_seed, args.duel_seeds )
        else:
            top = results[args.color]["top"]
            value = -strategy_games.hero_equivalent( duel_env, top, reference, duel_seed, args.duel_seeds )
        out.write( json.dumps( {"seed": seed, "m": m, "reward": round( value, 4 ), "fixed": fixed is not None,
                                "state": state, "label": args.label, "duel_seeds": args.duel_seeds},
                               separators=( ",", ":" ) ) + "\n" )
        rows += 1
    out.flush()
    return rows


def summarize( rows: list[dict], spread: float ) -> dict:
    """The duel-noise decomposition (see the module doc): the mean per-game variance of the label over the duel-seed
    sets against the reference group spread, the share of fixed (never duelled) games, and the variance across the
    games themselves (over seeds here: context, not a collector group's within-seed spread)."""
    games: dict[int, dict[int, float]] = defaultdict( dict )
    fixed: dict[int, bool] = {}
    for row in rows:
        games[row["seed"]][row["m"]] = row["reward"]
        fixed[row["seed"]] = row["fixed"]
    duelled = {seed: list( values.values() ) for seed, values in games.items() if not fixed[seed]}
    variances = [statistics.pvariance( values ) for values in duelled.values() if len( values ) >= 2]
    duel_var = statistics.fmean( variances ) if variances else 0.0
    means = {seed: statistics.fmean( values.values() ) for seed, values in games.items()}
    between = statistics.pvariance( list( means.values() ) ) if len( means ) >= 2 else 0.0
    out = {
        "games": len( games ),
        "duelled_games": len( duelled ),
        "fixed_games": sum( fixed.values() ),
        "duel_sd_of_one_measurement": round( math.sqrt( duel_var ), 4 ),
        "duel_variance": round( duel_var, 4 ),
        "reference_group_spread": spread,
        "duel_share_of_group_variance": round( duel_var / ( spread * spread ), 3 ) if spread > 0 else None,
        "between_game_sd_across_seeds": round( math.sqrt( between ), 4 ),
        "duel_seeds": sorted( {row.get( "duel_seeds", 0 ) for row in rows} ),
        "label": sorted( {row.get( "label", "" ) for row in rows} ),
    }
    if variances:
        out["per_game_duel_sd_median"] = round( statistics.median( math.sqrt( v ) for v in variances ), 4 )
    if any( row.get( "label" ) == "hero" for row in rows ):
        out["hero_resolution"] = round( 2 * strategy_games.HERO_LOG2_RANGE / 2 ** strategy_games.HERO_STEPS, 4 )
    return out


def main() -> None:
    parser = argparse.ArgumentParser( description="Duel noise vs game differences in the strategic reward" )
    parser.add_argument( "--binary", default="./fheroes2" )
    parser.add_argument( "--map", default="2kings.mp2" )
    parser.add_argument( "--days", type=int, default=30 )
    parser.add_argument( "--color", default="Blue" )
    parser.add_argument( "--seeds", default="1-24" )
    parser.add_argument( "--plan", default="champion=1,split_singles=1,max_heroes=3",
                         help="engine plan of the games ('' = none; the collectors run this one)" )
    parser.add_argument( "--no-rng-streams", dest="rng_streams", action="store_false" )
    parser.add_argument( "--label", choices=["hero", "final"], default="hero",
                         help="hero: the GRPO/JOLT reward (-equivalent_of); final: the value data's final_label" )
    parser.add_argument( "--sets", type=int, default=8, help="duel-seed sets per game" )
    parser.add_argument( "--duel-seeds", type=int, default=strategy_games.DUEL_SEEDS,
                         help="battle seeds per side within one measurement" )
    parser.add_argument( "--group-spread", type=float, default=0.77, help="the within-group reward spread to compare "
                         "against (0.77 in the jolt2 data)" )
    parser.add_argument( "--out", required=True )
    parser.add_argument( "--summary-only", action="store_true", help="summarize an existing --out file" )
    args = parser.parse_args()
    sys.stdout.reconfigure( line_buffering=True )

    if not args.summary_only:
        game_args = types.SimpleNamespace( binary=args.binary, map=args.map, days=args.days, color=args.color,
                                           plan=args.plan or None, rng_streams=args.rng_streams or None, label=args.label,
                                           sets=args.sets, duel_seeds=args.duel_seeds )
        duel_env = BattleEnv( binary=args.binary, map_name=args.map )
        started = time.time()
        try:
            with open( args.out, "a" ) as out:
                for seed in parse_seeds( args.seeds ):
                    rows = measure_game( game_args, seed, duel_env, out )
                    print( f"seed {seed}: {rows} measurements, {time.time() - started:.0f}s" )
        finally:
            duel_env.close()

    with open( args.out ) as f:
        rows = [json.loads( line ) for line in f if line.strip()]
    summary = summarize( rows, args.group_spread )
    print( json.dumps( summary ) )
    with open( os.path.splitext( args.out )[0] + "_summary.json", "w" ) as f:
        json.dump( summary, f, indent=1 )


if __name__ == "__main__":
    main()
