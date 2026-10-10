"""Single-monster stacks in battle (plan key split_singles), measured directly in the battle server.

Every harvested real battle (harvest_battles.py) is fought once per variant of the side under test
(a side with a commander hero): 0 = as it is, 2 = its weakest stack split into the free slots,
1 = split + the champion's targeting (single stacks attack where the retaliation is still to come).
The built-in AI plays both sides to the end, several battle seeds per setup. Score of the side under
test as in battle_prefs.move_score: outcome + own strength left (share) + enemy strength killed
(share) - 1; also the own strength lost. Differences are paired by (setup, side, seed).

  rl/.venv/bin/python rl/split_bench.py --setups rl/data/battles_gate_Battlefi.jsonl --map Battlefi.mp2
"""

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert( 0, str( Path( __file__ ).resolve().parent ) )

from battle_prefs import move_score, side_strength  # noqa: E402
from engine_bridge import BattleEnv, new_battle_from_setup  # noqa: E402
from gen_expert import load_setups  # noqa: E402

VARIANTS = ( 0, 2, 1, 4, 3 )  # none, weakest split, + targeting, fastest-weak split, + targeting
NAMES = { 0: "none", 2: "split", 1: "split+target", 4: "fast", 3: "fast+target" }


def play( env: BattleEnv, setup: dict, seed: int, side: str, split: int ) -> dict | None:
    root = new_battle_from_setup( env, setup, seed=seed, att_split=split if side == "att" else 0, def_split=split if side == "def" else 0 )
    if root is None or root.get( "ev" ) != "state":
        return None
    if root.get( "result" ):
        final = root
    else:
        env.snapshot_save( 1 )
        final = env.snapshot_restore( 1, rollout=True )
    if final is None or final.get( "ev" ) != "state":
        return None
    units = sum( 1 for u in root["units"] if u["side"] == side )
    enemy = "def" if side == "att" else "att"
    return {
        "units": units,
        "score": move_score( root, final, side ),
        "win": final.get( "result" ) == side,
        "lost": side_strength( root, side ) - side_strength( final, side ),
        "own0": side_strength( root, side ),
        "enemy_hero": "hero" in setup[enemy],
    }


def bootstrap( values: list[float], n: int = 4000 ) -> tuple[float, float]:
    rng = random.Random( 0 )
    means = sorted( statistics.fmean( rng.choices( values, k=len( values ) ) ) for _ in range( n ) )
    return means[int( 0.025 * n )], means[int( 0.975 * n )]


def summarize( rows: list[dict] ) -> None:
    for subset, keep in ( ( "all", lambda r: True ), ( "vs hero", lambda r: r["enemy_hero"] ), ( "vs neutral", lambda r: not r["enemy_hero"] ) ):
        chosen = [r for r in rows if keep( r )]
        if not chosen:
            continue
        print( f"-- {subset}: {len( chosen )} battles (split applied)" )
        for variant in ( 2, 1, 4, 3 ):
            for metric in ( "score", "win", "lost_share" ):
                diff = [float( r[variant][metric] ) - float( r[0][metric] ) for r in chosen]
                low, high = bootstrap( diff )
                base = statistics.fmean( float( r[0][metric] ) for r in chosen )
                print( f"   {NAMES[variant]:13s} {metric:10s} base {base:+.3f}  diff {statistics.fmean( diff ):+.4f} [{low:+.4f}, {high:+.4f}]" )


def main() -> None:
    parser = argparse.ArgumentParser( description="Single-monster stacks in battle: paired battle-server benchmark" )
    parser.add_argument( "--setups", nargs="+", required=True )
    parser.add_argument( "--map", required=True )
    parser.add_argument( "--seeds", type=int, default=5, help="battle seeds per setup" )
    parser.add_argument( "--limit", type=int, default=0, help="at most this many setups (0 = all)" )
    parser.add_argument( "--out", default="" )
    args = parser.parse_args()

    setups = load_setups( args.setups )
    if args.limit:
        setups = setups[: args.limit]
    env = BattleEnv( map_name=args.map )
    rows = []
    out = open( args.out, "w" ) if args.out else None
    try:
        for index, setup in enumerate( setups ):
            for side in ( "att", "def" ):
                if "hero" not in setup[side]:
                    continue
                for k in range( args.seeds ):
                    seed = setup["seed"] + 7919 * k
                    results = { variant: play( env, setup, seed, side, variant ) for variant in VARIANTS }
                    if any( r is None for r in results.values() ):
                        continue
                    if results[2]["units"] == results[0]["units"] and results[4]["units"] == results[0]["units"]:
                        break  # no free slot / nothing to split: every seed is the same
                    for r in results.values():
                        r["lost_share"] = r["lost"] / r["own0"] if r["own0"] else 0.0
                    row = { **results, "enemy_hero": results[0]["enemy_hero"], "setup": index, "side": side, "seed": seed }
                    rows.append( row )
                    if out:
                        out.write( json.dumps( row, separators=( ",", ":" ) ) + "\n" )
            if ( index + 1 ) % 50 == 0:
                print( f"{index + 1}/{len( setups )} setups, {len( rows )} battles", flush=True )
    finally:
        env.close()
        if out:
            out.close()
    summarize( rows )


if __name__ == "__main__":
    main()
