"""Harvests real battle setups from seeded autonomous playtests (built-in AI everywhere).

Every battle of the game is reported by the engine's battle-agent channel as a "battle_start"
event (see az/README.md, "Real-battle integration"): stacks, tile, world seed, formations,
colors, the commander heroes and the castle as save-game serializations. The runner answers
every decision with "planner", so the game is exactly the built-in AI's game. The setups rebuild
the battles in the headless battle server (engine_bridge.new_battle_from_setup) — for expert
data (gen_expert.py --setups) and the gate (gate.py --setups) on the battles that real games
produce: hero skills, artifacts, spells, sieges, town garrisons.

Usage (at most 2 engines in parallel, see AGENTS.md "Machine load rule"):
    az/.venv/bin/python az/harvest_battles.py --map Battlefi.mp2 --days 30 --seeds 1-6 \\
        --out az/data/battles_Battlefi.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

from battle_agent import BattleAgentRunner  # noqa: E402


def harvest( binary: str, map_name: str, days: int, seed: int ) -> list[dict]:
    """Plays one seeded game with the built-in AI; returns its battle_start events (each with the
    map name and the game seed added)."""
    runner = BattleAgentRunner( binary, map_name, days, 1, "planner",
                                extra_env={"FHEROES2_AUTO_PLAYTEST_SEED": str( seed )} )
    setups: list[dict] = []

    def handle( ev: dict ) -> bool:
        if ev.get( "ev" ) == "battle_start":
            setups.append( dict( ev, map=map_name, game_seed=seed ) )
        return False

    runner._handle_event = handle
    try:
        runner.run()
    finally:
        runner.close()
    return setups


def parse_seeds( text: str ) -> list[int]:
    """'1-6,9' -> [1, 2, 3, 4, 5, 6, 9]."""
    seeds: list[int] = []
    for part in text.split( "," ):
        if "-" in part:
            first, last = part.split( "-" )
            seeds.extend( range( int( first ), int( last ) + 1 ) )
        elif part:
            seeds.append( int( part ) )
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser( description="Harvest real battle setups from seeded playtests" )
    parser.add_argument( "--binary", type=str, default="./fheroes2" )
    parser.add_argument( "--map", type=str, default="Battlefi.mp2" )
    parser.add_argument( "--days", type=int, default=30 )
    parser.add_argument( "--seeds", type=str, default="1-4" )
    parser.add_argument( "--jobs", type=int, default=2, help="parallel engines (keep <= 2)" )
    parser.add_argument( "--out", type=str, default=None )
    args = parser.parse_args()

    out_path = args.out or f"az/data/battles_{os.path.splitext( args.map )[0]}.jsonl"
    os.makedirs( os.path.dirname( out_path ), exist_ok=True )

    seeds = parse_seeds( args.seeds )
    t0 = time.time()
    total = 0
    with open( out_path, "w" ) as out, ThreadPoolExecutor( max_workers=args.jobs ) as pool:
        for seed, setups in zip( seeds, pool.map( lambda s: harvest( args.binary, args.map, args.days, s ), seeds ) ):
            for setup in setups:
                out.write( json.dumps( setup, separators=( ",", ":" ) ) + "\n" )
            total += len( setups )
            heroes = sum( 1 for s in setups if "hid" in s["att"] or "hid" in s["def"] )
            castles = sum( 1 for s in setups if "castle" in s )
            print( f"seed {seed}: {len( setups )} battles ({heroes} with heroes, {castles} on castle/town tiles)", flush=True )

    print( f"done in {time.time() - t0:.0f}s: {total} battles -> {out_path}" )


if __name__ == "__main__":
    main()
