"""Time profile of one strategy_loop collection game (strategy_games.label_game, the loop's settings):
where the wall time goes — the strategic net, the game engine, the final-label duels.

    python rl/profile_loop.py --model rl/models/unified_200m_best.pt --seed 1000 --per-game 1 --salts 2
"""

import argparse
import collections
import cProfile
import hashlib
import io
import json
import os
import pstats
import random
import sys
import time
import types

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

import strategy_net  # noqa: E402  (first: the other rl modules put rl/ at the front of sys.path)
import engine_bridge  # noqa: E402
import strategy_env  # noqa: E402
import strategy_games  # noqa: E402

TOTAL = collections.defaultdict( float )
CALLS = collections.Counter()
_depth = collections.Counter()


def timed( owner, name: str, label: str ) -> None:
    """Wraps owner.name: wall time and calls under `label` (outermost call only for recursion)."""
    original = getattr( owner, name )

    def wrapper( *args, **kwargs ):
        _depth[label] += 1
        start = time.perf_counter()
        try:
            return original( *args, **kwargs )
        finally:
            _depth[label] -= 1
            if _depth[label] == 0:
                TOTAL[label] += time.perf_counter() - start
            CALLS[label] += 1

    setattr( owner, name, wrapper )


def main() -> None:
    parser = argparse.ArgumentParser( description=__doc__.splitlines()[0] )
    parser.add_argument( "--model", required=True )
    parser.add_argument( "--seed", type=int, default=1000 )
    parser.add_argument( "--per-game", type=int, default=1 )
    parser.add_argument( "--salts", type=int, default=2 )
    parser.add_argument( "--days", type=int, default=45 )
    parser.add_argument( "--no-cache", action="store_true", help="the strategic net without its inference caches" )
    parser.add_argument( "--cprofile", default=None, help="write the cProfile stats here" )
    args = parser.parse_args()

    timed( strategy_games, "play", "game (engine + net)" )
    timed( strategy_net.NetStrategyPolicy, "probabilities", "net: forward" )
    timed( strategy_net.NetStrategyPolicy, "observe_turn", "net: observe_turn" )
    timed( strategy_games, "final_label", "final label (duels)" )
    timed( strategy_games, "duel_wins", "duel_wins (10 battles)" )
    timed( engine_bridge.BattleEnv, "new_battle", "duel: new_battle" )
    timed( engine_bridge.BattleEnv, "snapshot_restore", "duel: rollout" )
    timed( strategy_env.StrategyEnv, "__init__", "engine start" )

    start = time.perf_counter()
    policy = strategy_net.NetStrategyPolicy( args.model, cache=not args.no_cache )
    load = time.perf_counter() - start
    duel_env = engine_bridge.BattleEnv( binary="./fheroes2", map_name="2kings.mp2" )
    game_args = types.SimpleNamespace( binary="./fheroes2", map="2kings.mp2", days=args.days, horizons="21", color="Blue",
                                       per_game=args.per_game, random=1, margin=0.1, label="final", salts=args.salts )
    trajectories: list[dict] = []
    profiler = cProfile.Profile()
    start = time.perf_counter()
    profiler.enable()
    pairs = strategy_games.label_game( game_args, policy, args.seed, random.Random( args.seed ), duel_env, trajectories )
    profiler.disable()
    wall = time.perf_counter() - start
    duel_env.close()

    games = CALLS["game (engine + net)"]
    net = TOTAL["net: forward"] + TOTAL["net: observe_turn"]
    game = TOTAL["game (engine + net)"]
    # final_label runs outside play() (after it), so the game time holds the engine + the net only.
    print( f"model load {load:.1f}s; label_game {wall:.1f}s: {games} games played, {len( pairs )} pairs, {len( trajectories )} value records" )
    rows = [
        ( "net (forward + observe_turn)", net ),
        ( "engine (game - net)", game - net ),
        ( "final label duels", TOTAL["final label (duels)"] ),
        ( "other (python glue)", wall - game - TOTAL["final label (duels)"] ),
    ]
    for name, seconds in rows:
        print( f"  {name:32s} {seconds:8.1f}s {100 * seconds / wall:5.1f}%" )
    print( "details:" )
    for label in sorted( TOTAL ):
        calls = CALLS[label]
        print( f"  {label:32s} {TOTAL[label]:8.1f}s {calls:7d} calls {1000 * TOTAL[label] / max( 1, calls ):9.1f} ms/call" )
    prefix_cache = getattr( policy, "prefix_cache", None )
    if prefix_cache is not None:
        print( f"caches: {policy.answer_hits} answers reused, prefix chunks {prefix_cache.hits} hits / {prefix_cache.misses} computed" )
    digest = hashlib.sha256( json.dumps( [pairs, trajectories], sort_keys=True ).encode() ).hexdigest()[:16]
    print( f"output digest {digest} (pairs + value records: equal digests = the same games and labels)" )
    if games:
        print( f"per game: {game / games:.1f}s (net {net / games:.1f}s, {CALLS['net: forward'] / games:.0f} net queries)" )

    stream = io.StringIO()
    stats = pstats.Stats( profiler, stream=stream ).sort_stats( "tottime" )
    stats.print_stats( 25 )
    print( stream.getvalue() )
    if args.cprofile:
        stats.dump_stats( args.cprofile )


if __name__ == "__main__":
    main()
