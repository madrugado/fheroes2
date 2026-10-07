"""How often does a strategic net leave the built-in answer, and how sure is it? (2026-10-07)

The oracle test (oracle_headroom.py) found real gains in only ~8% of the queries; a DPO model that
deviates far more often must be wrong in most of its deviations. This plays seeded games where one
color is answered by the net (the rest built-in, as play_vs_builtin does) and reports per query kind
how many answers differ from the built-in answer, with the net's probability of its pick and of the
built-in answer — the input for a confidence gate (strategy_net.NetStrategyPolicy min_margin).

    rl/.venv/bin/python rl/deviation_stats.py --model rl/data/strategy_loop_hero/model_r4.pt --seeds 101-110 --out rl/data/dev_r4.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import types
from collections import defaultdict

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

import strategy_games  # noqa: E402
from harvest_battles import parse_seeds  # noqa: E402
from strategy_net import NetStrategyPolicy, query_options  # noqa: E402


class RecordingPolicy( NetStrategyPolicy ):
    """NetStrategyPolicy that keeps the probabilities of its last decision."""

    def decide( self, kind: str, event: dict ):
        options = query_options( kind, event )
        if len( options ) < 2:
            self.last = None
            return None
        probs = self.probabilities( kind, event )
        self.last = probs
        return max( range( len( options ) ), key=lambda i: probs[i] )


def play_game( args, policy: RecordingPolicy, seed: int, color: str, out ) -> int:
    rows = []
    original = policy.decide

    def decide( kind, event ):
        index = original( kind, event )
        if index is not None and policy.last is not None:
            options = query_options( kind, event )
            builtin = strategy_games.builtin_index( kind, options, event )
            rows.append( {"seed": seed, "color": color, "kind": kind, "t": event.get( "t" ), "pick": index, "builtin": builtin,
                          "p_pick": policy.last[index], "p_builtin": None if builtin is None else policy.last[builtin],
                          "options": len( options )} )
        return index

    policy.decide = decide
    try:
        game_args = types.SimpleNamespace( binary=args.binary, map=args.map, days=args.days, color=color )
        strategy_games.play( game_args, seed, args.days, strategy_games.PolicyBranch( policy, color ) )
    finally:
        policy.decide = original
    for row in rows:
        out.write( json.dumps( row, separators=( ",", ":" ) ) + "\n" )
    out.flush()
    return len( rows )


def summarize( rows: list[dict] ) -> dict:
    games = {( r["seed"], r["color"] ) for r in rows}
    by_kind: dict[str, dict] = defaultdict( lambda: {"queries": 0, "deviations": 0} )
    deviating, agreeing = [], []
    for row in rows:
        if row["builtin"] is None:
            continue
        stats = by_kind[row["kind"]]
        stats["queries"] += 1
        if row["pick"] != row["builtin"]:
            stats["deviations"] += 1
            deviating.append( row["p_pick"] - row["p_builtin"] )
        else:
            agreeing.append( row["p_pick"] )

    def quantiles( values ):
        if not values:
            return None
        values = sorted( values )
        return [round( values[int( q * ( len( values ) - 1 ) )], 3 ) for q in ( 0.1, 0.25, 0.5, 0.75, 0.9 )]

    total = sum( s["deviations"] for s in by_kind.values() )
    return {"games": len( games ), "deviations_per_game": round( total / max( 1, len( games ) ), 2 ),
            "by_kind": dict( by_kind ), "deviation_margin_quantiles": quantiles( deviating ),
            "agreeing_p_quantiles": quantiles( agreeing ),
            "deviations_with_margin_above": {str( m ): sum( 1 for d in deviating if d > m ) for m in ( 0.1, 0.3, 0.5, 0.8, 0.95 )},
            "mean_margin": round( statistics.fmean( deviating ), 3 ) if deviating else None}


def main() -> None:
    parser = argparse.ArgumentParser( description="Deviations of a strategic net from the built-in answers" )
    parser.add_argument( "--model", required=True )
    parser.add_argument( "--binary", default="./fheroes2" )
    parser.add_argument( "--map", default="2kings.mp2" )
    parser.add_argument( "--days", type=int, default=30 )
    parser.add_argument( "--seeds", default="101-110" )
    parser.add_argument( "--colors", default="Blue,Red" )
    parser.add_argument( "--out", required=True )
    parser.add_argument( "--summary-only", action="store_true" )
    args = parser.parse_args()
    sys.stdout.reconfigure( line_buffering=True )

    if not args.summary_only:
        policy = RecordingPolicy( args.model )
        with open( args.out, "a" ) as out:
            for seed in parse_seeds( args.seeds ):
                for color in args.colors.split( "," ):
                    print( f"seed {seed} {color}: {play_game( args, policy, seed, color, out )} queries" )
    with open( args.out ) as f:
        rows = [json.loads( line ) for line in f if line.strip()]
    print( json.dumps( summarize( rows ) ) )


if __name__ == "__main__":
    main()
