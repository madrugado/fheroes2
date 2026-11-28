"""Analysis of a play_vs_builtin --strategy-trace file: WHAT the student picks instead of the built-in AI's
answer (the raw option descriptors of the trace entries: build {"b": id}, NOTHING, BUILTIN), per query kind,
and how the deviations map to the paired-game deltas (castles k, army str, outcome) of the eval JSON.

    python rl/analyze_build_trace.py TRACE.jsonl PAIRS_JSON
"""

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict

# BuildingType bits (src/fheroes2/castle/castle.h)
BUILDING_NAMES = {
    0x1: "thievesguild", 0x2: "tavern", 0x4: "shipyard", 0x8: "well", 0x10: "statue", 0x20: "leftturret",
    0x40: "rightturret", 0x80: "marketplace", 0x100: "wel2", 0x200: "moat", 0x400: "special",
    0x800: "castle", 0x1000: "captain", 0x2000: "shrine",
    0x4000: "mageguild1", 0x8000: "mageguild2", 0x10000: "mageguild3", 0x20000: "mageguild4", 0x40000: "mageguild5",
    0x80000: "tent", 0x100000: "dwelling1", 0x200000: "dwelling2", 0x400000: "dwelling3", 0x800000: "dwelling4",
    0x1000000: "dwelling5", 0x2000000: "dwelling6",
    0x4000000: "upgrade2", 0x8000000: "upgrade3", 0x10000000: "upgrade4", 0x20000000: "upgrade5",
    0x40000000: "upgrade6", 0x80000000: "upgrade7",
}


def option_name( option ) -> str:
    """A short name of a raw option descriptor: building bits by name, NOTHING, BUILTIN, hero dicts, budgets."""
    if isinstance( option, dict ):
        if "b" in option and isinstance( option["b"], int ):
            return BUILDING_NAMES.get( option["b"], "b" + str( option["b"] ) )
        if "hero" in option:
            return f"hero{option['hero']}(army{int( option.get( 'army', 0 ) )})"
        return json.dumps( option )
    return str( option )


def main() -> None:
    parser = argparse.ArgumentParser( description=__doc__ )
    parser.add_argument( "trace" )
    parser.add_argument( "pairs" )
    args = parser.parse_args()

    entries = [json.loads( line ) for line in open( args.trace ) if line.strip()]
    pairs = {( p["seed"], p["color"] ): p for p in json.load( open( args.pairs ) )["pairs"]}

    print( f"trace: {len( entries )} answered queries over {len( {e[ 'seed' ] for e in entries} )} seeds" )
    for kind in sorted( {e["kind"] for e in entries} ):
        rows = [e for e in entries if e["kind"] == kind]
        deviating = [e for e in rows if e["bi"] is not None and e["pick"] != e["bi"]]
        print( f"\n== {kind}: {len( rows )} queries, {len( deviating )} deviate from the built-in answer "
               f"({len( deviating ) / max( len( rows ), 1 ):.0%})" )
        if kind in ( "build", "hire" ) and deviating:
            transitions = Counter( ( option_name( e.get( "po" ) ), option_name( e.get( "bo" ) ) ) for e in deviating )
            print( f"  top picks instead of the built-in one (picked -> built-in):" )
            for ( po, bo ), count in transitions.most_common( 12 ):
                print( f"    {po:>14} -> {bo:<14} x{count}" )
        if deviating:
            confidence = [e["pp"] for e in deviating if e["pp"] is not None]
            if confidence:
                print( f"  confidence on deviations: mean p(pick) {statistics.fmean( confidence ):.3f}, "
                       f"p(builtin) {statistics.fmean( [e['pb'] for e in deviating if e['pb'] is not None] ):.3f}" )

    # the game level: does a higher build-deviation share come with fewer castles / less army?
    by_game: dict[tuple, list[dict]] = defaultdict( list )
    for e in entries:
        by_game[( e["seed"], e["p"] )].append( e )
    buckets: dict[str, list[tuple]] = defaultdict( list )
    for key, rows in by_game.items():
        pair = pairs.get( key )
        if pair is None:
            continue  # the control arm has no trace
        builds = [e for e in rows if e["kind"] == "build" and e["bi"] is not None]
        if not builds:
            continue
        share = sum( e["pick"] != e["bi"] for e in builds ) / len( builds )
        share = min( share, 0.99 )
        buckets[str( int( share * 4 ) )].append( ( pair["k"], pair["str"], pair["outcome"], pair["won"] - pair["control_won"] ) )
    print( "\n== games bucketed by the share of deviating build answers (quartile 0-3 of the share):" )
    print( "  bucket  games  mean d_k (castles)  mean d_str (army)  mean d_outcome  more wins than control" )
    for bucket in sorted( buckets ):
        rows = list( buckets[bucket] )
        ks = [r[0] for r in rows]
        strs = [r[1] for r in rows]
        outs = [r[2] for r in rows]
        wins = [r[3] for r in rows]
        print( f"  {bucket}/4+    {len( rows ):5d}  {statistics.fmean( ks ):+8.2f}          {statistics.fmean( strs ):+9.0f}  "
               f"{statistics.fmean( outs ):+8.2f}         {sum( wins )}" )
    print( "\n(d_* are treatment minus the built-in control of the same seed/color; the castle term of the GRPO reward" )
    print( " trains the build answers to keep d_k up.)" )
    return


if __name__ == "__main__":
    sys.exit( main() )
