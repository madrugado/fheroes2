"""Rule mining: candidate whole-game rules from recorded games (step 1 of the rule mechanism).

Input: strategic value trajectories (strategy_value.py / strategy_loop: one record per player and game with the
player's turn_context of every day, the answered decisions and the final duel label "final" in [-1, 1]).

For a checkpoint day d every player-game becomes a row of interpretable strategy features (army concentration
in the strongest hero, number of heroes, the strongest hero's level, skills, artifacts, spell book, buildings,
unspent gold, castles, ...). Raw associations with the final label are dominated by "the stronger player does
everything more": the label is first regressed on the player's strength at day d (log army strength, castles),
and every feature is related to the RESIDUAL — what goes with a better final duel at the SAME strength. Each
feature gets a bootstrap CI over seeds (the two players of a seed share the game). Features with a clear residual
association are rule candidates; they are NOT causal: turn a candidate into a plan key and test it with
plan_race.py (paired games on fresh seeds).

  rl/.venv/bin/python rl/rule_mining.py --data rl/data/strategy_value_final5_2kings.jsonl --days 7,14,21
"""

import argparse
import json
import math
import random
from collections import defaultdict

import numpy as np

# Secondary skills in turn_context "sk" order (Skill::Secondary::PATHFINDING .. ESTATES).
SKILLS = ("pathfinding", "archery", "logistics", "scouting", "diplomacy", "navigation", "leadership", "wisdom",
          "mysticism", "luck", "ballistics", "eagle_eye", "necromancy", "estates")
MAGEGUILD1 = 0x00004000


def popcount( value: int ) -> int:
    return bin( value & 0xFFFFFFFF ).count( "1" )


def day_features( day: dict, decisions: list[dict] ) -> dict[str, float]:
    """Strategy features of one player at one turn_context (and the decisions answered up to that day)."""
    heroes = day.get( "heroes", [] )
    castles = day.get( "castles", [] )
    strengths = [h.get( "str", 0.0 ) for h in heroes]
    total = sum( strengths ) + sum( sum( s[1] * s[2] for s in c.get( "army", [] ) ) for c in castles )
    top = max( heroes, key=lambda h: h.get( "str", 0.0 ) ) if heroes else None
    res = day.get( "res", [0] * 7 )
    f: dict[str, float] = {
        "heroes": len( heroes ),
        "castles": len( castles ),
        "top_share": ( top["str"] / total ) if top and total > 0 else 0.0,
        "garrison_share": ( sum( sum( s[1] * s[2] for s in c.get( "army", [] ) ) for c in castles ) / total ) if total > 0 else 0.0,
        "gold_k": res[6] / 1000.0,
        "rare_res": res[1] + res[3] + res[4] + res[5],
        "buildings": sum( popcount( c.get( "b", 0 ) ) for c in castles ),
        "dwellings": sum( 1 for c in castles for dw in c.get( "dw", [] ) if dw and dw[0] ),
        "guilds": sum( 1 for c in castles if c.get( "b", 0 ) & MAGEGUILD1 ),
    }
    if top is not None:
        army = top.get( "army", [] )
        f.update( {
            "top_level": top.get( "lvl", 0 ),
            "top_attack": top.get( "a", 0 ),
            "top_defense": top.get( "d", 0 ),
            "top_power": top.get( "pw", 0 ),
            "top_knowledge": top.get( "k", 0 ),
            "top_book": top.get( "book", 0 ),
            "top_spell_points": top.get( "msp", 0 ),
            "top_artifacts": len( top.get( "art", [] ) ),
            "top_morale": top.get( "mor", 0 ),
            "top_luck": top.get( "luck", 0 ),
            "top_move": top.get( "mmp", 0 ) / 100.0,
            "top_stacks": len( army ),
            "top_slowest": min( ( s[4] for s in army ), default=0 ),
            "top_shooter_share": ( sum( s[1] * s[2] for s in army if s[5] ) / max( sum( s[1] * s[2] for s in army ), 1e-9 ) ),
            "top_max_level": max( ( s[3] for s in army ), default=0 ),
        } )
        for name, level in zip( SKILLS, top.get( "sk", [] ) ):
            f[f"top_{name}"] = level
    hires = sum( 1 for d in decisions if d["kind"] == "hire" and d.get( "answer" ) not in ( None, 0 ) )
    f["hire_answers"] = hires
    return f


def strength_controls( day: dict ) -> list[float]:
    heroes = day.get( "heroes", [] )
    castles = day.get( "castles", [] )
    total = sum( h.get( "str", 0.0 ) for h in heroes ) + sum( sum( s[1] * s[2] for s in c.get( "army", [] ) ) for c in castles )
    return [1.0, math.log1p( total ), float( len( castles ) )]


def rows_at( records: list[dict], day: int ):
    rows, controls, labels, groups = [], [], [], []
    for record in records:
        days = {d["t"]: d for d in record.get( "days", [] )}
        if day not in days:
            continue
        decisions = [d for d in record.get( "decisions", [] ) if d["event"].get( "t", 0 ) <= day]
        rows.append( day_features( days[day], decisions ) )
        controls.append( strength_controls( days[day] ) )
        labels.append( float( record["final"] ) )
        groups.append( ( record.get( "map" ), record.get( "seed" ) ) )
    return rows, np.array( controls ), np.array( labels ), groups


def residual_associations( rows, controls, labels, groups, boot: int, rng: random.Random ):
    names = sorted( {key for row in rows for key in row} )
    x = np.array( [[row.get( name, 0.0 ) for name in names] for row in rows], dtype=np.float64 )
    beta, *_ = np.linalg.lstsq( controls, labels, rcond=None )
    residual = labels - controls @ beta

    def slopes( index: np.ndarray ) -> np.ndarray:
        # The residual's association with each feature, also residualized on the strength controls:
        # standardized slope (label units per feature sd).
        c, r, xs = controls[index], residual[index], x[index]
        bx, *_ = np.linalg.lstsq( c, xs, rcond=None )
        xr = xs - c @ bx
        sd = xr.std( axis=0 )
        sd[sd == 0] = np.inf
        return ( xr * ( r - r.mean() )[:, None] ).mean( axis=0 ) / sd

    point = slopes( np.arange( len( labels ) ) )
    by_group = defaultdict( list )
    for i, group in enumerate( groups ):
        by_group[group].append( i )
    keys = list( by_group )
    samples = []
    for _ in range( boot ):
        index = np.array( [i for key in rng.choices( keys, k=len( keys ) ) for i in by_group[key]] )
        samples.append( slopes( index ) )
    samples = np.sort( np.array( samples ), axis=0 )
    low, high = samples[int( 0.025 * boot )], samples[int( 0.975 * boot ) - 1]
    return [( names[i], float( point[i] ), float( low[i] ), float( high[i] ), float( x[:, i].mean() ) ) for i in range( len( names ) )]


def main() -> None:
    parser = argparse.ArgumentParser( description="Rule candidates: strategy features that go with a better final duel at the same strength" )
    parser.add_argument( "--data", nargs="+", required=True )
    parser.add_argument( "--days", default="7,14,21" )
    parser.add_argument( "--boot", type=int, default=500 )
    parser.add_argument( "--top", type=int, default=15 )
    parser.add_argument( "--out", default="" )
    args = parser.parse_args()

    records = []
    for path in args.data:
        with open( path ) as f:
            records.extend( json.loads( line ) for line in f if line.strip() )
    rng = random.Random( 0 )
    report = {}
    for day in [int( d ) for d in args.days.split( "," )]:
        rows, controls, labels, groups = rows_at( records, day )
        if len( rows ) < 50:
            continue
        found = residual_associations( rows, controls, labels, groups, args.boot, rng )
        clear = sorted( ( a for a in found if a[2] > 0 or a[3] < 0 ), key=lambda a: -abs( a[1] ) )
        print( f"== day {day}: {len( rows )} player-games; features clearly associated with the final duel at the same strength:" )
        for name, slope, low, high, mean in clear[: args.top]:
            print( f"   {name:22s} {slope:+.3f} per sd [{low:+.3f}, {high:+.3f}]  (mean {mean:.2f})" )
        report[day] = [dict( zip( ( "feature", "slope", "low", "high", "mean" ), a ) ) for a in found]
    if args.out:
        with open( args.out, "w" ) as f:
            json.dump( report, f, indent=1 )


if __name__ == "__main__":
    main()
