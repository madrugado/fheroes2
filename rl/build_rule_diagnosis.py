"""Diagnosis of the build query: does the built-in AI's dry-run pick ("bi") follow from information the query
carries? Replicates CastleDevelopment (ai_planner_castle.cpp) for every collected build query (kind="build" in
the strategy_game_grpo collect files): the per-race build orders with their funds multipliers (the savings rule:
build only when funds >= cost * priority), the well-on-day-7 rule, income structures, guild upgrades and the
defensive fallbacks — and measures the exact-match with "bi".

Known replica gaps (mismatches are bucketed): TRADE (the marketplace greedy is approximated by direct
affordability), SHIPYARD (the island branch needs the region's neighbour count, unknown here), GUILD
(the spellLevel/safetyFactor branch runs with a fixed guild level cap in the ON variant).

    python rl/build_rule_diagnosis.py rl/data/grpo1/games_bf_c1.jsonl rl/data/grpo1/games_bf_c2.jsonl [--limit N]
"""

import argparse
import json
import sys

# BuildingType bits (castle.h).
WELL, STATUE, MARKETPLACE, WEL2, MOAT, SPEC, CASTLE, CAPTAIN = 0x8, 0x10, 0x80, 0x100, 0x200, 0x400, 0x800, 0x1000
SHRINE, SHIPYARD, TAVERN, THIEVESGUILD = 0x2000, 0x4, 0x2, 0x1
LEFTTURRET, RIGHTTURRET = 0x20, 0x40
MAGEGUILD1, MAGEGUILD2, MAGEGUILD3, MAGEGUILD4, MAGEGUILD5 = 0x4000, 0x8000, 0x10000, 0x20000, 0x40000
DW1, DW2, DW3, DW4, DW5, DW6 = 0x100000, 0x200000, 0x400000, 0x800000, 0x1000000, 0x2000000
UP2, UP3, UP4, UP5, UP6, UP7 = 0x4000000, 0x8000000, 0x10000000, 0x20000000, 0x40000000, 0x80000000

# GetBuildOrder (ai_planner_castle.cpp lines 78-122): per race, in the built-in's order, with the funds multiplier.
GENERIC = [ ( CASTLE, 2 ), ( STATUE, 1 ), ( MARKETPLACE, 1 ), ( UP7, 1 ), ( UP6, 1 ), ( DW6, 1 ), ( UP5, 1 ), ( DW5, 1 ), ( UP4, 1 ),
            ( DW4, 1 ), ( UP3, 2 ), ( DW3, 2 ), ( UP2, 3 ), ( DW2, 3 ), ( DW1, 4 ), ( MAGEGUILD1, 2 ), ( WEL2, 10 ), ( TAVERN, 5 ),
            ( THIEVESGUILD, 10 ), ( MAGEGUILD2, 3 ), ( MAGEGUILD3, 4 ), ( MAGEGUILD4, 5 ), ( MAGEGUILD5, 5 ), ( SHIPYARD, 4 ) ]
RACE_ORDERS = {
    1: [ ( CASTLE, 2 ), ( STATUE, 1 ), ( MARKETPLACE, 1 ), ( UP6, 2 ), ( DW6, 1 ), ( UP5, 2 ), ( DW5, 2 ), ( UP4, 2 ), ( DW4, 1 ),
         ( UP3, 2 ), ( DW3, 1 ), ( UP2, 1 ), ( DW2, 3 ), ( DW1, 4 ), ( WELL, 1 ), ( TAVERN, 1 ), ( MAGEGUILD1, 2 ), ( MAGEGUILD2, 3 ),
         ( MAGEGUILD3, 5 ), ( MAGEGUILD4, 5 ), ( MAGEGUILD5, 5 ), ( SPEC, 5 ), ( THIEVESGUILD, 10 ), ( WEL2, 20 ), ( SHIPYARD, 4 ) ],
    2: [ ( CASTLE, 2 ), ( STATUE, 1 ), ( MARKETPLACE, 1 ), ( DW6, 1 ), ( UP5, 1 ), ( DW5, 1 ), ( UP4, 1 ), ( DW4, 1 ), ( DW3, 1 ),
         ( UP2, 2 ), ( DW2, 2 ), ( DW1, 4 ), ( MAGEGUILD1, 3 ), ( WEL2, 10 ), ( TAVERN, 5 ), ( THIEVESGUILD, 10 ), ( MAGEGUILD2, 4 ),
         ( MAGEGUILD3, 5 ), ( MAGEGUILD4, 6 ), ( MAGEGUILD5, 7 ), ( SHIPYARD, 4 ) ],
    4: [ ( CASTLE, 2 ), ( STATUE, 1 ), ( MARKETPLACE, 1 ), ( DW6, 1 ), ( DW5, 1 ), ( DW4, 1 ), ( MAGEGUILD1, 1 ), ( DW3, 1 ), ( UP4, 1 ),
         ( UP3, 2 ), ( UP2, 5 ), ( DW2, 2 ), ( TAVERN, 2 ), ( DW1, 4 ), ( WEL2, 10 ), ( THIEVESGUILD, 10 ), ( MAGEGUILD2, 3 ),
         ( MAGEGUILD3, 4 ), ( MAGEGUILD4, 5 ), ( MAGEGUILD5, 5 ), ( SHIPYARD, 4 ) ],
    8: [ ( CASTLE, 2 ), ( STATUE, 1 ), ( MARKETPLACE, 1 ), ( UP7, 1 ), ( UP6, 1 ), ( DW6, 1 ), ( UP5, 1 ), ( DW5, 1 ), ( UP4, 1 ),
         ( DW4, 1 ), ( UP3, 2 ), ( DW3, 2 ), ( UP2, 3 ), ( DW2, 3 ), ( DW1, 4 ), ( MAGEGUILD1, 2 ), ( WEL2, 10 ), ( TAVERN, 5 ),
         ( THIEVESGUILD, 10 ), ( MAGEGUILD2, 3 ), ( MAGEGUILD3, 4 ), ( MAGEGUILD4, 5 ), ( MAGEGUILD5, 5 ), ( SHIPYARD, 4 ) ],
    16: [ ( CASTLE, 2 ), ( STATUE, 1 ), ( MARKETPLACE, 1 ), ( UP6, 1 ), ( DW6, 1 ), ( UP5, 1 ), ( DW5, 1 ), ( DW4, 1 ), ( DW3, 1 ),
          ( DW2, 1 ), ( DW1, 1 ), ( MAGEGUILD1, 1 ), ( UP3, 4 ), ( SPEC, 2 ), ( WEL2, 8 ), ( MAGEGUILD2, 3 ), ( MAGEGUILD3, 4 ),
          ( MAGEGUILD4, 4 ), ( MAGEGUILD5, 4 ), ( TAVERN, 10 ), ( THIEVESGUILD, 10 ), ( SHIPYARD, 4 ) ],
    32: [ ( CASTLE, 2 ), ( STATUE, 1 ), ( MARKETPLACE, 1 ), ( UP6, 1 ), ( DW6, 1 ), ( UP5, 2 ), ( DW5, 1 ), ( MAGEGUILD1, 1 ), ( UP4, 2 ),
          ( DW4, 1 ), ( UP3, 3 ), ( DW3, 3 ), ( UP2, 4 ), ( DW2, 2 ), ( DW1, 3 ), ( MAGEGUILD2, 2 ), ( THIEVESGUILD, 2 ), ( WEL2, 8 ),
          ( MAGEGUILD3, 4 ), ( MAGEGUILD4, 5 ), ( MAGEGUILD5, 5 ), ( SHRINE, 10 ), ( SHIPYARD, 4 ) ],
}
INCOME = [ ( CASTLE, 2 ), ( STATUE, 1 ), ( MARKETPLACE, 1 ) ]
GUILD_UPGRADES = [ ( MAGEGUILD2, 2 ), ( MAGEGUILD3, 2 ), ( MAGEGUILD4, 1 ), ( MAGEGUILD5, 1 ) ]
DEFENSIVE = [ ( LEFTTURRET, 1 ), ( RIGHTTURRET, 1 ), ( MOAT, 1 ), ( CAPTAIN, 1 ) ]
SUPPORT_DEFENSIVE = [ ( MAGEGUILD1, 1 ), ( SPEC, 2 ), ( TAVERN, 1 ) ]
WRLK_EXTRA = ( SPEC, 1 )  # appended to the income structures for Race::WRLK
# resource_trading.cpp: the trade rates at one marketplace (index 0 of each table).
RATE_RES_TO_RES = [ 10, 7, 5, 4, 4, 3, 3, 3, 2 ]
RATE_COMMON_TO_RARE = [ 20, 14, 10, 8, 7, 6, 5, 5, 4 ]
RATE_RARE_TO_COMMON = [ 5, 4, 3, 2, 2, 2, 2, 2, 1 ]
RATE_COMMON_TO_GOLD = [ 25, 37, 50, 62, 74, 87, 100, 112, 124 ]
RATE_RARE_TO_GOLD = [ 50, 74, 100, 124, 149, 175, 200, 224, 249 ]
RATE_GOLD_TO_COMMON = [ 2500, 1667, 1250, 1000, 834, 715, 625, 556, 500 ]
RATE_GOLD_TO_RARE = [ 5000, 3334, 2500, 2000, 1667, 1429, 1250, 1112, 1000 ]
RARE_SET = frozenset( ( 1, 3, 4, 5 ) )  # mercury, sulfur, crystal, gems; wood/ore are common, gold is its own
RES_ORDER = ( 0, 1, 2, 3, 4, 5, 6 )  # wood, mercury, ore, sulfur, crystal, gems, gold (writeFunds order; the enum keys of std::map)


def trade_cost( marketplace: int, from_res: int, to_res: int ) -> int:
    """getTradeCost (resource_trading.cpp): the rate tables are indexed by marketplaceCount - 1 (capped at 9)."""
    idx = min( marketplace, 9 ) - 1
    common_from, common_to = from_res not in RARE_SET, to_res not in RARE_SET
    if to_res == 6:  # GOLD
        return RATE_COMMON_TO_GOLD[idx] if common_from else RATE_RARE_TO_GOLD[idx]
    if from_res == 6:  # GOLD
        return RATE_GOLD_TO_COMMON[idx] if common_to else RATE_GOLD_TO_RARE[idx]
    if common_from == common_to:
        return RATE_RES_TO_RES[idx]  # common->common and rare->rare share one table
    return RATE_COMMON_TO_RARE[idx] if common_from else RATE_RARE_TO_COMMON[idx]


def trade_transaction( res: list, required: list, marketplace: int ) -> bool:
    """calculateMarketplaceTransaction (ai_common.cpp): greedy per-unit exchange over the planned balance
    (funds - required), missing resources in enum order, the donor with the maximum potential revenue each
    step. marketplace = the KINGDOM's marketplace count (all own castles, GetCountMarketplace); 0 never trades."""
    if marketplace <= 0:
        return False
    balance = [ res[i] - required[i] for i in range( 7 ) ]
    for mi in RES_ORDER:
        while balance[mi] < 0:
            best, best_revenue = None, 0
            for ri in RES_ORDER:
                if ri == mi or balance[ri] <= 0:
                    continue
                rate = trade_cost( marketplace, ri, mi )
                revenue = balance[ri] * rate if mi == 6 else balance[ri] // rate
                if revenue > best_revenue:
                    best, best_revenue = ri, revenue
            if best is None:
                return False
            rate = trade_cost( marketplace, best, mi )
            if mi == 6:
                balance[best] -= 1
                balance[mi] += rate
            else:
                balance[best] -= rate
                balance[mi] += 1
    return True


def replica_pick( event: dict, mask: int, is_castle: bool, guild_cap: int | None, marketplace: int ) -> tuple:
    """The built-in pick (a building bit or 0) and which branch decided it. `marketplace` = the kingdom's
    marketplace count over ALL own castles (GetCountMarketplace), not just this one."""
    race, t = event.get( "race" ), event.get( "t", 0 )
    res = event["res"]
    costs = { c["b"]: c["cost"] for c in event.get( "cands", [] ) }

    def afford( bit: int, mult: int ) -> bool:
        cost = costs.get( bit )
        if cost is None:
            return False  # not offered: already built, blocked, or not affordable
        required = [ cost[i] * mult for i in range( 7 ) ]
        if all( res[i] >= required[i] for i in range( 7 ) ):
            return True
        return trade_transaction( res, required, marketplace )

    def walk( orders: list, scale: int = 1 ) -> int:
        for bit, prio in orders:
            if mask & bit:
                continue  # already built
            if afford( bit, prio * scale ):
                return bit
            # not affordable at prio x: skipped, the walk goes on (the savings rule)
        return 0

    if is_castle and not ( mask & WELL ) and t > 6 and afford( WELL, 1 ):
        return WELL, "well"
    orders = list( INCOME ) + ( [ WRLK_EXTRA ] if race == 8 else [] )
    pick = walk( orders )
    if pick:
        return pick, "income"
    pick = walk( RACE_ORDERS.get( race, [] ) )
    if pick:
        return pick, "main"
    if guild_cap is not None and sum( 1 for m in ( MAGEGUILD1, MAGEGUILD2, MAGEGUILD3, MAGEGUILD4, MAGEGUILD5 ) if mask & m ) < guild_cap:
        pick = walk( GUILD_UPGRADES )
        if pick:
            return pick, "guild"
    pick = walk( DEFENSIVE, 10 )
    if pick:
        return pick, "defensive"
    pick = walk( SUPPORT_DEFENSIVE, 10 )
    if pick:
        return pick, "support"
    return 0, "nothing"


def main() -> None:
    parser = argparse.ArgumentParser( description=__doc__ )
    parser.add_argument( "data", nargs="+" )
    parser.add_argument( "--limit", type=int, default=0, help="stop after this many build queries" )
    parser.add_argument( "--dump", type=int, default=0, help="print this many mismatch events verbatim" )
    args = parser.parse_args()

    total = matched = trade_mismatch = dumped = 0
    branch_counts: dict = {}
    mismatches: dict = {}
    nothing_explained = nothing_total = 0
    cap_matched = {}
    for cap in ( None, 2 ):
        cap_matched[cap] = [ 0, 0 ]
    for path in args.data:
        with open( path ) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    group = json.loads( line )
                except json.JSONDecodeError:
                    continue
                for game in group["games"]:
                    days = game["days"]
                    for answer in game["answers"]:
                        if answer["kind"] != "build":
                            continue
                        event = answer["event"]
                        if event.get( "defensive", 0 ):
                            continue  # the defensive path has its own logic
                        ctx = days[answer["ctx"]] if 0 <= answer.get( "ctx", -1 ) < len( days ) else None
                        castle_record = next( ( c for c in ( ctx.get( "castles" ) or [] ) if c.get( "i" ) == event.get( "castle" ) ), None ) if ctx else None
                        if castle_record is None:
                            continue
                        mask, is_castle = int( castle_record.get( "b", 0 ) ), bool( castle_record.get( "castle" ) )
                        # the kingdom's marketplace count over ALL own castles (GetCountMarketplace), capped at 9
                        marketplace = min( sum( 1 for c in ( ctx.get( "castles" ) or [] ) if int( c.get( "b", 0 ) ) & MARKETPLACE ), 9 )
                        bi = int( event.get( "bi", 0 ) )
                        total += 1
                        for cap in ( None, 2 ):
                            pick, branch = replica_pick( event, mask, is_castle, cap, marketplace )
                            ok = pick == bi
                            cap_matched[cap][0] += ok
                            cap_matched[cap][1] += 1
                            if cap is not None:
                                continue
                            branch_counts[branch] = branch_counts.get( branch, 0 ) + 1
                            if bi == 0:
                                nothing_total += 1
                                nothing_explained += ok
                            if not ok:
                                trades = { c["b"] for c in event.get( "cands", [] ) if c.get( "trade" ) }
                                if pick in trades or bi in trades:
                                    trade_mismatch += 1
                                mismatches[( branch, pick, bi )] = mismatches.get( ( branch, pick, bi ), 0 ) + 1
                                if dumped < args.dump:
                                    dumped += 1
                                    print( f"--- mismatch #{dumped}: replica {branch} pick {pick}, bi {bi}; race {event.get('race')} t {event.get('t')}" )
                                    print( f"    mask {mask:#x} isCastle {is_castle} res {event['res']}" )
                                    print( "    cands:", [ ( c['b'], c['cost'], c.get( 'trade', 0 ) ) for c in event.get( 'cands', [] ) ] )
                        if args.limit and total >= args.limit:
                            break
                    if args.limit and total >= args.limit:
                        break
                if args.limit and total >= args.limit:
                    break
            if args.limit and total >= args.limit:
                break

    print( f"build queries (defensive=0): {total}" )
    for cap in ( None, 2 ):
        hits, seen = cap_matched[cap]
        label = "guild cap none" if cap is None else "guild cap 2 (spellLevel default)"
        print( f"  exact match with bi, {label}: {hits}/{seen} ({hits / max( seen, 1 ):.1%})" )
    print( f"built-in 'nothing' explained: {nothing_explained}/{nothing_total} ({nothing_explained / max( nothing_total, 1 ):.1%})" )
    print( f"mismatches involving a trade-flagged candidate: {trade_mismatch}" )
    print( "replica branch of the pick:", { k: v for k, v in sorted( branch_counts.items(), key=lambda kv: -kv[1] ) } )
    print( "top mismatches (branch, replica pick, bi):" )
    for key, count in sorted( mismatches.items(), key=lambda kv: -kv[1] )[:15]:
        print( f"  {key}: x{count}" )


if __name__ == "__main__":
    sys.exit( main() )
