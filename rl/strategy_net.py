"""Strategic decisions with the unified battle transformer (masked output over the answer options).

The strategic layer (hero targets, castle building, hero hiring, army budget; see "Strategic
protocol" in AGENTS.md) is answered by the SAME transformer that plays battles
(transformer_model.AzBattleTransformer): a query becomes one context token plus one token per
answer option (the features of strategy_model.py, padded to a fixed width, with the query kind
one-hot), the body scores every option (transformer_model.strategic_logits) and a softmax over the
options of this query — the masked output — picks the answer.

Data:
  sft   — the built-in AI's answer to EVERY query of seeded base games (strategy_rollout.Branch
          with built-in answers; build: the base game's build_result): imitation targets.
  prefs — DPO pairs from the exact counterfactual labels of strategy_rollout.py: per labeled query
          the best and the worst option by label_score (the built-in option has label 0).

Usage:
    rl/.venv/bin/python rl/strategy_net.py sft --map 2kings.mp2 --days 30 --seeds 1-40 --out rl/data/strategy_sft_2kings.jsonl
    rl/.venv/bin/python rl/strategy_net.py prefs --data rl/data/strategy_rollouts_all_Battlefi_h7.jsonl --out rl/data/strategy_prefs.jsonl
    (training: rl/train_strategy_net.py)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from array import array
from collections import Counter, OrderedDict, defaultdict

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

from strategy_model import MAX_OBJ_VOCAB, context_features, label_score, option_features, options_of  # noqa: E402
from strategy_policies import NOTHING  # noqa: E402
import encoding as enc  # noqa: E402
from transformer_model import STRAT_FEATURES, STRAT_KINDS, STRAT_MON_SLOTS, STRAT_TOKEN_TYPES  # noqa: E402

# The whole strategic sequence (history, today's snapshot, context, options twice) fits the model's
# window (user decision: 512 tokens — a 45-day 2kings game needs at most ~490; longer games lose
# their oldest days).
MAX_STRATEGIC_TOKENS = 512
MAX_CACHED_ANSWERS = 200_000  # NetStrategyPolicy answer cache (least recently used dropped)

TARGET_TOP = 8  # hero-target candidates offered to the network (sorted by value; 0 = built-in)

# The built-in AI's building logic (src/fheroes2/ai/ai_planner_castle.cpp, GetBuildOrder /
# GetIncomeStructures / defensiveStructures): a per-race priority list; a building is taken when
# the kingdom has cost x priority of every resource. Mirrored here as build-option features — the
# network cannot imitate the built-in choice without knowing its priorities (SFT accuracy on
# building was 0.77 without them, 1.0 on the other kinds).
_B = {"THIEVESGUILD": 0x1, "TAVERN": 0x2, "SHIPYARD": 0x4, "WELL": 0x8, "STATUE": 0x10, "LEFTTURRET": 0x20,
      "RIGHTTURRET": 0x40, "MARKETPLACE": 0x80, "WEL2": 0x100, "MOAT": 0x200, "SPEC": 0x400, "CASTLE": 0x800,
      "CAPTAIN": 0x1000, "SHRINE": 0x2000, "MAGEGUILD1": 0x4000, "MAGEGUILD2": 0x8000, "MAGEGUILD3": 0x10000,
      "MAGEGUILD4": 0x20000, "MAGEGUILD5": 0x40000, "DWELLING1": 0x100000, "DWELLING2": 0x200000,
      "DWELLING3": 0x400000, "DWELLING4": 0x800000, "DWELLING5": 0x1000000, "DWELLING6": 0x2000000,
      "UPGRADE2": 0x4000000, "UPGRADE3": 0x8000000, "UPGRADE4": 0x10000000, "UPGRADE5": 0x20000000,
      "UPGRADE6": 0x40000000, "UPGRADE7": 0x80000000}


def _order( text: str ) -> list[tuple[int, int]]:
    return [( _B[name], int( prio ) ) for name, prio in ( item.split( ":" ) for item in text.split() )]


_GENERIC = _order( "CASTLE:2 STATUE:1 MARKETPLACE:1 UPGRADE7:1 UPGRADE6:1 DWELLING6:1 UPGRADE5:1 DWELLING5:1 UPGRADE4:1 "
                   "DWELLING4:1 UPGRADE3:2 DWELLING3:2 UPGRADE2:3 DWELLING2:3 DWELLING1:4 MAGEGUILD1:2 WEL2:10 TAVERN:5 "
                   "THIEVESGUILD:10 MAGEGUILD2:3 MAGEGUILD3:4 MAGEGUILD4:5 MAGEGUILD5:5 SHIPYARD:4" )
BUILD_ORDERS = {  # Race::KNGT 1, BARB 2, SORC 4, WRLK 8 (generic), WZRD 16, NECR 32
    1: _order( "CASTLE:2 STATUE:1 MARKETPLACE:1 UPGRADE6:2 DWELLING6:1 UPGRADE5:2 DWELLING5:2 UPGRADE4:2 DWELLING4:1 "
               "UPGRADE3:2 DWELLING3:1 UPGRADE2:1 DWELLING2:3 DWELLING1:4 WELL:1 TAVERN:1 MAGEGUILD1:2 MAGEGUILD2:3 "
               "MAGEGUILD3:5 MAGEGUILD4:5 MAGEGUILD5:5 SPEC:5 THIEVESGUILD:10 WEL2:20 SHIPYARD:4" ),
    2: _order( "CASTLE:2 STATUE:1 MARKETPLACE:1 DWELLING6:1 UPGRADE5:1 DWELLING5:1 UPGRADE4:1 DWELLING4:1 DWELLING3:1 "
               "UPGRADE2:2 DWELLING2:2 DWELLING1:4 MAGEGUILD1:3 WEL2:10 TAVERN:5 THIEVESGUILD:10 MAGEGUILD2:4 MAGEGUILD3:5 "
               "MAGEGUILD4:6 MAGEGUILD5:7 SHIPYARD:4" ),
    4: _order( "CASTLE:2 STATUE:1 MARKETPLACE:1 DWELLING6:1 DWELLING5:1 DWELLING4:1 MAGEGUILD1:1 DWELLING3:1 UPGRADE4:1 "
               "UPGRADE3:2 UPGRADE2:5 DWELLING2:2 TAVERN:2 DWELLING1:4 WEL2:10 THIEVESGUILD:10 MAGEGUILD2:3 MAGEGUILD3:4 "
               "MAGEGUILD4:5 MAGEGUILD5:5 SHIPYARD:4" ),
    8: _GENERIC,
    16: _order( "CASTLE:2 STATUE:1 MARKETPLACE:1 UPGRADE6:1 DWELLING6:1 UPGRADE5:1 DWELLING5:1 DWELLING4:1 DWELLING3:1 "
                "DWELLING2:1 DWELLING1:1 MAGEGUILD1:1 UPGRADE3:4 SPEC:2 WEL2:8 MAGEGUILD2:3 MAGEGUILD3:4 MAGEGUILD4:4 "
                "MAGEGUILD5:4 TAVERN:10 THIEVESGUILD:10 SHIPYARD:4" ),
    32: _order( "CASTLE:2 STATUE:1 MARKETPLACE:1 UPGRADE6:1 DWELLING6:1 UPGRADE5:2 DWELLING5:1 MAGEGUILD1:1 UPGRADE4:2 "
                "DWELLING4:1 UPGRADE3:3 DWELLING3:3 UPGRADE2:4 DWELLING2:2 DWELLING1:3 MAGEGUILD2:2 THIEVESGUILD:2 WEL2:8 "
                "MAGEGUILD3:4 MAGEGUILD4:5 MAGEGUILD5:5 SHRINE:10 SHIPYARD:4" ),
}
INCOME = {_B["CASTLE"], _B["STATUE"], _B["MARKETPLACE"]}
DEFENSIVE = {_B["LEFTTURRET"], _B["RIGHTTURRET"], _B["MOAT"], _B["CAPTAIN"], _B["MAGEGUILD1"], _B["SPEC"], _B["TAVERN"]}


def build_priority_features( event: dict, option ) -> list[float]:
    """rank and priority of the building in its race's built-in order, income/defensive flags, the
    smallest resources/cost ratio over all resources and whether it covers cost x priority."""
    if option == NOTHING:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    order = BUILD_ORDERS.get( event.get( "race" ), _GENERIC )
    positions = {b: ( i, prio ) for i, ( b, prio ) in enumerate( order )}
    rank, prio = positions.get( option["b"], ( len( order ), 20 ) )
    res = event.get( "res" ) or [0] * 7
    cost = option.get( "cost" ) or [0] * 7
    ratios = [float( r ) / c for r, c in zip( res, cost ) if c > 0]
    min_ratio = min( ratios ) if ratios else 10.0
    return [
        rank / 30.0,
        prio / 20.0,
        1.0 if option["b"] in INCOME else 0.0,
        1.0 if option["b"] in DEFENSIVE else 0.0,
        math.log1p( min( min_ratio, 50.0 ) ),
        1.0 if min_ratio >= prio else 0.0,
        0.0,
    ]


def _pad( features: list[float] ) -> list[float]:
    if len( features ) > STRAT_FEATURES:
        raise ValueError( f"{len( features )} strategic features > STRAT_FEATURES ({STRAT_FEATURES})" )
    return features + [0.0] * ( STRAT_FEATURES - len( features ) )


def _kind_hot( kind: str | None ) -> list[float]:
    return [1.0 if kind == k else 0.0 for k in STRAT_KINDS]


def _type_hot( token_type: str, army: list | None = None ) -> list[float]:
    """The token type one-hot and the creatures of the token's army (_mon_slots): the tail of every
    strategic token."""
    return [1.0 if token_type == t else 0.0 for t in STRAT_TOKEN_TYPES] + _mon_slots( army )


def _mon_slots( army: list | None ) -> list[float]:
    """STRAT_MON_SLOTS creatures (enc.monster_token, 0: empty slot) from the engine's [monster, count, ...]
    stacks — the same encoding as the battle state's "mon", embedded by the same mon_embed."""
    ids = [enc.monster_token( stack[0] ) for stack in ( army or [] )[:STRAT_MON_SLOTS] if stack]
    return ids + [0.0] * ( STRAT_MON_SLOTS - len( ids ) )


# "Let the built-in AI decide" — the first option of every build query (answered with `skip`).
# The built-in building choice depends on inputs the query does not carry (marketplace deals,
# safety factor, regions), so imitation of explicit buildings topped out at 0.81; deferring is
# exact, and DPO learns when an explicit building (or nothing) beats the built-in development.
BUILTIN = "builtin"


def query_options( kind: str, event: dict ) -> list:
    options = options_of( kind, event, TARGET_TOP )
    return [BUILTIN] + options if kind == "build" else options


def builtin_option( kind: str, options: list, event: dict ) -> int | None:
    """Index of the built-in AI's answer among query_options (target: the top candidate, army: 100%,
    hire: `bi` or "nobody", build: "let the built-in AI decide")."""
    if kind == "target":
        return 0
    if kind == "army":
        return options.index( 100 ) if 100 in options else None
    if kind == "hire":
        bi = event.get( "bi", -1 )
        return bi if bi >= 0 else len( options ) - 1
    return options.index( BUILTIN ) if BUILTIN in options else None


def answer_of( option ):
    """The strategic policy answer of an option (BUILTIN -> None: the built-in AI decides)."""
    return None if option == BUILTIN else option


def _option_features( kind: str, event: dict, context: dict | None, option, index: int, obj_vocab: list[int] ) -> list[float]:
    if option == BUILTIN:
        return [0.0] * ( STRAT_FEATURES - 1 ) + [1.0]  # the last feature slot marks "built-in decides"
    extra = []
    if kind == "build":
        extra = build_priority_features( event, option )
    elif kind == "target":
        extra = target_map_features( option, context )
    return _pad( option_features( kind, event, context, option, index, obj_vocab ) + extra )


def target_map_features( option: dict, context: dict | None ) -> list[float]:
    """Where a hero target lies relative to what the player sees: distances to the nearest visible
    enemy hero, castle of another owner and own castle (1.0 = none seen)."""
    context = context or {}
    width = int( context.get( "w" ) or 0 )
    index = option.get( "i" )
    if index is None or not width:
        return [0.0, 0.0, 0.0]
    groups = ( [r.get( "i" ) for r in context.get( "rivals" ) or []], [c.get( "i" ) for c in context.get( "rcastles" ) or []],
               [c.get( "i" ) for c in context.get( "castles" ) or []] )
    return [_nearest( index, group, width ) if group else 1.0 for group in groups]


# Every strategic token carries its day at this slot (t/30); the last slot flags BUILTIN options.
DAY_SLOT = STRAT_FEATURES - 2
HERO_SLOTS = 5  # army slots of a hero (own and rival stacks)
DWELLINGS = 6
RACE_HOT = ( 1, 2, 4, 8, 16, 32, 128 )  # Race::KNGT, BARB, SORC, WRLK, WZRD, NECR, RAND (neutral castles)


def _dated( features: list[float], day ) -> list[float]:
    features = _pad( features )
    if any( features[DAY_SLOT:] ):
        raise ValueError( "strategic features overlap the day slot" )
    features[DAY_SLOT] = float( day or 0 ) / 30.0
    return features


def _race_hot( race ) -> list[float]:
    return [1.0 if race == r else 0.0 for r in RACE_HOT]


def _stack_features( army: list | None ) -> list[float]:
    """HERO_SLOTS x [log count, log strength of one creature, speed, shooter, flyer] from the engine's
    [monster, count, strength, level, speed, shooter, flyer] stacks (count 0 = not shown: types
    only; older 2-field records give counts only)."""
    features: list[float] = []
    for stack in ( army or [] )[:HERO_SLOTS]:
        stack = list( stack ) + [0] * ( 7 - len( stack ) )
        features += [math.log1p( float( stack[1] ) ), math.log1p( float( stack[2] ) ), float( stack[4] ) / 10.0, float( stack[5] ), float( stack[6] )]
    return features + [0.0] * ( 5 * HERO_SLOTS - len( features ) )


def _xy_features( index, width: int ) -> list[float]:
    if index is None or not width:
        return [0.0, 0.0]
    x, y = _xy( int( index ), width )
    return [x / float( width ), y / float( width )]


def context_token_features( event: dict, context: dict | None ) -> list[float]:
    """Kingdom state: strategy_model's six numbers, the day of the week (creatures grow on day 1),
    the week and every resource on its own."""
    ctx = context or {}
    res = list( ctx.get( "res" ) or event.get( "res" ) or [0] * 7 )
    weekday = int( ctx.get( "wd" ) or 0 )
    return ( context_features( event, context ) + [1.0 if weekday == d else 0.0 for d in range( 1, 8 )]
             + [float( ctx.get( "wk" ) or 0 ) / 4.0] + [math.log1p( max( float( r ), 0.0 ) ) for r in res] )


def day_token( context: dict ) -> list[float]:
    """A previous turn of the player (the kingdom part; its heroes and castles follow it)."""
    return _dated( context_token_features( context, context ), context.get( "t" ) ) + _kind_hot( None ) + _type_hot( "day" )


def fresh_context( event: dict, context: dict | None ) -> dict | None:
    """The turn context as it is at the moment of the query: a query's "now" (resources, castles, heroes;
    the engine sends it since 2026-10-10) replaces the start-of-turn snapshot of those keys — by a
    mid-turn query the heroes have moved, fought and bought troops. Older records have no "now"."""
    now = event.get( "now" )
    if not now or context is None:
        return context
    return {**context, **now}


def decision_token( decision: dict, obj_vocab: list[int] ) -> list[float]:
    """One previous answer of the player: the query kind and the chosen option's features."""
    kind, event = decision["kind"], decision["event"]
    options = query_options( kind, event )
    answer = decision["answer"]
    features = _option_features( kind, event, fresh_context( event, decision.get( "context" ) ), options[answer], answer, obj_vocab )
    features[DAY_SLOT] = float( event.get( "t", 0 ) ) / 30.0
    return features + _kind_hot( kind ) + _type_hot( "decision" )


def hero_tokens( event: dict, context: dict | None ) -> list[list[float]]:
    """The player's heroes as the hero dialog shows them: army strength and stacks, move and spell
    points, level, primary and secondary skills, artifacts, position; the query's hero marked."""
    context = context or {}
    width = int( context.get( "w" ) or 0 )
    tokens = []
    for hero in context.get( "heroes" ) or []:
        mmp = float( hero.get( "mmp" ) or 1000.0 )
        features = [math.log1p( float( hero.get( "str", 0.0 ) ) ), float( hero.get( "mp", mmp ) ) / mmp,
                    1.0 if hero.get( "id" ) == event.get( "h" ) else 0.0]
        features += [float( hero.get( key, 0 ) ) / scale for key, scale in
                     ( ( "lvl", 10.0 ), ( "a", 10.0 ), ( "d", 10.0 ), ( "pw", 10.0 ), ( "k", 10.0 ), ( "sp", 50.0 ), ( "msp", 50.0 ), ( "mor", 3.0 ),
                       ( "luck", 3.0 ), ( "book", 1.0 ) )]
        features += [len( hero.get( "art" ) or [] ) / 10.0] + _race_hot( hero.get( "race" ) )
        skills = list( hero.get( "sk" ) or [] )
        features += [float( level ) / 3.0 for level in skills[:14]] + [0.0] * ( 14 - len( skills[:14] ) )
        features += _stack_features( hero.get( "army" ) ) + _xy_features( hero.get( "i" ), width )
        tokens.append( _dated( features, context.get( "t" ) ) + _kind_hot( None ) + _type_hot( "hero", hero.get( "army" ) ) )
    return tokens


def _xy( index: int, width: int ) -> tuple[int, int]:
    return ( index % width, index // width ) if width else ( 0, 0 )


def _distance( a: int, b: int, width: int ) -> float:
    """Chebyshev distance in tiles (a hero moves diagonally too)."""
    ( ax, ay ), ( bx, by ) = _xy( a, width ), _xy( b, width )
    return float( max( abs( ax - bx ), abs( ay - by ) ) )


def _nearest( index, others: list, width: int ) -> float:
    return min( ( _distance( index, other, width ) for other in others if other is not None ), default=0.0 ) / 50.0


def castle_tokens( context: dict | None ) -> list[list[float]]:
    """The player's castles as the castle screen shows them: race, castle or town, built buildings,
    the garrison, creatures available per dwelling level; distance to the nearest visible enemy hero."""
    context = context or {}
    width = int( context.get( "w" ) or 0 )
    rivals = [r.get( "i" ) for r in context.get( "rivals" ) or []]
    tokens = []
    for castle in context.get( "castles" ) or []:
        buildings = int( castle.get( "b" ) or 0 )
        garrison = sum( float( s[1] ) * float( s[2] ) for s in castle.get( "army" ) or [] if len( s ) > 2 )
        features = _race_hot( castle.get( "race" ) ) + [float( castle.get( "castle", 0 ) ), math.log1p( garrison )]
        features += [1.0 if buildings & ( 1 << bit ) else 0.0 for bit in range( 32 )]
        dwellings = list( castle.get( "dw" ) or [] )[:DWELLINGS]
        for dwelling in dwellings:
            dwelling = list( dwelling ) + [0] * ( 7 - len( dwelling ) )
            features += [math.log1p( float( dwelling[1] ) ), math.log1p( float( dwelling[1] ) * float( dwelling[2] ) )]
        features += [0.0] * ( 2 * ( DWELLINGS - len( dwellings ) ) )
        features += [_nearest( castle.get( "i" ), rivals, width ) if rivals else 1.0] + _xy_features( castle.get( "i" ), width )
        tokens.append( _dated( features, context.get( "t" ) ) + _kind_hot( None ) + _type_hot( "castle", castle.get( "army" ) ) )
    return tokens


def _query_hero_tile( event: dict, context: dict ):
    return next( ( h.get( "i" ) for h in context.get( "heroes" ) or [] if h.get( "id" ) == event.get( "h" ) ), event.get( "from" ) )


def rival_tokens( event: dict, context: dict | None ) -> list[list[float]]:
    """Enemy heroes the player can see (turn_context "rivals": only outside the fog; the army as
    monster types with the size word unless full information via Identify Hero / Crystal Ball, see
    AIDecision writeVisibleRivals): estimated strength, full-info flag, stacks, distances to the
    query's hero, our nearest hero and castle; primary skills etc. only with full information."""
    context = context or {}
    width = int( context.get( "w" ) or 0 )
    ours = context.get( "heroes" ) or []
    castles = [c.get( "i" ) for c in context.get( "castles" ) or [] if c.get( "i" ) is not None]
    query_hero = _query_hero_tile( event, context )
    tokens = []
    for rival in context.get( "rivals" ) or []:
        index = rival.get( "i", 0 )
        to_query = _distance( index, query_hero, width ) if query_hero is not None else 0.0
        to_hero = min( ( _distance( index, h.get( "i", 0 ), width ) for h in ours ), default=0.0 )
        to_castle = min( ( _distance( index, c, width ) for c in castles ), default=0.0 )
        full = float( rival.get( "full", 0 ) )
        features = [math.log1p( float( rival.get( "est", 0 ) ) ), full, len( rival.get( "army" ) or [] ) / 7.0,
                    to_query / 50.0, to_hero / 50.0, to_castle / 50.0]
        features += [float( rival.get( key, 0 ) ) / scale for key, scale in
                     ( ( "lvl", 10.0 ), ( "a", 10.0 ), ( "d", 10.0 ), ( "pw", 10.0 ), ( "k", 10.0 ), ( "sp", 50.0 ), ( "mor", 3.0 ), ( "luck", 3.0 ) )]
        features += _stack_features( rival.get( "army" ) ) + _xy_features( index, width )
        tokens.append( _dated( features, context.get( "t" ) ) + _kind_hot( None ) + _type_hot( "rival", rival.get( "army" ) ) )
    return tokens


def rival_castle_tokens( event: dict, context: dict | None ) -> list[list[float]]:
    """Castles of other owners outside the fog (turn_context "rcastles", as the castle quick info
    shows them): neutral or enemy, race, castle or town, what is seen of the defenders (nothing /
    types / size words / exact, AIDecision writeVisibleCastles), estimated strength, distances to
    the query's hero, our nearest hero and castle."""
    context = context or {}
    width = int( context.get( "w" ) or 0 )
    ours = [h.get( "i" ) for h in context.get( "heroes" ) or []]
    castles = [c.get( "i" ) for c in context.get( "castles" ) or []]
    query_hero = _query_hero_tile( event, context )
    tokens = []
    for castle in context.get( "rcastles" ) or []:
        index = castle.get( "i", 0 )
        vis = int( castle.get( "vis", 0 ) )
        features = [1.0 if castle.get( "c" ) in ( None, "None", "" ) else 0.0] + _race_hot( castle.get( "race" ) )
        features += [float( castle.get( "castle", 0 ) )] + [1.0 if vis == v else 0.0 for v in range( 4 )]
        features += [math.log1p( float( castle.get( "est", 0 ) ) ), len( castle.get( "army" ) or [] ) / 5.0,
                     _distance( index, query_hero, width ) / 50.0 if query_hero is not None else 0.0,
                     _nearest( index, ours, width ), _nearest( index, castles, width )]
        features += _stack_features( castle.get( "army" ) ) + _xy_features( index, width )
        tokens.append( _dated( features, context.get( "t" ) ) + _kind_hot( None ) + _type_hot( "rcastle", castle.get( "army" ) ) )
    return tokens


def snapshot_tokens( event: dict, context: dict | None ) -> list[list[float]]:
    """What the player sees at the start of a day: heroes, castles, enemy heroes and castles."""
    return hero_tokens( event, context ) + castle_tokens( context ) + rival_tokens( event, context ) + rival_castle_tokens( event, context )


def history_tokens( history: dict | None, obj_vocab: list[int], budget: int = MAX_STRATEGIC_TOKENS ) -> list[list[float]]:
    """The player's game so far in time order: every previous day (its kingdom token and snapshot)
    followed by the answers given that day; then the answers already given today. The oldest days
    are dropped when the history exceeds `budget` tokens."""
    if not history:
        return []
    days = list( history.get( "days", [] ) )
    decisions = list( history.get( "decisions", [] ) )
    blocks: list[list[list[float]]] = []
    position = 0
    for day in days:
        block = [day_token( day )] + snapshot_tokens( {}, day )
        while position < len( decisions ) and decisions[position]["event"].get( "t", 0 ) <= day.get( "t", 0 ):
            block.append( decision_token( decisions[position], obj_vocab ) )
            position += 1
        blocks.append( block )
    tokens = [decision_token( d, obj_vocab ) for d in decisions[position:]]
    for block in reversed( blocks ):
        if len( tokens ) + len( block ) > budget:
            break
        tokens = block + tokens
    return tokens[-budget:] if budget > 0 else []


def query_tokens( kind: str, event: dict, context: dict | None, obj_vocab: list[int], history: dict | None = None,
                  window: int = MAX_STRATEGIC_TOKENS ):
    """(prefix tokens, context token, option tokens) of a query. The prefix is the player's history
    (history_tokens, as much as the window allows) and what it sees now (snapshot_tokens); the
    options are query_options(kind, event)."""
    context = fresh_context( event, context )
    context_token = _dated( context_token_features( event, context ), event.get( "t", ( context or {} ).get( "t" ) ) ) + _kind_hot( kind ) + _type_hot( "context" )
    option_tokens = [_option_features( kind, event, context, option, index, obj_vocab ) + _kind_hot( kind ) + _type_hot( "option" )
                     for index, option in enumerate( query_options( kind, event ) )]
    for token in option_tokens:
        token[DAY_SLOT] = float( event.get( "t", 0 ) ) / 30.0
    snapshot = snapshot_tokens( event, context )
    budget = window - len( snapshot ) - 1 - 2 * len( option_tokens )
    prefix = history_tokens( history, obj_vocab, budget ) + snapshot
    return prefix, context_token, option_tokens


def value_tokens( context: dict, obj_vocab: list[int], history: dict | None = None, window: int = MAX_STRATEGIC_TOKENS ):
    """(prefix tokens, context token) of a strategic state for the value head: the player's game so
    far (history_tokens) and what it sees now, as in query_tokens but without a query."""
    context_token = _dated( context_token_features( context, context ), context.get( "t" ) ) + _kind_hot( None ) + _type_hot( "context" )
    snapshot = snapshot_tokens( {}, context )
    return history_tokens( history, obj_vocab, window - len( snapshot ) - 1 ) + snapshot, context_token


def attach_history( records: list[dict] ) -> list[dict]:
    """Adds "history" (previous days and answered decisions of the same player in the same game)
    to records that carry an answer index in `answer_key`-order: records of one game are grouped by
    (map, seed, player) and ordered by the query number n. SFT records answer with "target"."""
    games: dict[tuple, list[dict]] = defaultdict( list )
    for record in records:
        games[( record.get( "map" ), record.get( "seed" ), record["event"].get( "p" ) )].append( record )
    for game in games.values():
        game.sort( key=lambda r: r["n"] )
        days: list[dict] = []
        decisions: list[dict] = []
        for record in game:
            context = record.get( "context" )
            if context and ( not days or days[-1].get( "t" ) != context.get( "t" ) ):
                days.append( context )
            record["history"] = {"days": list( days[:-1] ), "decisions": list( decisions )}
            decisions.append( {"kind": record["kind"], "event": record["event"], "context": context, "answer": record["target"]} )
    return records


def obj_vocab_of( records: list[dict] ) -> list[int]:
    """The most common hero-target object types (one-hot features; the rest is "other")."""
    counts = Counter( c.get( "obj" ) for r in records if r["kind"] == "target" for c in r["event"].get( "cands" ) or [] )
    return [obj for obj, _ in counts.most_common( MAX_OBJ_VOCAB )]


# --- data ------------------------------------------------------------------------------------


def sft_records( binary: str, map_name: str, days: int, seed: int ) -> list[dict]:
    """Every strategic query of a seeded all-built-in game with the built-in answer's index."""
    from strategy_rollout import Branch, builtin_option, play

    base = Branch()
    _, records = play( binary, map_name, days, seed, base )
    results = [r.get( "result" ) for r in records]
    out = []
    for query in base.queries:
        options = query_options( query["kind"], query["event"] )
        if len( options ) < 2:
            continue
        builtin = 0 if query["kind"] == "build" else builtin_option( query["kind"], query["event"], options, results[query["n"]] )
        if builtin is None:
            continue
        out.append( {"kind": query["kind"], "event": query["event"], "context": query["context"], "target": builtin,
                     "seed": seed, "map": map_name, "n": query["n"]} )
    return out


ROLLOUT_TOP = 4  # strategy_rollout.py --top: its option lists index the "builtin_index"


def preference_pairs( rollout_records: list[dict], margin: float = 100.0 ) -> list[dict]:
    """DPO pairs from strategy_rollout.py labels: per query the best vs the worst option (the
    built-in option is the baseline branch, label 0)."""
    by_query: dict[tuple, list[dict]] = defaultdict( list )
    for record in rollout_records:
        by_query[( record["map"], record["seed"], record["n"] )].append( record )

    pairs = []
    for ( map_name, seed, n ), records in by_query.items():
        first = records[0]
        kind, event = first["kind"], first["event"]
        options = query_options( kind, event )
        rollout_options = options_of( kind, event, ROLLOUT_TOP )

        scores: dict[int, float] = {}
        builtin = first["builtin_index"]
        if kind == "build":
            scores[0] = 0.0  # BUILTIN: the baseline branch
        elif 0 <= builtin < len( rollout_options ) and rollout_options[builtin] in options:
            scores[options.index( rollout_options[builtin] )] = 0.0
        for record in records:
            if record["option"] in options:
                scores[options.index( record["option"] )] = label_score( record["delta"] )
        if len( scores ) < 2:
            continue
        best = max( scores, key=scores.get )
        worst = min( scores, key=scores.get )
        if scores[best] - scores[worst] <= margin:
            continue
        pairs.append( {"kind": kind, "event": event, "context": first["context"], "chosen": best, "rejected": worst,
                       "scores": {str( i ): s for i, s in scores.items()}, "seed": seed, "map": map_name, "n": n} )
    return pairs


# --- policy ------------------------------------------------------------------------------------


class NetStrategyPolicy:
    """Answers every strategic query with the transformer's most probable option (masked softmax
    over the query's options), conditioned on the player's history in this game (previous days
    and answers); a plain strategic policy (strategy_policies interface). One instance serves one
    game at a time: `reset()` between games. Callers that override an answer (strategy_games
    branches) use `decide()` + `record()` instead of the plain interface."""

    def __init__( self, model_path: str, device: str = "cpu", cache: bool = True, min_margin: float = 0.0 ):
        import torch

        from transformer_model import StrategicPrefixCache, load_checkpoint

        torch.set_num_threads( 2 )
        self.model = load_checkpoint( model_path, device ).eval()
        self.obj_vocab = list( self.model.config.get( "obj_vocab", [] ) )
        # Confidence gate (2026-10-07): leave the built-in answer only when the pick is more probable than
        # it by more than this (the oracle test found real gains in only ~8% of the queries).
        self.min_margin = min_margin
        # Inference caches (2026-10-05, 97% of a DPO collection was the forward pass): the prefix
        # key/value chunks shared by the queries of a game, and whole answers by their exact input —
        # every replay of a seeded game repeats the base game's queries up to its branch point.
        # Both are keyed by content, so they stay valid across games (reset() keeps them).
        self.prefix_cache = StrategicPrefixCache() if cache else None
        self._answers: "OrderedDict[bytes, list[float]]" = OrderedDict()
        self.answer_hits = 0
        self.reset()

    def reset( self ) -> None:
        self._contexts: dict[str, dict] = {}
        self._history: dict[str, dict] = defaultdict( lambda: {"days": [], "decisions": []} )

    def observe_turn( self, turn_context: dict ) -> None:
        color = turn_context.get( "p" )
        previous = self._contexts.get( color )
        if previous is not None and previous.get( "t" ) != turn_context.get( "t" ):
            self._history[color]["days"].append( previous )  # a finished day joins the history
        self._contexts[color] = turn_context

    def history( self, color: str ) -> dict:
        return self._history[color]

    def probabilities( self, kind: str, event: dict ) -> list[float]:
        import torch

        from transformer_model import strategic_logits, strategic_logits_cached

        color = event.get( "p" )
        tokens = query_tokens( kind, event, self._contexts.get( color ), self.obj_vocab, self._history[color],
                               self.model.config.get( "window", MAX_STRATEGIC_TOKENS ) )
        if self.prefix_cache is None:
            with torch.no_grad():
                logits, mask = strategic_logits( self.model, [tokens] )
                return torch.softmax( logits[0][mask[0]], dim=0 ).tolist()
        prefix, context, options = tokens
        rows = ( *prefix, context, *options )
        key = hashlib.blake2b( f"{len( prefix )}/{len( options )}/{len( context )}:".encode()
                               + array( "f", [x for row in rows for x in row] ).tobytes(), digest_size=16 ).digest()
        probs = self._answers.get( key )
        if probs is not None:
            self.answer_hits += 1
            self._answers.move_to_end( key )
            return list( probs )
        with torch.no_grad():
            probs = torch.softmax( strategic_logits_cached( self.model, tokens, self.prefix_cache ), dim=0 ).tolist()
        self._answers[key] = probs
        while len( self._answers ) > MAX_CACHED_ANSWERS:
            self._answers.popitem( last=False )
        return list( probs )

    def decide( self, kind: str, event: dict ) -> int | None:
        """Index of the most probable option (None: fewer than two options)."""
        options = query_options( kind, event )
        if len( options ) < 2:
            return None
        probs = self.probabilities( kind, event )
        pick = max( range( len( options ) ), key=lambda i: probs[i] )
        if self.min_margin > 0:
            builtin = builtin_option( kind, options, event )
            if builtin is not None and probs[pick] - probs[builtin] <= self.min_margin:
                return builtin
        return pick

    def record( self, kind: str, event: dict, index: int | None ) -> None:
        """Adds an answered query (option index) to the player's history."""
        if index is not None:
            color = event.get( "p" )
            self._history[color]["decisions"].append( {"kind": kind, "event": event, "context": self._contexts.get( color ), "answer": index} )

    def _choose( self, kind: str, event: dict ):
        index = self.decide( kind, event )
        self.record( kind, event, index )
        return None if index is None else answer_of( query_options( kind, event )[index] )

    def __call__( self, decision: dict ):
        choice = self._choose( "target", decision )
        return None if choice is None or choice is ( decision.get( "cands" ) or [None] )[0] else choice

    def build( self, event: dict ):
        return self._choose( "build", event )

    def hire( self, event: dict ):
        return self._choose( "hire", event )

    def army( self, event: dict ):
        return self._choose( "army", event )


def main() -> None:
    parser = argparse.ArgumentParser( description="Strategic data for the unified transformer" )
    sub = parser.add_subparsers( dest="cmd", required=True )
    sft = sub.add_parser( "sft", help="built-in answers of every query of seeded games" )
    sft.add_argument( "--binary", default="./fheroes2" )
    sft.add_argument( "--map", default="2kings.mp2" )
    sft.add_argument( "--days", type=int, default=30 )
    sft.add_argument( "--seeds", default="1-20" )
    sft.add_argument( "--jobs", type=int, default=2 )
    sft.add_argument( "--out", required=True )
    prefs = sub.add_parser( "prefs", help="DPO pairs from strategy_rollout.py labels" )
    prefs.add_argument( "--data", nargs="+", required=True )
    prefs.add_argument( "--margin", type=float, default=100.0 )
    prefs.add_argument( "--out", required=True )
    args = parser.parse_args()

    if args.cmd == "sft":
        from concurrent.futures import ThreadPoolExecutor

        from harvest_battles import parse_seeds

        seeds = parse_seeds( args.seeds )
        total = 0
        with open( args.out, "w" ) as out, ThreadPoolExecutor( max_workers=args.jobs ) as pool:
            for seed, records in zip( seeds, pool.map( lambda s: sft_records( args.binary, args.map, args.days, s ), seeds ) ):
                for record in records:
                    out.write( json.dumps( record, separators=( ",", ":" ) ) + "\n" )
                total += len( records )
                print( f"seed {seed}: {len( records )} queries ({dict( Counter( r['kind'] for r in records ) )})", flush=True )
        print( f"done: {total} queries -> {args.out}" )
    else:
        records = []
        for path in args.data:
            with open( path ) as f:
                records += [json.loads( line ) for line in f if line.strip()]
        pairs = preference_pairs( records, args.margin )
        with open( args.out, "w" ) as out:
            for pair in pairs:
                out.write( json.dumps( pair, separators=( ",", ":" ) ) + "\n" )
        print( f"{len( pairs )} pairs ({dict( Counter( p['kind'] for p in pairs ) )}) -> {args.out}" )


if __name__ == "__main__":
    main()
