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
    az/.venv/bin/python az/strategy_net.py sft --map 2kings.mp2 --days 30 --seeds 1-40 --out az/data/strategy_sft_2kings.jsonl
    az/.venv/bin/python az/strategy_net.py prefs --data az/data/strategy_rollouts_all_Battlefi_h7.jsonl --out az/data/strategy_prefs.jsonl
    (training: az/train_strategy_net.py)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

from strategy_model import MAX_OBJ_VOCAB, context_features, label_score, option_features, options_of  # noqa: E402
from strategy_policies import NOTHING  # noqa: E402
from transformer_model import STRAT_FEATURES, STRAT_KINDS, STRAT_TOKEN_TYPES  # noqa: E402

# History kept in the input (the window is 2048 tokens; a 30-day game has ~150 queries per player).
MAX_HISTORY_DAYS = 60
MAX_HISTORY_DECISIONS = 400

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


def _type_hot( token_type: str ) -> list[float]:
    return [1.0 if token_type == t else 0.0 for t in STRAT_TOKEN_TYPES]


# "Let the built-in AI decide" — the first option of every build query (answered with `skip`).
# The built-in building choice depends on inputs the query does not carry (marketplace deals,
# safety factor, regions), so imitation of explicit buildings topped out at 0.81; deferring is
# exact, and DPO learns when an explicit building (or nothing) beats the built-in development.
BUILTIN = "builtin"


def query_options( kind: str, event: dict ) -> list:
    options = options_of( kind, event, TARGET_TOP )
    return [BUILTIN] + options if kind == "build" else options


def answer_of( option ):
    """The strategic policy answer of an option (BUILTIN -> None: the built-in AI decides)."""
    return None if option == BUILTIN else option


def _option_features( kind: str, event: dict, context: dict | None, option, index: int, obj_vocab: list[int] ) -> list[float]:
    if option == BUILTIN:
        return [0.0] * ( STRAT_FEATURES - 1 ) + [1.0]  # the last feature slot marks "built-in decides"
    return _pad( option_features( kind, event, context, option, index, obj_vocab )
                 + ( build_priority_features( event, option ) if kind == "build" else [] ) )


def day_token( context: dict ) -> list[float]:
    """One previous turn of the player: day, gold, other resources, castles, heroes, army strength."""
    features = context_features( context, context ) + [float( context.get( "t", 0 ) % 7 ) / 7.0]
    return _pad( features ) + _kind_hot( None ) + _type_hot( "day" )


def decision_token( decision: dict, obj_vocab: list[int] ) -> list[float]:
    """One previous answer of the player: the query kind and the chosen option's features."""
    kind, event = decision["kind"], decision["event"]
    options = query_options( kind, event )
    answer = decision["answer"]
    features = _option_features( kind, event, decision.get( "context" ), options[answer], answer, obj_vocab )
    # Option features use at most 58 slots and the last one flags BUILTIN: the day goes next to it.
    features[STRAT_FEATURES - 2] = float( event.get( "t", 0 ) ) / 30.0
    return features + _kind_hot( kind ) + _type_hot( "decision" )


def hero_tokens( event: dict, context: dict | None ) -> list[list[float]]:
    """The player's heroes at the start of the day (army strength, move points), the query's hero marked."""
    tokens = []
    for hero in ( context or {} ).get( "heroes" ) or []:
        mmp = float( hero.get( "mmp" ) or 1000.0 )
        features = [math.log1p( float( hero.get( "str", 0.0 ) ) ), float( hero.get( "mp", mmp ) ) / mmp,
                    1.0 if hero.get( "id" ) == event.get( "h" ) else 0.0]
        tokens.append( _pad( features ) + _kind_hot( None ) + _type_hot( "hero" ) )
    return tokens


def _xy( index: int, width: int ) -> tuple[int, int]:
    return ( index % width, index // width ) if width else ( 0, 0 )


def _distance( a: int, b: int, width: int ) -> float:
    """Chebyshev distance in tiles (a hero moves diagonally too)."""
    ( ax, ay ), ( bx, by ) = _xy( a, width ), _xy( b, width )
    return float( max( abs( ax - bx ), abs( ay - by ) ) )


def rival_tokens( event: dict, context: dict | None ) -> list[list[float]]:
    """Enemy heroes the player can see (turn_context "rivals": only outside the fog; the army as
    monster types with the size word unless full information via Identify Hero / Crystal Ball, see
    AIDecision writeVisibleRivals): estimated strength, full-info flag, stacks, distances to the
    query's hero, our nearest hero and castle; primary skills etc. only with full information."""
    context = context or {}
    width = int( context.get( "w" ) or 0 )
    ours = context.get( "heroes" ) or []
    castles = [c.get( "i" ) for c in context.get( "castles" ) or [] if c.get( "i" ) is not None]
    query_hero = next( ( h.get( "i" ) for h in ours if h.get( "id" ) == event.get( "h" ) ), event.get( "from" ) )
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
        tokens.append( _pad( features ) + _kind_hot( None ) + _type_hot( "rival" ) )
    return tokens


def history_tokens( history: dict | None, obj_vocab: list[int] ) -> list[list[float]]:
    """Previous days, then previous decisions of the player (oldest first, capped)."""
    if not history:
        return []
    days = [day_token( ctx ) for ctx in history.get( "days", [] )[-MAX_HISTORY_DAYS:]]
    decisions = [decision_token( d, obj_vocab ) for d in history.get( "decisions", [] )[-MAX_HISTORY_DECISIONS:]]
    return days + decisions


def query_tokens( kind: str, event: dict, context: dict | None, obj_vocab: list[int], history: dict | None = None ):
    """(prefix tokens, context token, option tokens) of a query; the prefix is the player's history
    (days and decisions), its heroes and the enemy heroes it can see; the options are
    query_options(kind, event)."""
    prefix = history_tokens( history, obj_vocab ) + hero_tokens( event, context ) + rival_tokens( event, context )
    context_token = _pad( context_features( event, context ) ) + _kind_hot( kind ) + _type_hot( "context" )
    option_tokens = [_option_features( kind, event, context, option, index, obj_vocab ) + _kind_hot( kind ) + _type_hot( "option" )
                     for index, option in enumerate( query_options( kind, event ) )]
    return prefix, context_token, option_tokens


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

    def __init__( self, model_path: str, device: str = "cpu" ):
        import torch

        from transformer_model import load_checkpoint

        torch.set_num_threads( 2 )
        self.model = load_checkpoint( model_path, device ).eval()
        self.obj_vocab = list( self.model.config.get( "obj_vocab", [] ) )
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

        from transformer_model import strategic_logits

        color = event.get( "p" )
        tokens = query_tokens( kind, event, self._contexts.get( color ), self.obj_vocab, self._history[color] )
        with torch.no_grad():
            logits, mask = strategic_logits( self.model, [tokens] )
            return torch.softmax( logits[0][mask[0]], dim=0 ).tolist()

    def decide( self, kind: str, event: dict ) -> int | None:
        """Index of the most probable option (None: fewer than two options)."""
        options = query_options( kind, event )
        if len( options ) < 2:
            return None
        probs = self.probabilities( kind, event )
        return max( range( len( options ) ), key=lambda i: probs[i] )

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
