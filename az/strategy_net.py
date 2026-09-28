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
from transformer_model import STRAT_FEATURES, STRAT_KINDS  # noqa: E402

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


def _kind_hot( kind: str ) -> list[float]:
    return [1.0 if kind == k else 0.0 for k in STRAT_KINDS]


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


def query_tokens( kind: str, event: dict, context: dict | None, obj_vocab: list[int] ) -> tuple[list[float], list[list[float]]]:
    """(context token, option tokens) of a query; the options are query_options(kind, event)."""
    context_token = _pad( context_features( event, context ) ) + _kind_hot( kind ) + [1.0]
    option_tokens = []
    for index, option in enumerate( query_options( kind, event ) ):
        if option == BUILTIN:
            features = [0.0] * ( STRAT_FEATURES - 1 ) + [1.0]  # the last feature slot marks "built-in decides"
        else:
            features = _pad( option_features( kind, event, context, option, index, obj_vocab )
                             + ( build_priority_features( event, option ) if kind == "build" else [] ) )
        option_tokens.append( features + _kind_hot( kind ) + [0.0] )
    return context_token, option_tokens


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
    over the query's options); a plain strategic policy (strategy_policies interface)."""

    def __init__( self, model_path: str, device: str = "cpu" ):
        import torch

        from transformer_model import load_checkpoint

        torch.set_num_threads( 2 )
        self.model = load_checkpoint( model_path, device ).eval()
        self.obj_vocab = list( self.model.config.get( "obj_vocab", [] ) )
        self._contexts: dict[str, dict] = {}

    def observe_turn( self, turn_context: dict ) -> None:
        self._contexts[turn_context.get( "p" )] = turn_context

    def probabilities( self, kind: str, event: dict ) -> list[float]:
        import torch

        from transformer_model import strategic_logits

        tokens = query_tokens( kind, event, self._contexts.get( event.get( "p" ) ), self.obj_vocab )
        with torch.no_grad():
            logits, mask = strategic_logits( self.model, [tokens] )
            return torch.softmax( logits[0][mask[0]], dim=0 ).tolist()

    def _choose( self, kind: str, event: dict ):
        options = query_options( kind, event )
        if len( options ) < 2:
            return None
        probs = self.probabilities( kind, event )
        return answer_of( options[max( range( len( options ) ), key=lambda i: probs[i] )] )

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
