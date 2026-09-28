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
import os
import sys
from collections import Counter, defaultdict

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

from strategy_model import MAX_OBJ_VOCAB, context_features, label_score, option_features, options_of  # noqa: E402
from strategy_policies import NOTHING  # noqa: E402
from transformer_model import STRAT_FEATURES, STRAT_KINDS  # noqa: E402

TARGET_TOP = 8  # hero-target candidates offered to the network (sorted by value; 0 = built-in)


def _pad( features: list[float] ) -> list[float]:
    if len( features ) > STRAT_FEATURES:
        raise ValueError( f"{len( features )} strategic features > STRAT_FEATURES ({STRAT_FEATURES})" )
    return features + [0.0] * ( STRAT_FEATURES - len( features ) )


def _kind_hot( kind: str ) -> list[float]:
    return [1.0 if kind == k else 0.0 for k in STRAT_KINDS]


def query_options( kind: str, event: dict ) -> list:
    return options_of( kind, event, TARGET_TOP )


def query_tokens( kind: str, event: dict, context: dict | None, obj_vocab: list[int] ) -> tuple[list[float], list[list[float]]]:
    """(context token, option tokens) of a query; the options are query_options(kind, event)."""
    context_token = _pad( context_features( event, context ) ) + _kind_hot( kind ) + [1.0]
    option_tokens = [_pad( option_features( kind, event, context, option, index, obj_vocab ) ) + _kind_hot( kind ) + [0.0]
                     for index, option in enumerate( query_options( kind, event ) )]
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
        builtin = builtin_option( query["kind"], query["event"], options, results[query["n"]] )
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
        if 0 <= builtin < len( rollout_options ) and rollout_options[builtin] in options:
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
        return options[max( range( len( options ) ), key=lambda i: probs[i] )]

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
