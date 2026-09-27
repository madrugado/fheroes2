"""Learned strategic target choice: an advantage model trained on counterfactual rollouts.

Training data comes from az/strategy_rollout.py: for a decision, every labeled candidate j has
the exact difference of the deciding player's kingdom stats at day t + H between "pick j" and the
built-in choice (j = 0, label 0). The model regresses that advantage from candidate + context
features; the policy picks the candidate with the highest predicted advantage over the built-in
choice if it exceeds a margin, and keeps the built-in choice otherwise.

Models: "ridge" (closed-form, numpy) and "mlp" (tiny torch MLP). Both are saved as JSON (the MLP
weights inline) so the policy has no training-time dependencies. Model selection uses
leave-seeds-out cross-validation measured by the realized gain of the resulting policy.

Usage:
    az/.venv/bin/python az/strategy_model.py --data az/data/strategy_rollouts_Battlefi_h7.jsonl --out az/models/strategy_ridge.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict

import numpy as np

CASTLE_WEIGHT = 2000.0  # one castle ~ this much army strength in the label
DEFAULT_MMP = 1000.0
MAX_OBJ_VOCAB = 16


def label_score( delta: dict, castle_weight: float = CASTLE_WEIGHT ) -> float:
    return float( delta["str"] ) + castle_weight * float( delta["k"] ) + 10000.0 * float( delta.get( "outcome", 0 ) )


def hero_info( decision: dict, context: dict | None ) -> dict:
    for hero in ( context or {} ).get( "heroes" ) or []:
        if hero.get( "id" ) == decision.get( "h" ):
            return hero
    return {}


def candidate_features( decision: dict, context: dict | None, j: int, obj_vocab: list[int] ) -> list[float]:
    """Features of candidate j of a decision (relative to the top-value candidate) + context."""
    cands = decision["cands"]
    cand, top = cands[j], cands[0]
    hero = hero_info( decision, context )
    mmp = float( hero.get( "mmp" ) or DEFAULT_MMP )
    mp = float( hero.get( "mp", mmp ) )
    ctx = context or {}
    res = ctx.get( "res" ) or [0] * 7

    features = [
        1.0 if j == 0 else 0.0,
        float( j ),
        math.log1p( max( cand["v"], 0.0 ) ),
        cand["v"] / top["v"] if top["v"] > 0 else 0.0,
        cand["d"] / mmp,
        ( cand["d"] - top["d"] ) / mmp,
        1.0 if cand["d"] <= mp else 0.0,
        decision.get( "t", 0 ) / 30.0,
        math.log1p( float( hero.get( "str", 0.0 ) ) ),
        mp / mmp,
        float( len( ctx.get( "heroes" ) or [] ) ),
        float( len( ctx.get( "castles" ) or [] ) ),
        math.log1p( float( res[6] ) ),
        len( cands ) / 10.0,
    ]
    obj = cand.get( "obj" )
    features += [1.0 if obj == o else 0.0 for o in obj_vocab]
    features.append( 0.0 if obj in obj_vocab else 1.0 )  # "other" object type
    return features


def build_obj_vocab( records: list[dict] ) -> list[int]:
    counts = Counter( r["cand"].get( "obj" ) for r in records )
    return [obj for obj, _ in counts.most_common( MAX_OBJ_VOCAB )]


def design( records: list[dict], obj_vocab: list[int], castle_weight: float = CASTLE_WEIGHT ):
    x = np.array( [candidate_features( r["decision"], r["context"], r["j"], obj_vocab ) for r in records], dtype=np.float64 )
    y = np.array( [label_score( r["delta"], castle_weight ) for r in records], dtype=np.float64 )
    return x, y


class Standardizer:
    def __init__( self, mean=None, std=None ):
        self.mean = mean
        self.std = std

    def fit( self, x ):
        self.mean = x.mean( axis=0 )
        self.std = x.std( axis=0 )
        self.std[self.std < 1e-9] = 1.0
        return self

    def __call__( self, x ):
        return ( x - self.mean ) / self.std


def fit_ridge( x, y, alpha: float = 10.0 ) -> dict:
    scaler = Standardizer().fit( x )
    xs = np.hstack( [scaler( x ), np.ones( ( len( x ), 1 ) )] )
    reg = alpha * np.eye( xs.shape[1] )
    reg[-1, -1] = 0.0  # do not shrink the intercept
    w = np.linalg.solve( xs.T @ xs + reg, xs.T @ y )
    return {"kind": "ridge", "mean": scaler.mean.tolist(), "std": scaler.std.tolist(), "w": w.tolist()}


def fit_mlp( x, y, hidden: int = 32, epochs: int = 300, seed: int = 0 ) -> dict:
    import torch

    torch.set_num_threads( 2 )  # do not saturate the machine (tiny model anyway)
    torch.manual_seed( seed )
    scaler = Standardizer().fit( x )
    y_scale = float( np.abs( y ).mean() ) or 1.0
    xt = torch.tensor( scaler( x ), dtype=torch.float32 )
    yt = torch.tensor( y / y_scale, dtype=torch.float32 ).unsqueeze( 1 )

    net = torch.nn.Sequential( torch.nn.Linear( x.shape[1], hidden ), torch.nn.ReLU(), torch.nn.Linear( hidden, hidden ), torch.nn.ReLU(),
                               torch.nn.Linear( hidden, 1 ) )
    opt = torch.optim.Adam( net.parameters(), lr=3e-3, weight_decay=1e-3 )
    for _ in range( epochs ):
        opt.zero_grad()
        # Huber: rollout labels are heavy-tailed (a lost/won castle dominates the batch).
        loss = torch.nn.functional.smooth_l1_loss( net( xt ), yt )
        loss.backward()
        opt.step()

    layers = [m for m in net if isinstance( m, torch.nn.Linear )]
    return {
        "kind": "mlp",
        "mean": scaler.mean.tolist(),
        "std": scaler.std.tolist(),
        "y_scale": y_scale,
        "layers": [{"w": layer.weight.detach().numpy().tolist(), "b": layer.bias.detach().numpy().tolist()} for layer in layers],
    }


def predict( model: dict, x ) -> np.ndarray:
    xs = ( np.asarray( x, dtype=np.float64 ) - np.array( model["mean"] ) ) / np.array( model["std"] )
    if model["kind"] == "ridge":
        return np.hstack( [xs, np.ones( ( len( xs ), 1 ) )] ) @ np.array( model["w"] )

    h = xs
    layers = model["layers"]
    for i, layer in enumerate( layers ):
        h = h @ np.array( layer["w"] ).T + np.array( layer["b"] )
        if i < len( layers ) - 1:
            h = np.maximum( h, 0.0 )
    return h[:, 0] * model["y_scale"]


def group_decisions( records: list[dict] ) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict( list )
    for r in records:
        groups[( r["seed"], r["n"] )].append( r )
    return groups


def policy_gain( model: dict, records: list[dict], obj_vocab: list[int], margin: float, castle_weight: float = CASTLE_WEIGHT ) -> dict:
    """Realized label of the candidate the policy would pick, per decision (built-in = 0)."""
    gains, oracle, switched = [], [], 0
    for group in group_decisions( records ).values():
        x, y = design( group, obj_vocab, castle_weight )
        pred = predict( model, x )
        base = next( ( p for r, p in zip( group, pred ) if r["j"] == 0 ), 0.0 )
        best = int( np.argmax( pred ) )
        pick = best if pred[best] - base > margin and group[best]["j"] != 0 else None
        gains.append( y[pick] if pick is not None else 0.0 )
        oracle.append( max( 0.0, float( y.max() ) ) )
        switched += pick is not None
    n = len( gains )
    return {
        "decisions": n,
        "mean_gain": float( np.mean( gains ) ) if n else 0.0,
        "oracle_gain": float( np.mean( oracle ) ) if n else 0.0,
        "switched": switched,
        "positive": int( sum( g > 0 for g in gains ) ),
        "negative": int( sum( g < 0 for g in gains ) ),
    }


def cross_validate( records: list[dict], kind: str, margins: list[float], folds: int = 4, seed: int = 0 ) -> dict:
    """Leave-seeds-out CV: decisions of one game never land in train and test at once."""
    seeds = sorted( {r["seed"] for r in records} )
    random.Random( seed ).shuffle( seeds )
    fold_of = {s: i % folds for i, s in enumerate( seeds )}

    results = {m: [] for m in margins}
    for fold in range( folds ):
        train = [r for r in records if fold_of[r["seed"]] != fold]
        test = [r for r in records if fold_of[r["seed"]] == fold]
        if not train or not test:
            continue
        vocab = build_obj_vocab( train )
        x, y = design( train, vocab )
        model = fit_ridge( x, y ) if kind == "ridge" else fit_mlp( x, y, seed=seed + fold )
        for m in margins:
            results[m].append( policy_gain( model, test, vocab, m ) )

    summary = {}
    for m, parts in results.items():
        total = sum( p["decisions"] for p in parts )
        summary[m] = {
            "decisions": total,
            "mean_gain": sum( p["mean_gain"] * p["decisions"] for p in parts ) / total if total else 0.0,
            "oracle_gain": sum( p["oracle_gain"] * p["decisions"] for p in parts ) / total if total else 0.0,
            "switched": sum( p["switched"] for p in parts ),
            "positive": sum( p["positive"] for p in parts ),
            "negative": sum( p["negative"] for p in parts ),
        }
    return summary


def load_records( paths: list[str] ) -> list[dict]:
    records = []
    for path in paths:
        with open( path ) as f:
            records += [json.loads( line ) for line in f if line.strip()]
    return records


def train_model( records: list[dict], kind: str, margin: float, top: int ) -> dict:
    vocab = build_obj_vocab( records )
    x, y = design( records, vocab )
    model = fit_ridge( x, y ) if kind == "ridge" else fit_mlp( x, y )
    model.update( obj_vocab=vocab, margin=margin, top=top, castle_weight=CASTLE_WEIGHT )
    return model


def main() -> None:
    parser = argparse.ArgumentParser( description="Train the strategic advantage model on rollout labels" )
    parser.add_argument( "--data", nargs="+", required=True )
    parser.add_argument( "--kind", choices=["ridge", "mlp", "both"], default="both" )
    parser.add_argument( "--margins", type=str, default="0,100,250,500,1000" )
    parser.add_argument( "--top", type=int, default=4 )
    parser.add_argument( "--out", type=str, default=None, help="save the CV-best model (kind + margin) here" )
    args = parser.parse_args()

    records = load_records( args.data )
    margins = [float( m ) for m in args.margins.split( "," )]
    labels = [label_score( r["delta"] ) for r in records if r["j"] != 0]
    print( f"{len( records )} records, {len( group_decisions( records ) )} decisions, {len( {r['seed'] for r in records} )} seeds; "
           f"alternatives: {sum( l > 0 for l in labels )} better / {sum( l == 0 for l in labels )} equal / {sum( l < 0 for l in labels )} worse" )

    best = None
    for kind in ( ["ridge", "mlp"] if args.kind == "both" else [args.kind] ):
        cv = cross_validate( records, kind, margins )
        for m, s in cv.items():
            print( f"{kind:5s} margin {m:6.0f}: gain {s['mean_gain']:8.1f} / oracle {s['oracle_gain']:8.1f}  "
                   f"switched {s['switched']:4d} (+{s['positive']} / -{s['negative']}) of {s['decisions']}" )
            if best is None or s["mean_gain"] > best[2]:
                best = ( kind, m, s["mean_gain"] )

    print( f"CV-best: {best[0]} margin {best[1]} (gain {best[2]:.1f} per decision)" )
    if args.out:
        model = train_model( records, best[0], best[1], args.top )
        model["cv_gain"] = best[2]
        os.makedirs( os.path.dirname( args.out ) or ".", exist_ok=True )
        with open( args.out, "w" ) as f:
            json.dump( model, f )
        print( f"saved -> {args.out}" )


if __name__ == "__main__":
    main()
