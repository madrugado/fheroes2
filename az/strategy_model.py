"""Learned strategic policy: per-kind advantage models trained on counterfactual rollouts.

Training data comes from az/strategy_rollout.py: for a strategic query (hero target, build,
hire, army budget) every labeled option has the exact difference of the deciding player's
kingdom stats at day t + H between "answer with this option" and the built-in answer (label 0).
One model per query kind regresses that advantage from option + context features. The policy
answers with the option of the highest predicted advantage if it exceeds the kind's margin and
keeps the built-in choice otherwise.

By default (`--rule all`) the model answers every kind; the leave-seeds-out CV statistics are
stored in the model file. `--rule ci` enables a kind only if its CV gain is convincingly positive
(95% bootstrap lower bound > 0) — the setting for quality work: rollout labels are dominated by
chaos (a different answer reshuffles the RNG stream), and a positive mean alone proved misleading.

Architectures: "ridge" (closed-form, numpy) and "mlp" (tiny torch MLP, 2 threads). Models are
saved as JSON so the policy has no training-time dependencies.

Usage:
    az/.venv/bin/python az/strategy_model.py --data az/data/strategy_rollouts_all_Battlefi_h7.jsonl \\
        az/data/strategy_rollouts_Battlefi_h7.jsonl --out az/models/strategy_model.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter, defaultdict

import numpy as np

from strategy_policies import ARMY_BUDGETS, NOTHING

CASTLE_WEIGHT = 2000.0  # one castle ~ this much army strength in the label
DEFAULT_MMP = 1000.0
MAX_OBJ_VOCAB = 16
KINDS = ( "target", "build", "hire", "army" )
RACES = ( 1, 2, 4, 8, 16, 32 )  # Race::KNGT, BARB, SORC, WRLK, WZRD, NECR
ARMY_REASONS = ( "defense", "visit", "hire" )


def label_score( delta: dict, castle_weight: float = CASTLE_WEIGHT ) -> float:
    return float( delta["str"] ) + castle_weight * float( delta["k"] ) + 10000.0 * float( delta.get( "outcome", 0 ) )


def options_of( kind: str, event: dict, top: int = 4 ) -> list:
    """The answers considered for a query: candidate dicts, NOTHING, or army budget percents."""
    cands = list( event.get( "cands" ) or [] )
    if kind == "target":
        return cands[:top]
    if kind in ( "build", "hire" ):
        return cands + [NOTHING]
    if kind == "army":
        return list( ARMY_BUDGETS )
    raise ValueError( kind )


# --- features -------------------------------------------------------------------------------


def context_features( event: dict, context: dict | None ) -> list[float]:
    """Kingdom-level context shared by all kinds (from the last turn_context of the player)."""
    ctx = context or {}
    res = ctx.get( "res" ) or event.get( "res" ) or [0] * 7
    heroes = ctx.get( "heroes" ) or []
    return [
        event.get( "t", 0 ) / 30.0,
        math.log1p( max( float( res[6] ), 0.0 ) ),
        math.log1p( max( float( sum( res[:6] ) ), 0.0 ) ),
        float( len( ctx.get( "castles" ) or [] ) ),
        float( len( heroes ) ),
        math.log1p( sum( float( h.get( "str", 0.0 ) ) for h in heroes ) ),
    ]


def hero_info( decision: dict, context: dict | None ) -> dict:
    for hero in ( context or {} ).get( "heroes" ) or []:
        if hero.get( "id" ) == decision.get( "h" ):
            return hero
    return {}


def target_features( decision: dict, context: dict | None, j: int, obj_vocab: list[int] ) -> list[float]:
    """Candidate j of a hero-target decision (relative to the top-value candidate) + context."""
    cands = decision["cands"]
    cand, top = cands[j], cands[0]
    hero = hero_info( decision, context )
    mmp = float( hero.get( "mmp" ) or DEFAULT_MMP )
    mp = float( hero.get( "mp", mmp ) )

    features = [
        1.0 if j == 0 else 0.0,
        float( j ),
        math.log1p( max( cand["v"], 0.0 ) ),
        cand["v"] / top["v"] if top["v"] > 0 else 0.0,
        cand["d"] / mmp,
        ( cand["d"] - top["d"] ) / mmp,
        1.0 if cand["d"] <= mp else 0.0,
        math.log1p( float( hero.get( "str", 0.0 ) ) ),
        mp / mmp,
        len( cands ) / 10.0,
    ]
    obj = cand.get( "obj" )
    features += [1.0 if obj == o else 0.0 for o in obj_vocab]
    features.append( 0.0 if obj in obj_vocab else 1.0 )  # "other" object type
    return features + context_features( decision, context )


def build_features( event: dict, context: dict | None, option ) -> list[float]:
    res = event.get( "res" ) or [0] * 7
    gold = float( res[6] )
    nothing = option == NOTHING
    bits = [0.0] * 32
    cost = [0] * 7
    trade = 0.0
    if not nothing:
        bits[int( option["b"] ).bit_length() - 1] = 1.0
        cost = option.get( "cost" ) or cost
        trade = float( option.get( "trade", 0 ) )
    race = event.get( "race" )
    return (
        [
            1.0 if nothing else 0.0,
            trade,
            math.log1p( float( cost[6] ) ),
            math.log1p( float( sum( cost[:6] ) ) ),
            float( cost[6] ) / ( gold + 1.0 ),
            float( event.get( "defensive", 0 ) ),
            len( event.get( "cands" ) or [] ) / 10.0,
        ]
        + bits
        + [1.0 if race == r else 0.0 for r in RACES]
        + context_features( event, context )
    )


def hire_features( event: dict, context: dict | None, option, index: int ) -> list[float]:
    nothing = option == NOTHING
    cand = {} if nothing else option
    return [
        1.0 if nothing else 0.0,
        1.0 if index == event.get( "bi", -1 ) or ( nothing and event.get( "bi", -1 ) < 0 ) else 0.0,
        float( cand.get( "val", 0.0 ) ) / 1000.0,
        float( cand.get( "lvl", 0 ) ),
        math.log1p( float( cand.get( "army", 0.0 ) ) ),
        1.0 if cand.get( "slot" ) == 2 else 0.0,
        float( event.get( "heroes", 0 ) ),
        len( event.get( "cands" ) or [] ) / 10.0,
    ] + context_features( event, context )


def army_features( event: dict, context: dict | None, option ) -> list[float]:
    offer = event.get( "offer" ) or []
    reason = event.get( "reason" )
    return (
        [
            float( option ) / 100.0,
            math.log1p( sum( float( o.get( "str", 0.0 ) ) for o in offer ) ),
            math.log1p( float( sum( o.get( "n", 0 ) for o in offer ) ) ),
            math.log1p( float( event.get( "garrison", 0.0 ) ) ),
            math.log1p( float( event.get( "hero", 0.0 ) ) ),
            1.0 if event.get( "guest", -1 ) >= 0 else 0.0,
        ]
        # One-hot too: the value of a budget need not be monotonic in the percent.
        + [1.0 if option == b else 0.0 for b in ARMY_BUDGETS]
        + [1.0 if reason == r else 0.0 for r in ARMY_REASONS]
        + context_features( event, context )
    )


def option_features( kind: str, event: dict, context: dict | None, option, index: int, obj_vocab: list[int] ) -> list[float]:
    if kind == "target":
        return target_features( event, context, index, obj_vocab )
    if kind == "build":
        return build_features( event, context, option )
    if kind == "hire":
        return hire_features( event, context, option, index )
    if kind == "army":
        return army_features( event, context, option )
    raise ValueError( kind )


def build_obj_vocab( records: list[dict] ) -> list[int]:
    counts = Counter( r["option"].get( "obj" ) for r in records if r["kind"] == "target" )
    return [obj for obj, _ in counts.most_common( MAX_OBJ_VOCAB )]


def design( records: list[dict], obj_vocab: list[int], castle_weight: float = CASTLE_WEIGHT ):
    x = np.array( [option_features( r["kind"], r["event"], r["context"], r["option"], r["option_index"], obj_vocab ) for r in records], dtype=np.float64 )
    y = np.array( [label_score( r["delta"], castle_weight ) for r in records], dtype=np.float64 )
    return x, y


# --- models ---------------------------------------------------------------------------------


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
    return {"arch": "ridge", "mean": scaler.mean.tolist(), "std": scaler.std.tolist(), "w": w.tolist()}


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
        "arch": "mlp",
        "mean": scaler.mean.tolist(),
        "std": scaler.std.tolist(),
        "y_scale": y_scale,
        "layers": [{"w": layer.weight.detach().numpy().tolist(), "b": layer.bias.detach().numpy().tolist()} for layer in layers],
    }


def fit( arch: str, x, y, seed: int = 0 ) -> dict:
    return fit_ridge( x, y ) if arch == "ridge" else fit_mlp( x, y, seed=seed )


def predict( model: dict, x ) -> np.ndarray:
    xs = ( np.asarray( x, dtype=np.float64 ) - np.array( model["mean"] ) ) / np.array( model["std"] )
    if model.get( "arch", model.get( "kind" ) ) == "ridge":
        return np.hstack( [xs, np.ones( ( len( xs ), 1 ) )] ) @ np.array( model["w"] )

    h = xs
    layers = model["layers"]
    for i, layer in enumerate( layers ):
        h = h @ np.array( layer["w"] ).T + np.array( layer["b"] )
        if i < len( layers ) - 1:
            h = np.maximum( h, 0.0 )
    return h[:, 0] * model["y_scale"]


# --- evaluation -----------------------------------------------------------------------------


def group_queries( records: list[dict] ) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict( list )
    for r in records:
        groups[( r["seed"], r["n"] )].append( r )
    return groups


def query_gains( model: dict, records: list[dict], obj_vocab: list[int], margin: float ) -> list[float]:
    """Realized label of the option the policy would answer with, per query (built-in = 0)."""
    gains = []
    for group in group_queries( records ).values():
        x, y = design( group, obj_vocab )
        pred = predict( model, x )
        best = int( np.argmax( pred ) )
        take = pred[best] > margin and group[best]["option_index"] != group[best]["builtin_index"]
        gains.append( float( y[best] ) if take else 0.0 )
    return gains


def bootstrap_ci( values: list[float], iterations: int = 2000, seed: int = 0 ) -> tuple[float, float]:
    if not values:
        return ( 0.0, 0.0 )
    rng = np.random.default_rng( seed )
    data = np.asarray( values )
    means = rng.choice( data, ( iterations, len( data ) ) ).mean( axis=1 )
    return ( float( np.percentile( means, 2.5 ) ), float( np.percentile( means, 97.5 ) ) )


def cross_validate( records: list[dict], arch: str, margins: list[float], folds: int = 4, seed: int = 0 ) -> dict[float, dict]:
    """Leave-seeds-out CV of one kind's records: queries of one game never land in train and test
    at once. Returns per margin: mean gain, 95% CI, switched/positive/negative counts."""
    seeds = sorted( {r["seed"] for r in records} )
    random.Random( seed ).shuffle( seeds )
    fold_of = {s: i % folds for i, s in enumerate( seeds )}

    gains: dict[float, list[float]] = {m: [] for m in margins}
    for fold in range( folds ):
        train = [r for r in records if fold_of[r["seed"]] != fold]
        test = [r for r in records if fold_of[r["seed"]] == fold]
        if not train or not test:
            continue
        vocab = build_obj_vocab( train )
        x, y = design( train, vocab )
        model = fit( arch, x, y, seed=seed + fold )
        for m in margins:
            gains[m] += query_gains( model, test, vocab, m )

    summary = {}
    for m, values in gains.items():
        summary[m] = {
            "queries": len( values ),
            "mean_gain": float( np.mean( values ) ) if values else 0.0,
            "ci95": bootstrap_ci( values ),
            "switched": int( sum( v != 0 for v in values ) ),
            "positive": int( sum( v > 0 for v in values ) ),
            "negative": int( sum( v < 0 for v in values ) ),
        }
    return summary


def load_records( paths: list[str] ) -> list[dict]:
    from strategy_rollout import convert_legacy  # the first rollout version was target-only

    records = []
    for path in paths:
        with open( path ) as f:
            records += [convert_legacy( json.loads( line ) ) for line in f if line.strip()]
    return records


def select_and_train( records: list[dict], margins: list[float], archs: list[str], rule: str, log=print ) -> dict:
    """Per kind: CV every (arch, margin), pick the best by mean gain, enable the kind only if the
    rule holds ("all": always, "ci": 95% lower bound > 0, "mean": mean > 0), train on all data."""
    kinds_model: dict[str, dict] = {}
    for kind in KINDS:
        subset = [r for r in records if r["kind"] == kind]
        if not subset:
            continue
        alternatives = [label_score( r["delta"] ) for r in subset if r["option_index"] != r["builtin_index"]]
        log( f"[{kind}] {len( group_queries( subset ) )} queries, {len( {r['seed'] for r in subset} )} seeds; alternatives: "
             f"{sum( a > 0 for a in alternatives )} better / {sum( a == 0 for a in alternatives )} equal / {sum( a < 0 for a in alternatives )} worse" )

        best = None
        for arch in archs:
            for m, s in cross_validate( subset, arch, margins ).items():
                log( f"  {arch:5s} margin {m:6.0f}: gain {s['mean_gain']:8.1f}  CI [{s['ci95'][0]:8.1f}, {s['ci95'][1]:8.1f}]  "
                     f"switched {s['switched']:4d} (+{s['positive']} / -{s['negative']}) of {s['queries']}" )
                if best is None or s["mean_gain"] > best[2]["mean_gain"]:
                    best = ( arch, m, s )

        arch, margin, stats = best
        if rule == "all":
            enabled = True
        elif rule == "ci":
            enabled = stats["ci95"][0] > 0
        else:
            enabled = stats["mean_gain"] > 0
        vocab = build_obj_vocab( subset )
        x, y = design( subset, vocab )
        model = fit( arch, x, y )
        model.update( obj_vocab=vocab, margin=margin, enabled=enabled, cv=stats )
        kinds_model[kind] = model
        log( f"  -> {arch} margin {margin}: {'ENABLED' if enabled else 'disabled (built-in choice kept)'}" )

    return {"version": 2, "top": 4, "castle_weight": CASTLE_WEIGHT, "rule": rule, "kinds": kinds_model}


def main() -> None:
    parser = argparse.ArgumentParser( description="Train the per-kind strategic advantage models on rollout labels" )
    parser.add_argument( "--data", nargs="+", required=True )
    parser.add_argument( "--arch", choices=["ridge", "mlp", "both"], default="both" )
    parser.add_argument( "--margins", type=str, default="0,100,250,500,1000" )
    parser.add_argument( "--rule", choices=["all", "ci", "mean"], default="all",
                         help="when a kind is enabled: all (the model answers every kind), ci (95%% CV lower bound > 0), mean (CV mean > 0)" )
    parser.add_argument( "--out", type=str, default=None )
    args = parser.parse_args()

    records = load_records( args.data )
    archs = ["ridge", "mlp"] if args.arch == "both" else [args.arch]
    model = select_and_train( records, [float( m ) for m in args.margins.split( "," )], archs, args.rule )

    if args.out:
        os.makedirs( os.path.dirname( args.out ) or ".", exist_ok=True )
        with open( args.out, "w" ) as f:
            json.dump( model, f )
        print( f"saved -> {args.out} (enabled: {[k for k, m in model['kinds'].items() if m['enabled']]})" )


if __name__ == "__main__":
    main()
