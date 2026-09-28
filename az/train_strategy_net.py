"""Trains the strategic output of the unified transformer (az/strategy_net.py).

    sft — imitation of the built-in AI's answers (cross-entropy over the masked option softmax);
    dpo — Direct Preference Optimization on counterfactual preference pairs against a frozen
          reference copy of the starting network:
          -log sigmoid(beta * [(log pi(chosen) - log ref(chosen)) - (log pi(rejected) - log ref(rejected))]).

The body is shared with the battle policy, so every step also takes a battle imitation batch
(--battle-data, masked losses as in train.py) weighted by --battle-weight: training the strategic
head must not erase the battle skills. Validation (held-out seeds): strategic imitation accuracy
(sft) or preference accuracy (dpo), plus the battle imitation accuracy on held-out battles.

Usage:
    az/.venv/bin/python az/train_strategy_net.py sft --model az/models/az_battle_tr50m_expert_v4.pt \\
        --data az/data/strategy_sft_2kings.jsonl --battle-data az/data/expert.jsonl.gz --out az/models/unified_sft.pt
    az/.venv/bin/python az/train_strategy_net.py dpo --model az/models/unified_sft.pt \\
        --data az/data/strategy_prefs_Battlefi.jsonl --battle-data az/data/expert.jsonl.gz --out az/models/unified_dpo.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

import train  # noqa: E402
from strategy_net import obj_vocab_of, query_tokens  # noqa: E402
from transformer_model import load_checkpoint, save_checkpoint, strategic_logits  # noqa: E402


def load_jsonl( paths: list[str] ) -> list[dict]:
    records = []
    for path in paths:
        with open( path ) as f:
            records += [json.loads( line ) for line in f if line.strip()]
    return records


def split_by_seed( records: list[dict], val_fraction: float ) -> tuple[list[dict], list[dict]]:
    """Held-out whole games (map, seed): queries of one game are correlated."""
    games = sorted( {( r.get( "map" ), r.get( "seed" ) ) for r in records} )
    rng = random.Random( 0 )
    rng.shuffle( games )
    held = set( games[:max( 1, int( len( games ) * val_fraction ) )] ) if val_fraction > 0 and len( games ) > 1 else set()
    return [r for r in records if ( r.get( "map" ), r.get( "seed" ) ) not in held], [r for r in records if ( r.get( "map" ), r.get( "seed" ) ) in held]


def option_log_probs( model, records: list[dict], obj_vocab: list[int] ) -> torch.Tensor:
    """(B, max options) log-softmax over each query's options (padding: -inf-like)."""
    queries = [query_tokens( r["kind"], r["event"], r.get( "context" ), obj_vocab ) for r in records]
    logits, _ = strategic_logits( model, queries )
    return F.log_softmax( logits, dim=1 )


def picked( log_probs: torch.Tensor, indexes: list[int] ) -> torch.Tensor:
    return log_probs[torch.arange( len( indexes ), device=log_probs.device ), torch.tensor( indexes, device=log_probs.device )]


def battle_loss( model, samples: list[tuple], device ) -> torch.Tensor:
    """Masked battle imitation loss (train.py's transformer loss) on a batch of samples."""
    states = [s for s, _, _ in samples]
    cell_targets = torch.tensor( [train.SKIP_CELL if t["kind"] == "skip" else t["cell"] for _, t, _ in samples], device=device )
    decode_cells = [t["cell"] if ( t["kind"] == "attack" and t["cell"] is not None ) else None for _, t, _ in samples]
    dir_targets = torch.tensor( [t["dir"] if t["dir"] is not None else 0 for _, t, _ in samples], device=device )
    values = torch.tensor( [v for _, _, v in samples], dtype=torch.float32, device=device )
    cell_logits, dir_out, value = model.forward_batch( states, decode_cells )
    cell_mask = train.legal_mask( [t.get( "cells" ) for _, t, _ in samples], cell_logits.shape[1], device )
    loss = F.cross_entropy( cell_logits.masked_fill( ~cell_mask, -1e9 ), cell_targets ) + F.mse_loss( value, values )
    if dir_out is not None:
        rows, dir_logits = dir_out
        dir_mask = train.legal_mask( [samples[row][1].get( "dirs" ) for row in rows], dir_logits.shape[1], device )
        loss = loss + F.cross_entropy( dir_logits.masked_fill( ~dir_mask, -1e9 ), dir_targets[rows] )
    return loss


def evaluate( model, mode: str, records: list[dict], obj_vocab: list[int], ref, batch: int ) -> float:
    """sft: share of queries where the argmax option is the built-in answer; dpo: share of pairs
    where log pi(chosen) > log pi(rejected)."""
    model.eval()
    hits = 0
    with torch.no_grad():
        for start in range( 0, len( records ), batch ):
            chunk = records[start:start + batch]
            log_probs = option_log_probs( model, chunk, obj_vocab )
            if mode == "sft":
                hits += int( ( log_probs.argmax( dim=1 ).cpu() == torch.tensor( [r["target"] for r in chunk] ) ).sum() )
            else:
                hits += int( ( picked( log_probs, [r["chosen"] for r in chunk] ) > picked( log_probs, [r["rejected"] for r in chunk] ) ).sum() )
    return hits / max( len( records ), 1 )


def main() -> None:
    parser = argparse.ArgumentParser( description="Strategic SFT / DPO of the unified transformer" )
    parser.add_argument( "mode", choices=["sft", "dpo"] )
    parser.add_argument( "--model", required=True, help="starting unified/battle transformer checkpoint" )
    parser.add_argument( "--data", nargs="+", required=True )
    parser.add_argument( "--battle-data", nargs="*", default=[], help="battle imitation records (anchor)" )
    parser.add_argument( "--battle-weight", type=float, default=1.0 )
    parser.add_argument( "--battle-max", type=int, default=20000, help="battle anchor positions loaded" )
    parser.add_argument( "--beta", type=float, default=0.1 )
    parser.add_argument( "--epochs", type=int, default=3 )
    parser.add_argument( "--batch", type=int, default=32 )
    parser.add_argument( "--lr", type=float, default=5e-5 )
    parser.add_argument( "--val", type=float, default=0.15 )
    parser.add_argument( "--threads", type=int, default=2 )
    parser.add_argument( "--out", required=True )
    args = parser.parse_args()

    torch.set_num_threads( args.threads )
    sys.stdout.reconfigure( line_buffering=True )
    device = torch.device( "mps" if torch.backends.mps.is_available() else "cpu" )

    model = load_checkpoint( args.model, str( device ) )
    records = load_jsonl( args.data )
    if "obj_vocab" not in model.config:
        model.config["obj_vocab"] = obj_vocab_of( records )
    obj_vocab = model.config["obj_vocab"]
    train_records, val_records = split_by_seed( records, args.val )
    print( f"{args.mode}: {len( train_records )} train, {len( val_records )} validation records" )

    anchor, anchor_val = [], []
    if args.battle_data:
        battle = []
        for path in args.battle_data:
            battle += train.load_records( path )[:args.battle_max]
        battle_train, battle_val = train.split_records( battle, 0.1 )
        anchor, _ = train.build_transformer_samples( battle_train )
        anchor_val = battle_val[:500]
        print( f"battle anchor: {len( anchor )} positions" )

    ref = None
    if args.mode == "dpo":
        ref = copy.deepcopy( model ).eval()
        for param in ref.parameters():
            param.requires_grad_( False )

    def report( tag: str ) -> None:
        strategic = evaluate( model, args.mode, val_records, obj_vocab, ref, args.batch )
        text = f"{tag}: strategic {'accuracy' if args.mode == 'sft' else 'preference accuracy'} {strategic:.3f}"
        if anchor_val:
            text += f", battle imitation {train.imitation_accuracy( model, anchor_val )['exact']:.3f}"
        print( text )

    report( "before" )
    optimizer = torch.optim.AdamW( model.parameters(), lr=args.lr, weight_decay=0.01 )
    for epoch in range( args.epochs ):
        model.train()
        random.shuffle( train_records )
        total, steps = 0.0, 0
        for start in range( 0, len( train_records ), args.batch ):
            chunk = train_records[start:start + args.batch]
            log_probs = option_log_probs( model, chunk, obj_vocab )
            if args.mode == "sft":
                loss = F.nll_loss( log_probs, torch.tensor( [r["target"] for r in chunk], device=device ) )
            else:
                with torch.no_grad():
                    ref_log_probs = option_log_probs( ref, chunk, obj_vocab )
                chosen = [r["chosen"] for r in chunk]
                rejected = [r["rejected"] for r in chunk]
                margin = args.beta * ( ( picked( log_probs, chosen ) - picked( ref_log_probs, chosen ) )
                                       - ( picked( log_probs, rejected ) - picked( ref_log_probs, rejected ) ) )
                loss = -F.logsigmoid( margin ).mean()
            total += loss.item()
            if anchor and args.battle_weight > 0:
                loss = loss + args.battle_weight * battle_loss( model, random.sample( anchor, min( args.batch, len( anchor ) ) ), device )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_( model.parameters(), 1.0 )
            optimizer.step()
            steps += 1
        print( f"epoch {epoch + 1}: {args.mode} loss {total / max( steps, 1 ):.4f}" )
        report( f"epoch {epoch + 1}" )
        save_checkpoint( model, args.out )

    save_checkpoint( model, args.out )
    print( f"model saved -> {args.out}" )


if __name__ == "__main__":
    main()
