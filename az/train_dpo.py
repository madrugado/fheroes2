"""DPO fine-tuning of the battle policy on preference pairs (az/battle_prefs.py).

Direct Preference Optimization (Rafailov et al., 2023) with a frozen reference copy of the
starting network:

    loss = -log sigmoid( beta * [ (log pi(chosen) - log ref(chosen)) - (log pi(rejected) - log ref(rejected)) ] )

TRL's DPOTrainer is built for causal language models (tokenized prompt/completion pairs); our
battle networks score legal moves of a board state, so the loss is implemented here over the
same move probabilities MCTS uses (evaluate()): a masked softmax over the legal moves; spell
targets share their slot/token and split it; a transformer attack is P(target cell) x
P(direction | cell). An optional NLL term on the chosen move (`--nll`, as in RPO) keeps the
chosen moves' likelihood from drifting down. The value head is not trained.

Reports on held-out battles: preference accuracy (log pi(chosen) > log pi(rejected)) and the mean
implicit reward margin, before and after.

Usage:
    az/.venv/bin/python az/train_dpo.py --data az/data/battle_prefs.jsonl \\
        --model az/models/az_battle_expert_v3.pt --arch resnet --out az/models/az_battle_dpo_v1.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

import encoding as enc  # noqa: E402
from train import SKIP_CELL, legal_mask, split_records  # noqa: E402


def load_pairs( path: str ) -> list[dict]:
    with open( path ) as f:
        return [json.loads( line ) for line in f if line.strip()]


def _move( legal: list[dict], index: int ) -> tuple[int, list[int]]:
    move = legal[index]
    return move["act"], list( move["args"] )


def resnet_move_logps( model, records: list[dict], key: str, device ) -> torch.Tensor:
    """log pi(record[key]) under the ResNet for a batch of records."""
    planes = torch.tensor( [enc.state_planes( r["state"] ) for r in records], dtype=torch.float32, device=device )
    scalars = torch.tensor( [enc.state_scalars( r["state"] ) for r in records], dtype=torch.float32, device=device )
    mask = torch.zeros( len( records ), enc.ACTION_SPACE, dtype=torch.bool, device=device )
    targets, shares = [], []
    for row, record in enumerate( records ):
        cells = enc.unit_cells_map( record["state"]["units"] )
        slots = [enc.action_index( m["act"], m["args"], cells ) for m in record["legal"]]
        for slot in slots:
            if slot is not None:
                mask[row, slot] = True
        target = slots[record[key]]
        targets.append( target )
        shares.append( sum( 1 for slot in slots if slot == target ) )

    logits, _ = model( planes, scalars, mask )
    log_probs = F.log_softmax( logits, dim=1 )
    picked = log_probs[torch.arange( len( records ), device=device ), torch.tensor( targets, device=device )]
    return picked - torch.log( torch.tensor( shares, dtype=torch.float32, device=device ) )


def transformer_move_logps( model, records: list[dict], key: str, device ) -> torch.Tensor:
    """log pi(record[key]) under the transformer: log P(first-step token) (a spell token is split
    between its legal targets) + log P(direction | target cell) for attacks."""
    import transformer_model as tfm

    states, decode_cells, tokens, dirs, token_masks, dir_masks, shares = [], [], [], [], [], [], []
    for record in records:
        cells = enc.unit_cells_map( record["state"]["units"] )
        parts = [tfm.decompose_action( m["act"], m["args"], cells ) for m in record["legal"]]
        kind, cell, direction = parts[record[key]]
        token = SKIP_CELL if kind == "skip" else cell
        states.append( record["state"] )
        tokens.append( token )
        decode_cells.append( cell if kind == "attack" else None )
        dirs.append( direction if kind == "attack" else 0 )
        token_masks.append( sorted( {SKIP_CELL if p[0] == "skip" else p[1] for p in parts if p is not None} ) )
        dir_masks.append( sorted( {p[2] for p in parts if p is not None and p[0] == "attack" and p[1] == cell} ) )
        shares.append( sum( 1 for p in parts if p is not None and p[0] == "spell" and p[1] == token ) if kind == "spell" else 1 )

    cell_logits, dir_out, _ = model.forward_batch( states, decode_cells )
    cell_mask = legal_mask( token_masks, cell_logits.shape[1], device )
    cell_logp = F.log_softmax( cell_logits.masked_fill( ~cell_mask, -1e9 ), dim=1 )
    logp = cell_logp[torch.arange( len( records ), device=device ), torch.tensor( tokens, device=device )]
    logp = logp - torch.log( torch.tensor( shares, dtype=torch.float32, device=device ) )

    if dir_out is not None:
        rows, dir_logits = dir_out
        dir_mask = legal_mask( [dir_masks[row] for row in rows], dir_logits.shape[1], device )
        dir_logp = F.log_softmax( dir_logits.masked_fill( ~dir_mask, -1e9 ), dim=1 )
        picked = dir_logp[torch.arange( len( rows ), device=device ), torch.tensor( [dirs[row] for row in rows], device=device )]
        logp = logp.index_add( 0, torch.tensor( rows, device=device ), picked )
    return logp


def dpo_loss( pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta: float ):
    """(loss per pair, implicit reward margin per pair)."""
    margin = beta * ( ( pi_chosen - ref_chosen ) - ( pi_rejected - ref_rejected ) )
    return -F.logsigmoid( margin ), margin


def evaluate_pairs( move_logps, model, ref, pairs: list[dict], args, device ) -> dict:
    """Preference accuracy of `model` and the mean DPO margin against `ref`."""
    model.eval()
    correct, margins, n = 0, 0.0, 0
    with torch.no_grad():
        for start in range( 0, len( pairs ), args.batch ):
            batch = pairs[start:start + args.batch]
            pc, pr = move_logps( model, batch, "chosen", device ), move_logps( model, batch, "rejected", device )
            rc, rr = move_logps( ref, batch, "chosen", device ), move_logps( ref, batch, "rejected", device )
            _, margin = dpo_loss( pc, pr, rc, rr, args.beta )
            correct += int( ( pc > pr ).sum() )
            margins += float( margin.sum() )
            n += len( batch )
    return {"pairs": n, "accuracy": correct / max( n, 1 ), "margin": margins / max( n, 1 )}


def main() -> None:
    parser = argparse.ArgumentParser( description="DPO fine-tuning of the battle policy" )
    parser.add_argument( "--data", type=str, nargs="+", default=["az/data/battle_prefs.jsonl"] )
    parser.add_argument( "--model", type=str, required=True, help="starting checkpoint (also the reference)" )
    parser.add_argument( "--arch", choices=["resnet", "transformer"], default="resnet" )
    parser.add_argument( "--beta", type=float, default=0.1 )
    parser.add_argument( "--nll", type=float, default=0.0, help="weight of -log pi(chosen) (RPO-style)" )
    parser.add_argument( "--epochs", type=int, default=3 )
    parser.add_argument( "--batch", type=int, default=64 )
    parser.add_argument( "--lr", type=float, default=1e-5 )
    parser.add_argument( "--val", type=float, default=0.1 )
    parser.add_argument( "--threads", type=int, default=2 )
    parser.add_argument( "--out", type=str, required=True )
    args = parser.parse_args()

    torch.set_num_threads( args.threads )
    sys.stdout.reconfigure( line_buffering=True )
    device = torch.device( "mps" if torch.backends.mps.is_available() else "cpu" )

    pairs = [p for path in args.data for p in load_pairs( path )]
    train_pairs, val_pairs = split_records( pairs, args.val )
    print( f"pairs: {len( train_pairs )} train, {len( val_pairs )} validation" )

    if args.arch == "transformer":
        from transformer_model import load_checkpoint, save_checkpoint

        model = load_checkpoint( args.model, str( device ) )
        move_logps = transformer_move_logps
    else:
        from model import AzBattleNet

        model = AzBattleNet()
        model.load_state_dict( torch.load( args.model, map_location="cpu" ) )
        model.to( device )
        move_logps = resnet_move_logps

    ref = copy.deepcopy( model ).eval()
    for param in ref.parameters():
        param.requires_grad_( False )

    before = evaluate_pairs( move_logps, model, ref, val_pairs, args, device )
    print( f"validation before: accuracy {before['accuracy']:.3f} ({before['pairs']} pairs)" )

    optimizer = torch.optim.AdamW( [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0 )
    for epoch in range( args.epochs ):
        model.train()
        random.shuffle( train_pairs )
        total, batches = 0.0, 0
        for start in range( 0, len( train_pairs ), args.batch ):
            batch = train_pairs[start:start + args.batch]
            pc, pr = move_logps( model, batch, "chosen", device ), move_logps( model, batch, "rejected", device )
            with torch.no_grad():
                rc, rr = move_logps( ref, batch, "chosen", device ), move_logps( ref, batch, "rejected", device )
            losses, _ = dpo_loss( pc, pr, rc, rr, args.beta )
            loss = losses.mean() - args.nll * pc.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_( model.parameters(), 1.0 )
            optimizer.step()
            total += losses.mean().item()
            batches += 1
        after = evaluate_pairs( move_logps, model, ref, val_pairs, args, device )
        print( f"epoch {epoch + 1}: dpo loss {total / max( batches, 1 ):.4f} (random: {math.log( 2 ):.4f}), "
               f"validation accuracy {after['accuracy']:.3f}, margin {after['margin']:.3f}" )

    os.makedirs( os.path.dirname( args.out ) or ".", exist_ok=True )
    if args.arch == "transformer":
        save_checkpoint( model, args.out )
    else:
        torch.save( model.state_dict(), args.out )
    print( f"model saved -> {args.out}" )


if __name__ == "__main__":
    main()
