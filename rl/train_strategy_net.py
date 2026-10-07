"""Trains the strategic output of the unified transformer (rl/strategy_net.py).

    sft — imitation of the built-in AI's answers (cross-entropy over the masked option softmax);
    dpo — Direct Preference Optimization on counterfactual preference pairs against a frozen
          reference copy of the starting network:
          -log sigmoid(beta * [(log pi(chosen) - log ref(chosen)) - (log pi(rejected) - log ref(rejected))]).

The body is shared with the battle policy, so every step also takes a battle imitation batch
(--battle-data, masked losses as in train.py) weighted by --battle-weight: training the strategic
head must not erase the battle skills. Validation (held-out seeds): strategic imitation accuracy
(sft) or preference accuracy (dpo), plus the battle imitation accuracy on held-out battles.

--value-data (strategy_value.py gen trajectories) adds a strategic value batch per step (MSE of the
value head, --value-weight), so one run trains all three outputs of the unified model; with
--model new:<size> (transformer_model.PRESETS) it starts from a fresh network — use
--schedule warmup-cosine and a from-scratch learning rate then.

Usage:
    rl/.venv/bin/python rl/train_strategy_net.py sft --model rl/models/az_battle_tr50m_expert_v4.pt \\
        --data rl/data/strategy_sft_2kings.jsonl --battle-data rl/data/expert.jsonl.gz --out rl/models/unified_sft.pt
    rl/.venv/bin/python rl/train_strategy_net.py dpo --model rl/models/unified_sft.pt \\
        --data rl/data/strategy_prefs_Battlefi.jsonl --battle-data rl/data/expert.jsonl.gz --out rl/models/unified_dpo.pt
    rl/.venv/bin/python rl/train_strategy_net.py sft --model new:100m --window 512 --data rl/data/strategy_sft_2kings_v2.jsonl \
        --battle-data rl/data/expert.jsonl.gz --battle-max 100000 --value-data rl/data/strategy_value_2kings.jsonl \
        --schedule warmup-cosine --lr 3e-4 --epochs 20 --out rl/models/unified_100m.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from collections import Counter

import torch
import torch.nn.functional as F

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

import train  # noqa: E402
from strategy_net import MAX_STRATEGIC_TOKENS, attach_history, obj_vocab_of, query_tokens  # noqa: E402
import strategy_value  # noqa: E402
from transformer_model import AzBattleTransformer, load_checkpoint, save_checkpoint, strategic_logits  # noqa: E402


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


def gap_z( chosen: list, rejected: list ) -> tuple[float, float] | None:
    """(mean, standard error) of the per-luck gaps chosen - rejected (None parts skipped); None when
    fewer than two lucks pair up."""
    import math
    import statistics

    gaps = [c - r for c, r in zip( chosen or [], rejected or [] ) if c is not None and r is not None]
    if len( gaps ) < 2:
        return None
    return statistics.fmean( gaps ), statistics.stdev( gaps ) / math.sqrt( len( gaps ) )


def reliable_pairs( pairs: list[dict], min_z: float, rule: str = "score" ) -> list[dict]:
    """DPO pairs whose chosen-minus-rejected difference over the luck replays (salt_scores, strategy_games
    --salts) has a mean above `min_z` standard errors; pairs without per-luck scores are dropped.

    rule "army_hero" (user decision 2026-10-06, pairs of `--label hero` with `salt_parts`): the ARMY part
    decides (mean gap above `min_z` standard errors) and the HERO part must not be against it (its mean
    gap >= 0). Every ordered pair of the query's scored options is tried and the one with the clearest
    army gap is kept, re-oriented (chosen/rejected) accordingly."""
    kept = []
    for pair in pairs:
        if rule == "army_hero":
            parts = pair.get( "salt_parts" ) or {}
            best = None
            for chosen in parts:
                for rejected in parts:
                    if chosen == rejected:
                        continue
                    army = gap_z( parts[chosen]["army"], parts[rejected]["army"] )
                    hero = gap_z( parts[chosen]["hero"], parts[rejected]["hero"] )
                    if army is None or army[0] <= 0 or army[0] <= min_z * army[1] or ( hero is not None and hero[0] < 0 ):
                        continue
                    z = army[0] / army[1] if army[1] > 0 else float( "inf" )
                    if best is None or z > best[0]:
                        best = ( z, int( chosen ), int( rejected ) )
            if best is not None:
                kept.append( dict( pair, chosen=best[1], rejected=best[2] ) )
            continue
        gap = gap_z( ( pair.get( "salt_scores" ) or {} ).get( str( pair["chosen"] ) ),
                     ( pair.get( "salt_scores" ) or {} ).get( str( pair["rejected"] ) ) )
        if gap is not None and gap[0] > min_z * gap[1] and gap[0] > 0:
            kept.append( pair )
    return kept


def group_advantages( records: list[dict], rule: str = "mean" ) -> list[dict]:
    """`pg` mode (user decision 2026-10-07, the GRPO idea without selecting pairs): every scored option
    of a query weighted by its advantage — the mean over the lucks of its label (salt_parts of
    `--label hero` combined by strategy_games.combine_hero_parts with `rule`, else salt_scores),
    centred on the group's mean. Noise in the weights averages out over many queries instead of being
    selected by a best/worst choice. Records get "adv": {option: centred advantage}; groups with fewer
    than two scored options or no spread are dropped."""
    from strategy_games import combine_hero_parts

    kept = []
    for record in records:
        means = {}
        parts = record.get( "salt_parts" )
        if parts:
            for option, by_part in parts.items():
                values = [combine_hero_parts( h, a, rule ) for h, a in zip( by_part["hero"], by_part["army"] )]
                if values:
                    means[int( option )] = sum( values ) / len( values )
        else:
            for option, values in ( record.get( "salt_scores" ) or {} ).items():
                if values:
                    means[int( option )] = sum( values ) / len( values )
        if len( means ) < 2:
            continue
        centre = sum( means.values() ) / len( means )
        advantages = {option: value - centre for option, value in means.items()}
        if max( abs( a ) for a in advantages.values() ) < 1e-9:
            continue
        kept.append( dict( record, adv=advantages ) )
    return kept


def advantage_loss( log_probs: torch.Tensor, records: list[dict] ) -> torch.Tensor:
    """-sum_i adv_i log pi(i) per query (mean over the batch): raises the options that did better than
    the group, lowers the others, whatever their current probability (an expected-reward gradient
    sum_i pi(i) adv_i vanishes on the confident SFT policy)."""
    losses = []
    for row, record in enumerate( records ):
        losses.append( -sum( adv * log_probs[row, option] for option, adv in record["adv"].items() ) )
    return torch.stack( losses ).mean()


def kl_to_reference( log_probs: torch.Tensor, ref_log_probs: torch.Tensor ) -> torch.Tensor:
    """KL(ref || pi) over each query's real options (mean over the batch)."""
    real = ref_log_probs > -1e8
    ref_probs = ref_log_probs.exp().masked_fill( ~real, 0.0 )
    return ( ref_probs * ( ref_log_probs.masked_fill( ~real, 0.0 ) - log_probs.masked_fill( ~real, 0.0 ) ) ).sum( dim=1 ).mean()


def history_length( record: dict ) -> int:
    """A cheap proxy of a record's sequence length: its history days and decisions."""
    history = record.get( "history" ) or {}
    return 6 * len( history.get( "days" ) or [] ) + len( history.get( "decisions" ) or [] )


def length_batches( records: list[dict], batch: int, rng: random.Random ) -> list[list[dict]]:
    """Batches of records of similar length in random order: a batch is padded to its longest
    sequence, so mixing a 50-token query with a 490-token one wastes most of the step (and of the
    activation memory)."""
    ordered = sorted( records, key=lambda r: ( history_length( r ), rng.random() ) )
    batches = [ordered[start:start + batch] for start in range( 0, len( ordered ), batch )]
    rng.shuffle( batches )
    return batches


def option_log_probs( model, records: list[dict], obj_vocab: list[int] ) -> torch.Tensor:
    """(B, max options) log-softmax over each query's options (padding: -inf-like)."""
    window = model.config.get( "window", MAX_STRATEGIC_TOKENS )
    queries = [query_tokens( r["kind"], r["event"], r.get( "context" ), obj_vocab, r.get( "history" ), window ) for r in records]
    logits, _ = strategic_logits( model, queries )
    return F.log_softmax( logits, dim=1 )


def smoothed_nll( log_probs: torch.Tensor, targets: list[int], smoothing: float ) -> torch.Tensor:
    """Cross-entropy with label smoothing spread over each query's REAL options (padding excluded)."""
    target = torch.tensor( targets, device=log_probs.device )
    nll = F.nll_loss( log_probs, target )
    if smoothing <= 0:
        return nll
    real = log_probs > -1e8
    uniform = -( log_probs.masked_fill( ~real, 0.0 ).sum( dim=1 ) / real.sum( dim=1 ) ).mean()
    return ( 1.0 - smoothing ) * nll + smoothing * uniform


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
    where log pi(chosen) > log pi(rejected); pg: the mean advantage of the argmax option (0 = as
    good as the group's mean; positive = the policy picks what did better on these held-out queries)."""
    model.eval()
    hits = 0
    with torch.no_grad():
        for start in range( 0, len( records ), batch ):
            chunk = records[start:start + batch]
            log_probs = option_log_probs( model, chunk, obj_vocab )
            if mode == "sft":
                hits += int( ( log_probs.argmax( dim=1 ).cpu() == torch.tensor( [r["target"] for r in chunk] ) ).sum() )
            elif mode == "pg":
                for row, record in enumerate( chunk ):
                    scored = sorted( record["adv"] )
                    best = max( scored, key=lambda option: float( log_probs[row, option] ) )
                    hits += record["adv"][best]
            else:
                hits += int( ( picked( log_probs, [r["chosen"] for r in chunk] ) > picked( log_probs, [r["rejected"] for r in chunk] ) ).sum() )
    return hits / max( len( records ), 1 )


def sft_breakdown( model, records: list[dict], obj_vocab: list[int], batch: int, majority: dict[str, int] ) -> dict:
    """Imitation accuracy per query kind and on the queries whose answer is NOT the kind's usual one
    (`majority`: the most frequent training answer per kind — build and target are always option 0,
    army always 2, hire 2 in 83%), next to the accuracy of always answering the majority."""
    model.eval()
    rows = []
    with torch.no_grad():
        for start in range( 0, len( records ), batch ):
            chunk = records[start:start + batch]
            predicted = option_log_probs( model, chunk, obj_vocab ).argmax( dim=1 ).tolist()
            rows += [( r["kind"], r["target"], p ) for r, p in zip( chunk, predicted )]
    report: dict = {}

    def share( subset ) -> str:
        return f"{sum( t == p for _, t, p in subset ) / len( subset ):.3f} ({len( subset )})" if subset else "-"

    report["all"] = share( rows )
    report["majority baseline"] = f"{sum( t == majority.get( k ) for k, t, _ in rows ) / max( len( rows ), 1 ):.3f}"
    for kind in sorted( {k for k, _, _ in rows} ):
        report[kind] = share( [row for row in rows if row[0] == kind] )
    report["non-majority"] = share( [row for row in rows if row[1] != majority.get( row[0] )] )
    return report


def main() -> None:
    parser = argparse.ArgumentParser( description="Strategic SFT / DPO of the unified transformer" )
    parser.add_argument( "mode", choices=["sft", "dpo", "pg"] )
    parser.add_argument( "--model", required=True, help="starting unified/battle transformer checkpoint, or new:<size> for a fresh one" )
    parser.add_argument( "--data", nargs="+", required=True )
    parser.add_argument( "--battle-data", nargs="*", default=[], help="battle imitation records (anchor)" )
    parser.add_argument( "--battle-weight", type=float, default=1.0 )
    parser.add_argument( "--battle-max", type=int, default=20000, help="battle anchor positions loaded" )
    parser.add_argument( "--value-data", nargs="*", default=[], help="strategic value trajectories (strategy_value.py gen)" )
    parser.add_argument( "--value-weight", type=float, default=1.0 )
    parser.add_argument( "--value-val-max", type=int, default=1000, help="held-out value states evaluated per report" )
    parser.add_argument( "--min-z", type=float, default=0.0,
                         help="dpo: keep only pairs whose per-luck gap has a mean above this many standard errors (0: all)" )
    parser.add_argument( "--reliable-rule", choices=["score", "army_hero"], default="score",
                         help="--min-z on the pair's label (score) or on the army part with the hero part not against (army_hero)" )
    parser.add_argument( "--beta", type=float, default=0.1 )
    parser.add_argument( "--kl", type=float, default=0.1, help="pg: weight of KL(reference || policy)" )
    parser.add_argument( "--hero-rule", choices=["hero", "army", "mean"], default="mean",
                         help="pg: how the parts of --label hero make the advantage" )
    parser.add_argument( "--label-smoothing", type=float, default=0.0,
                         help="sft: keeps the policy from becoming certain of the built-in answer (an SFT model with "
                              "log p ~ -25 on the other options leaves DPO no room: its loss vanishes long before an "
                              "argmax flips)" )
    parser.add_argument( "--sft-data", nargs="*", default=[], help="dpo: built-in answers (strategy_net.py sft) as an anchor" )
    parser.add_argument( "--sft-weight", type=float, default=1.0, help="dpo: weight of the anchor's imitation loss" )
    parser.add_argument( "--epochs", type=int, default=3 )
    parser.add_argument( "--batch", type=int, default=32 )
    parser.add_argument( "--lr", type=float, default=5e-5 )
    parser.add_argument( "--schedule", choices=["constant", "warmup-cosine"], default="constant",
                         help="warmup-cosine (train.warmup_cosine) for a fresh network" )
    parser.add_argument( "--warmup", type=float, default=0.05, help="warmup-cosine: share of the steps spent warming up" )
    parser.add_argument( "--val", type=float, default=0.15 )
    parser.add_argument( "--threads", type=int, default=2 )
    parser.add_argument( "--no-grad-checkpoint", action="store_true",
                         help="keep every layer's activations (fast, but 32 x 490 tokens need ~17 GB on the 50m model)" )
    parser.add_argument( "--window", type=int, default=0, help="set the model's context window (0: keep the checkpoint's)" )
    parser.add_argument( "--out", required=True )
    args = parser.parse_args()

    torch.set_num_threads( args.threads )
    sys.stdout.reconfigure( line_buffering=True )
    device = torch.device( "mps" if torch.backends.mps.is_available() else "cpu" )

    if args.model.startswith( "new:" ):
        model = AzBattleTransformer( args.model[len( "new:" ):] ).to( device )
        print( f"fresh {args.model[len( 'new:' ):]} model: {sum( p.numel() for p in model.parameters() )} parameters" )
    else:
        model = load_checkpoint( args.model, str( device ) )
    if args.window:
        model.config["window"] = args.window
        model.body.config.max_position_embeddings = args.window
    records = load_jsonl( args.data )
    if args.mode == "dpo" and args.min_z > 0:
        total = len( records )
        records = reliable_pairs( records, args.min_z, args.reliable_rule )
        print( f"reliable pairs ({args.reliable_rule}: mean gap > {args.min_z} SE): {len( records )} of {total}" )
    if args.mode == "pg":
        total = len( records )
        records = group_advantages( records, args.hero_rule )
        print( f"pg groups (queries with >= 2 scored options and some spread): {len( records )} of {total}" )
    if args.mode == "sft":
        attach_history( records )  # the player's previous days and (built-in) answers in the game
    if "obj_vocab" not in model.config:
        model.config["obj_vocab"] = obj_vocab_of( records )
    obj_vocab = model.config["obj_vocab"]
    train_records, val_records = split_by_seed( records, args.val )
    print( f"{args.mode}: {len( train_records )} train, {len( val_records )} validation records" )

    anchor, anchor_val = [], []
    if args.battle_data:
        battle = []
        for path in args.battle_data:
            battle += train.attach_battle_history( train.load_records( path )[:args.battle_max] )
        battle_train, battle_val = train.split_records( battle, 0.1 )
        anchor, _ = train.build_transformer_samples( battle_train )
        anchor_val = battle_val[:500]
        print( f"battle anchor: {len( anchor )} positions" )

    value_train, value_val = [], []
    if args.value_data:
        value_train_traj, value_val_traj = strategy_value.split_games( strategy_value.load_trajectories( args.value_data ), args.val )
        value_train = [s for t in value_train_traj for s in strategy_value.states_of( t )]
        value_val = [s for t in value_val_traj for s in strategy_value.states_of( t )]
        if args.value_val_max and len( value_val ) > args.value_val_max:
            value_val = random.Random( 0 ).sample( value_val, args.value_val_max )
        value_mean = sum( t["final"] for t, _ in value_train ) / max( len( value_train ), 1 )
        print( f"value: {len( value_train )} train / {len( value_val )} validation states (weight {args.value_weight})" )

    sft_anchor = attach_history( load_jsonl( args.sft_data ) ) if args.mode in ( "dpo", "pg" ) and args.sft_data else []
    if sft_anchor:
        print( f"sft anchor: {len( sft_anchor )} queries (weight {args.sft_weight})" )

    ref = None
    if args.mode in ( "dpo", "pg" ):
        ref = copy.deepcopy( model ).eval()
        for param in ref.parameters():
            param.requires_grad_( False )

    majority = {kind: Counter( r["target"] for r in train_records if r["kind"] == kind ).most_common( 1 )[0][0]
                for kind in {r["kind"] for r in train_records}} if args.mode == "sft" else {}

    def report( tag: str ) -> float | None:
        """Prints the validation numbers; returns the held-out value MSE (None without value data)."""
        value_mse = None
        if args.mode == "sft":
            text = f"{tag}: strategic accuracy {json.dumps( sft_breakdown( model, val_records, obj_vocab, args.batch, majority ) )}"
        else:
            name = "held-out advantage of the argmax" if args.mode == "pg" else "strategic preference accuracy"
            text = f"{tag}: {name} {evaluate( model, args.mode, val_records, obj_vocab, ref, args.batch ):.3f}"
        if anchor_val:
            text += f", battle imitation {train.imitation_accuracy( model, anchor_val )['exact']:.3f}"
        if value_val:
            value_report = strategy_value.evaluate( model, value_val, obj_vocab, args.batch, value_mean )
            value_mse = value_report["mse"]
            text += f", value {json.dumps( value_report )}"
        print( text )
        return value_mse

    best_value_mse = report( "before" )
    optimizer = torch.optim.AdamW( model.parameters(), lr=args.lr, weight_decay=0.01 )
    # Gradient checkpointing: the strategic sequences are long (up to the window), and keeping every
    # layer's activations for the backward pass filled the 16 GB laptop (measured on the 50m model,
    # 32 x 490 tokens: 16.7 GB and 93 s per step in swap; with checkpointing 3.6 GB, 5.3 s).
    checkpointing = not args.no_grad_checkpoint
    if checkpointing:
        model.body.gradient_checkpointing_enable( gradient_checkpointing_kwargs={"use_reentrant": False} )
    batch_rng = random.Random( 1 )
    scheduler = None
    if args.schedule == "warmup-cosine":
        steps_per_epoch = -( -len( train_records ) // args.batch )
        scheduler = torch.optim.lr_scheduler.LambdaLR( optimizer, train.warmup_cosine( max( 1, args.epochs * steps_per_epoch ), args.warmup ) )
    value_batches = []
    for epoch in range( args.epochs ):
        model.train()
        total, battle_total, value_total, steps = 0.0, 0.0, 0.0, 0
        started = time.time()
        batches = length_batches( train_records, args.batch, batch_rng )
        for chunk in batches:
            log_probs = option_log_probs( model, chunk, obj_vocab )
            if args.mode == "sft":
                loss = smoothed_nll( log_probs, [r["target"] for r in chunk], args.label_smoothing )
            elif args.mode == "pg":
                with torch.no_grad():
                    ref_log_probs = option_log_probs( ref, chunk, obj_vocab )
                loss = advantage_loss( log_probs, chunk ) + args.kl * kl_to_reference( log_probs, ref_log_probs )
            else:
                with torch.no_grad():
                    ref_log_probs = option_log_probs( ref, chunk, obj_vocab )
                chosen = [r["chosen"] for r in chunk]
                rejected = [r["rejected"] for r in chunk]
                margin = args.beta * ( ( picked( log_probs, chosen ) - picked( ref_log_probs, chosen ) )
                                       - ( picked( log_probs, rejected ) - picked( ref_log_probs, rejected ) ) )
                loss = -F.logsigmoid( margin ).mean()
            total += loss.item()
            if sft_anchor and args.sft_weight > 0:
                # Imitation of the built-in answers keeps DPO close to sensible play while the
                # preference data is small.
                sft_batch = random.sample( sft_anchor, min( args.batch, len( sft_anchor ) ) )
                sft_log_probs = option_log_probs( model, sft_batch, obj_vocab )
                loss = loss + args.sft_weight * smoothed_nll( sft_log_probs, [r["target"] for r in sft_batch], args.label_smoothing )
            if anchor and args.battle_weight > 0:
                # The battle decode needs the prefill KV cache, which checkpointing turns off.
                if checkpointing:
                    model.body.gradient_checkpointing_disable()
                anchor_loss = battle_loss( model, random.sample( anchor, min( args.batch, len( anchor ) ) ), device )
                battle_total += anchor_loss.item()
                loss = loss + args.battle_weight * anchor_loss
                if checkpointing:
                    model.body.gradient_checkpointing_enable( gradient_checkpointing_kwargs={"use_reentrant": False} )
            if value_train and args.value_weight > 0:
                if not value_batches:  # batches of similar history length, reshuffled on every pass
                    ordered = sorted( value_train, key=lambda s: ( s[1], batch_rng.random() ) )
                    value_batches = [ordered[start:start + args.batch] for start in range( 0, len( ordered ), args.batch )]
                    batch_rng.shuffle( value_batches )
                chunk_values = value_batches.pop()
                target = torch.tensor( [t["final"] for t, _ in chunk_values], dtype=torch.float32, device=device )
                value_loss = F.mse_loss( strategy_value.values( model, chunk_values, obj_vocab ), target )
                value_total += value_loss.item()
                loss = loss + args.value_weight * value_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_( model.parameters(), 1.0 )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            steps += 1
            if steps % 50 == 0:
                print( f"  step {steps}/{len( batches )}: loss {total / steps:.4f}, battle {battle_total / steps:.4f}, "
                       f"value {value_total / steps:.4f}, {time.time() - started:.0f}s", flush=True )
        print( f"epoch {epoch + 1}: {args.mode} loss {total / max( steps, 1 ):.4f}, battle {battle_total / max( steps, 1 ):.4f}, "
               f"value {value_total / max( steps, 1 ):.4f}" )
        value_mse = report( f"epoch {epoch + 1}" )
        save_checkpoint( model, args.out )
        # The value head overfits within a few epochs: keep the epoch with the best held-out value MSE.
        if value_mse is not None and ( best_value_mse is None or value_mse < best_value_mse ):
            best_value_mse = value_mse
            best_path = os.path.splitext( args.out )[0] + "_best.pt"
            save_checkpoint( model, best_path )
            print( f"best value so far (MSE {value_mse}) -> {best_path}" )

    save_checkpoint( model, args.out )
    print( f"model saved -> {args.out}" )


if __name__ == "__main__":
    main()
