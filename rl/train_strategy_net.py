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
import contextlib
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
import encoding as enc  # noqa: E402
import stable_softmax as ss  # noqa: E402
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


def scored_advantage_loss( log_probs: torch.Tensor, records: list[dict] ) -> torch.Tensor:
    """advantage_loss with the softmax renormalised over each query's SCORED options only (the group). Over all options
    a negative advantage of the built-in answer pushed its probability to options nobody had scored (2026-10-20: the
    hero-target picks left the built-in answer in 89% of the queries, mostly for unscored targets); within the group the
    logit changes sum to zero, so the unscored options keep their place."""
    losses = []
    for row, record in enumerate( records ):
        options = torch.tensor( sorted( record["adv"] ), device=log_probs.device )
        group = log_probs[row, options]
        group = ss.log_softmax( group, 0 )
        adv = torch.tensor( [record["adv"][int( o )] for o in options], dtype=group.dtype, device=group.device )
        losses.append( -( adv * group ).sum() )
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
    return ss.log_softmax( logits, dim=1 )


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
    loss = ss.cross_entropy( cell_logits.masked_fill( ~cell_mask, -1e9 ), cell_targets ) + F.mse_loss( value, values )
    if dir_out is not None:
        rows, dir_logits = dir_out
        dir_mask = train.legal_mask( [samples[row][1].get( "dirs" ) for row in rows], dir_logits.shape[1], device )
        loss = loss + ss.cross_entropy( dir_logits.masked_fill( ~dir_mask, -1e9 ), dir_targets[rows] )
    return loss


def build_distill_samples( records: list[dict], teacher, log_every: int = 20000 ) -> list[tuple]:
    """Distillation samples (state, legal, priors, value): the frozen teacher's distribution over
    the record's legal moves, zeroed on the moves the transformer cannot decompose and
    renormalized over the rest. The battle anchor arm trains on these soft labels instead of the
    expert argmax (the teacher is the stronger battle player: grpo3 ResNet > expert imitation)."""
    import transformer_model as tfm

    samples, skipped = [], 0
    for n, record in enumerate( records ):
        if log_every and n % log_every == 0 and n:
            print( f"  distill targets: {n}/{len( records )}", flush=True )

        state = record["state"]
        # The teacher's evaluate() expects a live battle-server state (legal moves included);
        # in records the legal list sits next to the state dict.
        teacher_state = dict( state )
        teacher_state["legal"] = record["legal"]
        priors, _ = teacher.evaluate( teacher_state )

        cells = enc.unit_cells_map( state["units"] )
        probs = [0.0] * len( record["legal"] )
        for i, move in enumerate( record["legal"] ):
            act, args_ = ( move["act"], move["args"] ) if isinstance( move, dict ) else ( move[0], list( move[1] ) )
            if tfm.decompose_action( act, args_, cells ) is not None:
                probs[i] = priors.get( i, 0.0 )

        mass = sum( probs )
        if mass <= 0:
            skipped += 1
            continue

        probs = [p / mass for p in probs]
        mover = enc.side_to_move( state )
        samples.append( ( state, record["legal"], probs, enc.value_target( record["outcome"], mover, record.get( "flee" ), record.get( "how" ) ) ) )

    if skipped:
        print( f"  distill targets: {skipped} records without decomposable teacher mass skipped" )
    return samples


DISTILL_MIN_TARGET = 1e-3  # training: moves the teacher gives less are left out of the soft cross-entropy


def battle_group_forward( model, samples: list[tuple], device, min_target: float = 0.0, hints: list | None = None,
                          with_moves: bool = False ):
    """Shared-prefill forward of the battle policy over the distillation samples: log q for every
    decomposable legal move of every sample's state (one prefill per state, one direction decode
    per (state, attack cell) — the layout of train_dpo.transformer_group_logps), plus the battle
    value per state. Returns (log q per row, row -> sample index, teacher target per row, values).
    `min_target` > 0 keeps only the moves the teacher gives at least that much: their terms of
    -sum p log q are all that is left (the grpo3 teacher puts ~0.91 on one move), and only their attack
    cells get a direction decode — ~20 decodes per state were 60% of a joint21 step. The softmaxes stay
    over ALL legal cells and over all legal directions of a decoded cell, so the kept log q are exact.
    `hints` (per sample, see AzBattleTransformer.forward_batch): the JOLT teacher's forward. `with_moves`: a fifth
    return value, the legal-move index of every row."""
    import transformer_model as tfm

    state_index: dict[int, int] = {}
    states, token_masks, state_hints = [], [], []
    pair_index: dict[tuple[int, int], int] = {}
    pairs: list[tuple[int, int]] = []
    rows = []  # (state idx, first-step token, spell share, (dir pair idx, direction, legal dirs) | None, sample idx, legal idx)
    for n, sample in enumerate( samples ):
        state, legal = sample[0], sample[1]
        cells = enc.unit_cells_map( state["units"] )
        parts = [tfm.decompose_action( m["act"], m["args"], cells ) if isinstance( m, dict ) else tfm.decompose_action( m[0], list( m[1] ), cells )
                  for m in legal ]

        if id( state ) not in state_index:
            state_index[id( state )] = len( states )
            states.append( state )
            state_hints.append( hints[n] if hints is not None else None )
            token_masks.append( sorted( {train.SKIP_CELL if p[0] == "skip" else p[1] for p in parts if p is not None} ) )
        unique = state_index[id( state )]

        for i, part in enumerate( parts ):
            if part is None or ( min_target > 0 and sample[2][i] < min_target ):
                continue

            kind, cell, direction = part
            token = train.SKIP_CELL if kind == "skip" else cell
            share = sum( 1 for p in parts if p is not None and p[0] == "spell" and p[1] == token ) if kind == "spell" else 1
            pair = None
            if kind == "attack":
                if ( unique, cell ) not in pair_index:
                    pair_index[( unique, cell )] = len( pairs )
                    pairs.append( ( unique, cell ) )
                pair = ( pair_index[( unique, cell )], direction,
                         sorted( {p[2] for p in parts if p is not None and p[0] == "attack" and p[1] == cell} ) )
            rows.append( ( unique, token, share, pair, n, i ) )

    cell_logits, dir_out, value = model.forward_batch( states, [None] * len( states ), decode_pairs=pairs,
                                                       hints=state_hints if hints is not None else None )
    cell_mask = train.legal_mask( token_masks, cell_logits.shape[1], device )
    cell_logp = ss.log_softmax( cell_logits.float().masked_fill( ~cell_mask, -1e9 ), dim=1 )
    index = torch.tensor( [r[0] for r in rows], device=device )
    logq = cell_logp[index, torch.tensor( [r[1] for r in rows], device=device )]
    logq = logq - torch.log( torch.tensor( [r[2] for r in rows], dtype=torch.float32, device=device ) )

    if pairs:
        _, dir_logits = dir_out
        dir_masks = [None] * len( pairs )
        for row in rows:
            if row[3] is not None:
                dir_masks[row[3][0]] = row[3][2]
        dir_mask = train.legal_mask( dir_masks, dir_logits.shape[1], device )
        dir_logp = ss.log_softmax( dir_logits.float().masked_fill( ~dir_mask, -1e9 ), dim=1 )
        attack = torch.tensor( [i for i, r in enumerate( rows ) if r[3] is not None], device=device )
        picked = dir_logp[torch.tensor( [rows[i][3][0] for i in attack], device=device ),
                          torch.tensor( [rows[i][3][1] for i in attack], device=device )]
        logq = logq.index_add( 0, attack, picked.float() )

    targets = torch.tensor( [samples[r[4]][2][r[5]] for r in rows], dtype=torch.float32, device=device )
    owner = torch.tensor( [r[4] for r in rows], device=device )
    if with_moves:
        return logq, owner, targets, value.float(), [r[5] for r in rows]
    return logq, owner, targets, value.float()


def distill_loss( model, samples: list[tuple], device ) -> torch.Tensor:
    """Battle distillation loss: soft cross-entropy against the teacher's move distribution over
    each state's decomposable legal moves + the battle value MSE (the same value target as the
    imitation loss)."""
    logq, owner, targets, value = battle_group_forward( model, samples, device, DISTILL_MIN_TARGET )
    soft_ce = -( targets * logq ).sum() / len( samples )
    values = torch.tensor( [s[3] for s in samples], dtype=torch.float32, device=device )
    return soft_ce + F.mse_loss( value, values )


def micro_step( model, batch: list, loss_fn, weight: float, micro: int, device ):
    """One loss term of a training step (loss_fn(model, batch, device): a MEAN over the batch). With `micro` (> 0,
    smaller than the batch) the batch is split into micro-batches whose losses (weighted by their share, so the gradient
    equals the full batch's) are backpropagated at once, so only one micro-batch's activations are alive at a time. The
    battle anchor needs it (the per-attack-cell direction decodes copy the prefix KV cache, ~17 MB each on the 200m
    model: joint21 was OOM-killed on its first 64-state steps), and so do the strategic and value terms (64 histories of
    up to 512 tokens each with gradients: joint21 died three times at the same step ~150, where both came together).
    Returns (the weighted loss still to backpropagate or None, the batch's loss value)."""
    if not micro or len( batch ) <= micro:
        loss = loss_fn( model, batch, device )
        return weight * loss, loss.item()
    total = 0.0
    for start in range( 0, len( batch ), micro ):
        part = batch[start:start + micro]
        part_loss = loss_fn( model, part, device ) * ( len( part ) / len( batch ) )
        ( weight * part_loss ).backward()
        total += part_loss.item()
    return None, total


def distill_metrics( model, samples: list[tuple], device, chunk: int = 16 ) -> dict:
    """Held-out distillation quality: the share of states whose argmax move matches the teacher's
    and the mean KL(teacher || model) over the states' legal moves. In chunks of `chunk` states: every
    attack cell's direction decode copies the prefix KV cache, and the 500 held-out states in one pass
    killed joint21 at its first report (OOM)."""
    model.eval()
    agree, kl = 0, 0.0
    for start in range( 0, len( samples ), chunk ):
        part = samples[start:start + chunk]
        with torch.no_grad():
            logq, owner, targets, _ = battle_group_forward( model, part, device )
        logq, owner, targets = logq.cpu(), owner.cpu(), targets.cpu()
        for n in range( len( part ) ):
            rows = ( owner == n ).nonzero().flatten()
            q, p = logq[rows], targets[rows]
            agree += int( q.argmax() == p.argmax() )
            positive = p > 0
            kl += float( ( p[positive] * ( p[positive].log() - q[positive] ) ).sum() )

    count = max( len( samples ), 1 )
    return {"agree": agree / count, "kl": kl / count}


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


def game_pairs( args, out: str, extra: list[str], battle_model: str, battle_arch: str = "resnet" ) -> dict:
    """Paired games of play_vs_builtin.py (our color on the plan, every color of the check seeds) -> {(seed, color): pair};
    a finished run is reused."""
    import glob
    import subprocess

    if not glob.glob( os.path.join( out, "*.json" ) ):
        os.makedirs( out, exist_ok=True )
        with open( os.path.join( out, "games.log" ), "w" ) as log:
            subprocess.run( [sys.executable, os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), "play_vs_builtin.py" ),
                             "--map", args.game_check_map, "--days", str( args.game_check_days ), "--seeds", args.game_check_seeds, "--plan", args.game_check_plan,
                             "--battle", "policy", "--model", battle_model, "--arch", battle_arch, "--duel", "--hero",
                             "--out", out, *extra], stdout=log, stderr=subprocess.STDOUT, check=False )
    pairs = {}
    for path in glob.glob( os.path.join( out, "*.json" ) ):
        for pair in json.load( open( path ) ).get( "pairs", [] ):
            pairs[( pair["seed"], pair["color"] )] = pair
    return pairs


def game_check( args, model, tag: str ) -> str:
    """Whole games on the plan, paired with the all-built-in control game of play_vs_builtin (every pair metric is
    "ours - built-in AI"): the reference (built-in strategic answers, battles by the battle net), the network's
    strategic answers with the same battles, and the network for everything (its strategic answers AND its battle
    policy). Prints the mean hero / duel / army differences of each, and the network's strategy minus the reference."""
    root = args.game_check_out
    reference = game_pairs( args, os.path.join( root, "reference" ), ["--strategy", "rule", "--rule", ""], args.game_check_battle_model )
    release_memory()  # each arm's games leave heap fragments; without the trim the three arms stack up (joint18 spiked to 59 GB)
    path = os.path.join( root, f"{tag}.pt" )
    save_checkpoint( model, path )
    net = ["--strategy", "net", "--strategy-model", path, "--strategy-view", args.game_check_view]
    strategy = game_pairs( args, os.path.join( root, tag ), net, args.game_check_battle_model )
    release_memory()
    joint = game_pairs( args, os.path.join( root, f"{tag}_joint" ), net, path, "transformer" )
    release_memory()
    teacher = {}
    if args.game_check_view == "student":  # JOLT: the teacher (the built-in AI's answers as hints) plays too
        teacher_net = ["--strategy", "net", "--strategy-model", path, "--strategy-view", "teacher"]
        teacher = game_pairs( args, os.path.join( root, f"{tag}_teacher" ), teacher_net, args.game_check_battle_model )
        release_memory()
    os.remove( path )

    def wins( pairs: dict ) -> str:
        """Games won / lost and final duels won by our color, and by the same color in the control games."""
        if not pairs:
            return "none"
        rows = list( pairs.values() )
        won, lost = sum( p.get( "won", 0 ) == 1 for p in rows ), sum( p.get( "won", 0 ) == -1 for p in rows )
        duels = sum( p.get( "duel_ours", 0.0 ) > 0 for p in rows )
        control = sum( p.get( "duel_control", 0.0 ) > 0 for p in rows )
        return f"won {won}/{len( rows )}, lost {lost}, duels won {duels} (control {control})"

    tally = "; ".join( f"{name} {wins( pairs )}" for name, pairs in
                       ( ( "plan", reference ), ( "net strategy", strategy ), ( "teacher strategy", teacher ), ( "joint", joint ) ) if pairs )

    def means( pairs: dict, keys: list ) -> str:
        if not keys:
            return "none"
        values = {m: sum( pairs[k].get( m, 0.0 ) for k in keys ) / len( keys ) for m in ( "hero", "duel", "str" )}
        return f"hero {values['hero']:+.3f}, duel {values['duel']:+.3f}, army {values['str']:+.0f}"

    keys = sorted( set( reference ) & set( strategy ) )
    diff = {m: sum( strategy[k].get( m, 0.0 ) - reference[k].get( m, 0.0 ) for k in keys ) / max( len( keys ), 1 )
            for m in ( "hero", "duel", "str" )}
    return ( f", games vs the built-in AI ({len( keys )} pairs): plan+battle net {means( reference, keys )}; "
             f"+ net strategy {means( strategy, keys )} (net strategy - built-in answers: hero {diff['hero']:+.3f}, "
             f"duel {diff['duel']:+.3f}, army {diff['str']:+.0f}); joint model for everything "
             f"{means( joint, sorted( set( reference ) & set( joint ) ) )}; WINS: {tally}" )


def check_hero_ranges( ranges: list[float], what: str ) -> None:
    """Hero labels of different search ranges (strategy_games.HERO_LOG2_RANGE: 3 until 2026-10-25, then 6; the old
    records carry no "hero_range") are different scales — refuse to mix them."""
    found = sorted( set( ranges ) )
    if len( found ) > 1:
        raise SystemExit( f"{what}: hero labels of different ranges {found} (log2 army factor caps); use one range" )


def peak_reset() -> None:
    """Resets the process's peak resident memory (Linux: VmHWM); a no-op elsewhere."""
    try:
        with open( "/proc/self/clear_refs", "w" ) as f:
            f.write( "5" )
    except OSError:
        pass


def peak_gb() -> float:
    """The process's peak resident memory since the last peak_reset() in GB (Linux; 0 elsewhere)."""
    try:
        with open( "/proc/self/status" ) as f:
            for line in f:
                if line.startswith( "VmHWM:" ):
                    return int( line.split()[1] ) / 2**20
    except OSError:
        pass
    return 0.0


def release_memory() -> None:
    """Hands freed heap memory back to the system: the reports run the model on sequences of very different lengths and
    glibc kept the fragments (the joint run grew from 12 to 43 GB and was OOM-killed once)."""
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL( "libc.so.6" ).malloc_trim( 0 )
    except OSError:
        pass  # not glibc (macOS)


def main() -> None:
    parser = argparse.ArgumentParser( description="Strategic SFT / DPO of the unified transformer" )
    parser.add_argument( "mode", choices=["sft", "dpo", "pg"] )
    parser.add_argument( "--model", required=True, help="starting unified/battle transformer checkpoint, or new:<size> for a fresh one" )
    parser.add_argument( "--data", nargs="+", required=True )
    parser.add_argument( "--battle-data", nargs="*", default=[], help="battle imitation records (anchor)" )
    parser.add_argument( "--battle-distill", default=None, help="ResNet checkpoint: the battle anchor imitates "
                         "its move distribution (soft targets over the legal moves) instead of the expert argmax" )
    parser.add_argument( "--battle-weight", type=float, default=1.0 )
    parser.add_argument( "--micro", type=int, default=0, help="the strategic and value terms in micro-batches of this many records, "
                         "each with its own backward pass (64 histories of up to 512 tokens with gradients: ~20 GB each); 0 = one pass" )
    parser.add_argument( "--battle-micro", type=int, default=0, help="battle anchor in micro-batches of this many states, each "
                         "with its own backward pass (memory: the attack-cell decodes copy the prefix KV cache); 0 = one pass" )
    parser.add_argument( "--battle-max", type=int, default=20000, help="battle anchor positions loaded" )
    parser.add_argument( "--battle-grpo", nargs="*", default=[], help="battle GRPO decisions (battle_grpo.py collect --history): "
                         "group-advantage update of the battle policy over the rolled-out candidates" )
    parser.add_argument( "--battle-grpo-weight", type=float, default=1.0 )
    parser.add_argument( "--battle-grpo-batch", type=int, default=8, help="decisions per step (x ~5 options each)" )
    parser.add_argument( "--battle-kl", type=float, default=0.005, help="leash of the battle GRPO term to the starting network" )
    parser.add_argument( "--battle-grpo-val", type=int, default=300, help="held-out battle decisions evaluated per report" )
    parser.add_argument( "--short-data", nargs="*", default=[], help="strategy_short.py collect files (may still be growing): "
                         "re-read at every report, scored by the model's own value head, group-advantage term on the strategic "
                         "answers (short-horizon strategic GRPO)" )
    parser.add_argument( "--game-check-every", type=int, default=0, help="every N-th report also plays whole games with the "
                         "network's strategic answers vs the built-in answers (0 = never; the first report never)" )
    parser.add_argument( "--game-check-at-start", action="store_true", help="also check games at the first report" )
    parser.add_argument( "--game-check-seeds", default="3101-3102" )
    parser.add_argument( "--game-check-map", default="Battlefi.mp2" )
    parser.add_argument( "--game-check-days", type=int, default=30 )
    parser.add_argument( "--game-check-plan", default="champion=1,split_singles=1,max_heroes=3" )
    parser.add_argument( "--game-check-battle-model", default="rl/models/az_battle_grpo3_v4b.pt" )
    parser.add_argument( "--game-check-out", default="rl/data/game_check" )
    parser.add_argument( "--game-check-view", choices=["full", "teacher", "student"], default="full",
                         help="what the net is shown in the game checks (JOLT: student, the player without the built-in AI)" )
    parser.add_argument( "--game-grpo", nargs="*", default=[], help="strategy_game_grpo.py collect files (may still be growing): "
                         "whole-game GRPO on the exploration answers, re-read at every report" )
    parser.add_argument( "--game-grpo-weight", type=float, default=1.0 )
    parser.add_argument( "--game-grpo-batch", type=int, default=16 )
    parser.add_argument( "--game-grpo-eps", type=float, default=0.3, help="A = (R - mean) / (std + eps), hero-equivalent units" )
    parser.add_argument( "--game-grpo-kl", type=float, default=0.02 )
    parser.add_argument( "--game-grpo-values", nargs="*", default=[], help="strategy_game_grpo.py values files (a frozen critic): an "
                         "answer is credited with what its game gained after its day (actor-critic), not with the whole game" )
    parser.add_argument( "--game-grpo-lambda", type=float, default=1.0, help="GAE lambda over the days with --game-grpo-values" )
    parser.add_argument( "--game-grpo-temperature", type=float, default=2.0,
                         help="the collector's T: the update trains log softmax(log p / T), the distribution that sampled" )
    parser.add_argument( "--game-grpo-val", type=int, default=1000, help="held-out game answers evaluated per report; the full "
                         "val pool is tens of thousands of answers (78k on 976 groups = ~5h per report through the 200m net)" )
    parser.add_argument( "--jolt-games", nargs="*", default=[], help="JOLT strategic groups (strategy_game_grpo.py collect --jolt; may "
                         "still be growing): the student answers without the built-in AI, the teacher with its answers as hints (rl/jolt.py)" )
    parser.add_argument( "--jolt-battles", nargs="*", default=[], help="JOLT battle decisions (jolt.py collect; may still be growing): the "
                         "teacher sees the battle ResNet's moves as hints" )
    parser.add_argument( "--jolt-beta-t", type=float, default=0.01, help="KL(teacher || student) weight of the teacher (paper 0.001-0.03)" )
    parser.add_argument( "--jolt-beta-s", type=float, default=0.1, help="KL(student || teacher) weight of the student (paper 0.1)" )
    parser.add_argument( "--jolt-plain", action="store_true", help="JOLT without the student's own outcome advantage (default: JOLT+)" )
    parser.add_argument( "--jolt-weight", type=float, default=1.0 )
    parser.add_argument( "--jolt-game-batch", type=int, default=16 )
    parser.add_argument( "--jolt-battle-batch", type=int, default=8 )
    parser.add_argument( "--jolt-eps", type=float, default=0.3, help="strategic advantage = (R - mean) / (spread + eps) within the role" )
    parser.add_argument( "--jolt-epoch-steps", type=int, default=200, help="--epoch-by jolt: steps per epoch (checkpoint + report)" )
    parser.add_argument( "--jolt-teacher-anchor", type=float, default=0.0, help="the TEACHER's battle policy also imitates its hint "
                         "(the battle ResNet's moves; cross-entropy weight). The student gets no anchor: it learns from the teacher" )
    parser.add_argument( "--jolt-teacher-anchor-strategy", type=float, default=0.0, help="the TEACHER also imitates the built-in AI's "
                         "strategic answer (its hint), -w log pi_T(built-in), on every JOLT answer" )
    parser.add_argument( "--jolt-hint-scale", type=float, default=None, help="set the teacher's hint_scale to this at the start (a "
                         "zero-initialised one grew to 0.01 in a day of jolt1: the teacher barely saw its hint)" )
    parser.add_argument( "--jolt-keep", type=int, default=0, help="train only on the newest N battle decisions (older ones are "
                         "off-policy: the collectors played an older model); 0 = all" )
    parser.add_argument( "--jolt-keep-games", type=int, default=0, help="the same for the strategic answers (~10x as many per hour)" )
    parser.add_argument( "--keep-epochs", action="store_true", help="also save every epoch's model as <out>_e<N>.pt (jolt1 kept only "
                         "the last one, the good versions before its collapse were lost)" )
    parser.add_argument( "--jolt-val", type=int, default=300, help="held-out answers / battle decisions evaluated per report" )
    parser.add_argument( "--short-weight", type=float, default=1.0 )
    parser.add_argument( "--short-kinds", default="target,build,hire", help="query kinds the short-horizon term trains (army "
                         "left out: on a 3-day horizon the value net rewarded hoarding gold instead of hiring, 2026-10-19)" )
    parser.add_argument( "--short-batch", type=int, default=8 )
    parser.add_argument( "--short-kl", type=float, default=0.1, help="KL(reference || policy) weight of the short-horizon term" )
    parser.add_argument( "--short-ref", default="", help="reference checkpoint of that KL (default: the starting model — which "
                         "moves with every restart; pin it to keep one anchor across restarts)" )
    parser.add_argument( "--short-scale", type=float, default=3.0, help="value head output x this = label units" )
    parser.add_argument( "--short-eps", type=float, default=0.05,
                         help="GRPO normalisation: advantage = (option mean - group mean) / (group std + eps), label units" )
    parser.add_argument( "--value-data", nargs="*", default=[], help="strategic value trajectories (strategy_value.py gen)" )
    parser.add_argument( "--value-weight", type=float, default=1.0 )
    parser.add_argument( "--value-label", default="final", help="trajectory label of the value head: final, hero (log2 hero "
                         "equivalent, strategy_value.hero_score) or army" )
    parser.add_argument( "--value-scale", type=float, default=1.0, help="the label is divided by it (hero: 3) and clipped to [-1, 1]" )
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
    parser.add_argument( "--bf16", action="store_true", help="bf16 autocast on CPU (~2x faster matmuls with avx512_bf16; "
                         "with --no-grad-checkpoint ~2.4x per step on the server)" )
    parser.add_argument( "--sft-max", type=int, default=0, help="at most this many training queries (sampled; 0 = all)" )
    parser.add_argument( "--sft-kinds", default="", help="imitation steps only on these query kinds (e.g. army,build,hire while "
                         "the short-horizon term trains the hero targets); empty = all. The reports measure every kind" )
    parser.add_argument( "--val-max", type=int, default=0, help="at most this many held-out queries (sampled; 0 = all)" )
    parser.add_argument( "--gate-setups", nargs="*", default=[], help="held-out real battles (harvest_battles.py): every report "
                         "also plays the network's battle policy alone vs the built-in AI on them (gate.paired_gate, sims 0)" )
    parser.add_argument( "--gate-battles", type=int, default=60, help="setups per gate report (x 2 sides)" )
    parser.add_argument( "--sft-every", type=int, default=1, help="the strategic query batch only every N-th step (it is "
                         "learned long before the value / battle parts); 0 = never (the reports still measure it)" )
    parser.add_argument( "--value-every", type=int, default=1, help="the value batch only every N-th step" )
    parser.add_argument( "--report-minutes", type=float, default=0.0,
                         help="also report (and save the checkpoint) every this many minutes within an epoch (0 = only per epoch)" )
    parser.add_argument( "--epoch-by", choices=["sft", "value", "jolt"], default="sft",
                         help="value: an epoch is one pass over the value states, the query batches cycle" )
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
    if args.jolt_hint_scale is not None:
        with torch.no_grad():
            model.hint_scale.fill_( args.jolt_hint_scale )
        print( f"teacher hint_scale -> {args.jolt_hint_scale}" )
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
    if args.sft_kinds:
        train_records = [r for r in train_records if r["kind"] in set( args.sft_kinds.split( "," ) )]
    if args.sft_max and len( train_records ) > args.sft_max:
        train_records = random.Random( 0 ).sample( train_records, args.sft_max )
    if args.val_max and len( val_records ) > args.val_max:
        val_records = random.Random( 1 ).sample( val_records, args.val_max )
    print( f"{args.mode}: {len( train_records )} train, {len( val_records )} validation records" )

    anchor, anchor_val, distill_val = [], [], []
    if args.battle_data:
        battle = []
        for path in args.battle_data:
            battle += train.attach_battle_history( train.load_records( path )[:args.battle_max] )
        battle_train, battle_val = train.split_records( battle, 0.1 )
        anchor_val = battle_val[:500]
        if args.battle_distill:
            from selfplay import load_policy_value

            teacher = load_policy_value( args.battle_distill, "resnet", str( device ) )
            anchor = build_distill_samples( battle_train, teacher )
            distill_val = build_distill_samples( anchor_val, teacher, log_every=0 )
            print( f"battle anchor: {len( anchor )} positions, distillation targets from {args.battle_distill}" )
        else:
            anchor, _ = train.build_transformer_samples( battle_train )
            print( f"battle anchor: {len( anchor )} positions" )

    grpo_train, grpo_val = [], []
    if args.battle_grpo:
        import battle_grpo
        from train_dpo import transformer_group_logps as transformer_move_logps  # one prefill per decision

        grpo = [r for r in load_jsonl( args.battle_grpo ) if battle_grpo.group_advantages( r ) is not None]
        grpo_train, grpo_val = train.split_records( grpo, 0.1 )
        grpo_val = random.Random( 2 ).sample( grpo_val, min( args.battle_grpo_val, len( grpo_val ) ) )
        print( f"battle GRPO: {len( grpo_train )} train / {len( grpo_val )} validation decisions (weight {args.battle_grpo_weight}, "
               f"kl {args.battle_kl})" )

    value_train, value_val = [], []
    if args.value_data:
        trajectories = strategy_value.load_trajectories( args.value_data, lazy=True )  # compressed lines: ~30 GB less
        if args.value_label != "final":  # the value head predicts "final": put the chosen label there, scaled into [-1, 1]
            trajectories = [t for t in trajectories if args.value_label in t]
            if args.value_label == "hero":
                check_hero_ranges( [t["hero_range"] if "hero_range" in t else 3.0 for t in trajectories], "value data" )
            for t in trajectories:
                t["final"] = max( -1.0, min( 1.0, t[args.value_label] / args.value_scale ) )
        value_train_traj, value_val_traj = strategy_value.split_games( trajectories, args.val )
        value_train = [s for t in value_train_traj for s in strategy_value.states_of( t )]
        value_val = [s for t in value_val_traj for s in strategy_value.states_of( t )]
        if args.value_val_max and len( value_val ) > args.value_val_max:
            value_val = random.Random( 0 ).sample( value_val, args.value_val_max )
        value_mean = sum( t["final"] for t, _ in value_train ) / max( len( value_train ), 1 )
        print( f"value: {len( value_train )} train / {len( value_val )} validation states (weight {args.value_weight})" )

    sft_anchor = attach_history( load_jsonl( args.sft_data ) ) if args.mode in ( "dpo", "pg" ) and args.sft_data else []
    if sft_anchor:
        print( f"sft anchor: {len( sft_anchor )} queries (weight {args.sft_weight})" )

    short_reader = None
    short_train: list[dict] = []
    short_val: list[dict] = []
    if args.short_data:
        import strategy_short

        short_reader = strategy_short.ShortReader( args.short_data )
    precision = torch.autocast( "cpu", dtype=torch.bfloat16 ) if args.bf16 and device.type == "cpu" else contextlib.nullcontext()

    def refresh_short() -> None:
        """Scores the queries collected since the last call with the model's current value head."""
        if short_reader is None:
            return
        model.eval()
        kinds = set( args.short_kinds.split( "," ) )
        queries = [( base, record ) for base, record in short_reader.new_queries() if record["kind"] in kinds]
        scored = group_advantages( strategy_short.score_queries( model, queries, args.short_scale, precision ) )
        for record in scored:  # GRPO: centred advantages divided by the group's spread
            spread = ( sum( a * a for a in record["adv"].values() ) / len( record["adv"] ) ) ** 0.5
            record["adv"] = {option: a / ( spread + args.short_eps ) for option, a in record["adv"].items()}
        for record in scored:  # held out by game seed
            ( short_val if record["seed"] % 10 == 0 else short_train ).append( record )

    game_offsets = {path: 0 for path in args.game_grpo}
    game_train: list[dict] = []
    game_val: list[dict] = []
    game_groups = [0]
    game_ranges: list[float] = []  # the hero range of the groups read so far (all must be equal)
    game_values: dict = {}
    if args.game_grpo_values:
        import strategy_game_grpo

        game_values = strategy_game_grpo.load_values( args.game_grpo_values )
        print( f"game grpo: critic values for {len( game_values )} groups (lambda {args.game_grpo_lambda})" )

    def refresh_games() -> None:
        """Adds the groups written since the last call (complete lines only)."""
        import strategy_game_grpo

        for path in args.game_grpo:
            try:
                with open( path, "rb" ) as f:
                    f.seek( game_offsets[path] )
                    chunk = f.read()
            except FileNotFoundError:
                continue
            end = chunk.rfind( b"\n" )
            if end < 0:
                continue
            game_offsets[path] += end + 1
            for line in chunk[:end].splitlines():
                if line.strip():
                    group = json.loads( line )
                    check_hero_ranges( [group.get( "hero_range", 3.0 )] + game_ranges, "whole-game GRPO groups" )
                    game_ranges[:] = [group.get( "hero_range", 3.0 )]
                    game_groups[0] += 1
                    values = game_values.get( strategy_game_grpo.group_key( group ) ) if game_values else None
                    ( game_val if group["seed"] % 10 == 0 else game_train ).extend(
                        strategy_game_grpo.group_records( group, args.game_grpo_eps, values, args.game_grpo_lambda ) )

    # JOLT (rl/jolt.py): both files may still be growing (collectors on the model this run saves); complete lines only.
    jolt_offsets = {path: 0 for path in args.jolt_games + args.jolt_battles}
    jolt_game_train: list[dict] = []
    jolt_game_val: list[dict] = []
    jolt_battle_train: list[dict] = []
    jolt_battle_val: list[dict] = []

    def refresh_jolt() -> None:
        import jolt

        for path in args.jolt_games + args.jolt_battles:
            try:
                with open( path, "rb" ) as f:
                    f.seek( jolt_offsets[path] )
                    chunk = f.read()
            except FileNotFoundError:
                continue
            end = chunk.rfind( b"\n" )
            if end < 0:
                continue
            jolt_offsets[path] += end + 1
            for line in chunk[:end].splitlines():
                if not line.strip():
                    continue
                row = json.loads( line )
                if path in args.jolt_games:
                    ( jolt_game_val if row["seed"] % 10 == 0 else jolt_game_train ).extend( jolt.game_records( row, args.jolt_eps ) )
                else:
                    held = train.split_records( [row], 0.1 )[1]
                    ( jolt_battle_val if held else jolt_battle_train ).append( row )
        for pool in ( jolt_game_val, jolt_battle_val ):
            if args.jolt_val and len( pool ) > args.jolt_val:
                pool[:] = random.Random( 4 ).sample( pool, args.jolt_val )
        for pool, keep in ( ( jolt_game_train, args.jolt_keep_games ), ( jolt_battle_train, args.jolt_keep ) ):
            if keep and len( pool ) > keep:  # appended in file order: the newest at the end
                del pool[:len( pool ) - keep]

    if args.jolt_games or args.jolt_battles:
        refresh_jolt()
        print( f"JOLT{'' if args.jolt_plain else '+'}: {len( jolt_game_train )} train / {len( jolt_game_val )} val strategic answers, "
               f"{len( jolt_battle_train )} train / {len( jolt_battle_val )} val battle decisions (beta_T {args.jolt_beta_t}, "
               f"beta_S {args.jolt_beta_s}, weight {args.jolt_weight})" )

    ref = None
    if args.mode in ( "dpo", "pg" ) or grpo_train or short_reader is not None or args.game_grpo:
        ref = load_checkpoint( args.short_ref, str( device ) ).eval() if args.short_ref else copy.deepcopy( model ).eval()
        for param in ref.parameters():
            param.requires_grad_( False )

    majority = {kind: Counter( r["target"] for r in train_records if r["kind"] == kind ).most_common( 1 )[0][0]
                for kind in {r["kind"] for r in train_records}} if args.mode == "sft" else {}

    reports = [0]

    def report( tag: str ) -> float | None:
        """Prints the validation numbers; returns the held-out value MSE (None without value data)."""
        value_mse = None
        reports[0] += 1
        peaks: dict[str, float] = {}  # peak resident GB per part (the reports OOM-killed joint21 twice)
        release_memory()
        peak_reset()
        if args.mode == "sft":
            text = f"{tag}: strategic accuracy {json.dumps( sft_breakdown( model, val_records, obj_vocab, args.batch, majority ) )}"
        else:
            name = "held-out advantage of the argmax" if args.mode == "pg" else "strategic preference accuracy"
            text = f"{tag}: {name} {evaluate( model, args.mode, val_records, obj_vocab, ref, args.batch ):.3f}"
        peaks["strategy"] = peak_gb()
        peak_reset()
        if anchor_val:
            text += f", battle imitation {train.imitation_accuracy( model, anchor_val )['exact']:.3f}"
        peaks["imitation"] = peak_gb()
        peak_reset()
        if distill_val:
            metrics = distill_metrics( model, distill_val, device )
            text += f", distill agree {metrics['agree']:.3f} kl {metrics['kl']:.4f}"
        peaks["distill"] = peak_gb()
        peak_reset()
        if args.gate_setups:
            import gate

            model.eval()
            with torch.no_grad():
                result = gate.paired_gate( args.gate_setups, args.gate_battles, 0, model, log=None )
            text += ( f", gate {result['better']}/{result['equal']}/{result['worse']} of {result['pairs']} "
                      f"(wins {result['wins']:.0%} vs built-in {result['builtin_wins']:.0%})" )
        if short_reader is not None:
            refresh_short()
            if short_val:
                model.eval()
                with torch.no_grad():
                    picks = [option_log_probs( model, short_val[i:i + args.batch], obj_vocab ).argmax( dim=1 ).tolist()
                             for i in range( 0, len( short_val ), args.batch )]
                picks = [p for chunk in picks for p in chunk]
                advantage = sum( r["adv"].get( p, 0.0 ) for r, p in zip( short_val, picks ) ) / len( short_val )
                own = sum( r["adv"].get( r["own"], 0.0 ) for r in short_val ) / len( short_val )
                deviate = sum( p != r["own"] for r, p in zip( short_val, picks ) ) / len( short_val )
                text += ( f", short-horizon: {len( short_train )} train / {len( short_val )} val queries, advantage of the pick "
                          f"{advantage:+.4f} (built-in answer {own:+.4f}), deviates in {deviate:.0%}" )
            else:
                text += f", short-horizon: {len( short_train )} train queries"
        if args.game_grpo:
            refresh_games()
            pool = len( game_val )
            if args.game_grpo_val and pool > args.game_grpo_val:  # the full pool costs hours per report (200m x every answer)
                game_val[:] = random.Random( 3 ).sample( game_val, args.game_grpo_val )
            text += f", game GRPO: {game_groups[0]} groups, {len( game_train )} train / {len( game_val )} of {pool} val answers"
            if game_val:
                model.eval()
                with torch.no_grad():
                    lps = [ss.log_softmax( option_log_probs( model, game_val[i:i + args.batch], obj_vocab ) / args.game_grpo_temperature, dim=1 )
                           for i in range( 0, len( game_val ), args.batch )]
                taken = [float( lp[row, next( iter( r["adv"] ) )] ) for lp, start in zip( lps, range( 0, len( game_val ), args.batch ) )
                         for row, r in enumerate( game_val[start:start + args.batch] )]
                weighted = sum( next( iter( r["adv"].values() ) ) * t for r, t in zip( game_val, taken ) ) / len( game_val )
                text += f" (val: mean A x log pi_T(answer) {weighted:+.4f}, higher = the net prefers the answers of the better games)"
        if args.jolt_games or args.jolt_battles:
            import jolt

            refresh_jolt()
            text += f", JOLT: {len( jolt_game_train )} strategic / {len( jolt_battle_train )} battle train"
            if jolt_game_val:
                text += f", strategy {json.dumps( jolt.strategic_metrics( model, jolt_game_val, obj_vocab ) )}"
            if jolt_battle_val:
                text += f", battle {json.dumps( jolt.battle_metrics( model, jolt_battle_val, device ) )}"
        if grpo_val:
            scores = battle_grpo.pick_scores( transformer_move_logps, model, grpo_val, args.battle_grpo_batch, device )
            text += f", battle pick {scores['policy']:+.4f} (built-in move {scores['builtin']:+.4f})"
        peaks["gate+short+game"] = peak_gb()
        peak_reset()
        if value_val:
            value_report = strategy_value.evaluate( model, value_val, obj_vocab, args.batch, value_mean )
            value_mse = value_report["mse"]
            text += f", value {json.dumps( value_report )}"
        peaks["value"] = peak_gb()
        first = reports[0] == 1
        if args.game_check_every and ( ( first and args.game_check_at_start ) or ( not first and ( reports[0] - 1 ) % args.game_check_every == 0 ) ):
            text += game_check( args, model, tag.replace( " ", "_" ) )
        text += ", peak GB " + " ".join( f"{part} {gb:.1f}" for part, gb in peaks.items() )
        print( text )
        release_memory()
        return value_mse

    with precision:
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
    steps_per_epoch = -( -len( train_records ) // args.batch )
    if args.epoch_by == "value" and value_train:
        steps_per_epoch = -( -len( value_train ) // args.batch ) * args.value_every
    if args.epoch_by == "jolt":  # the JOLT data keeps growing (collectors): an "epoch" is a fixed number of steps
        steps_per_epoch = args.jolt_epoch_steps
    if args.schedule == "warmup-cosine":
        scheduler = torch.optim.lr_scheduler.LambdaLR( optimizer, train.warmup_cosine( max( 1, args.epochs * steps_per_epoch ), args.warmup ) )
    value_batches = []
    last_report = time.time()
    for epoch in range( args.epochs ):
        model.train()
        total, battle_total, value_total, steps = 0.0, 0.0, 0.0, 0
        timing = Counter()  # seconds per part of the step
        started = time.time()
        batches = length_batches( train_records, args.batch, batch_rng )
        for step_index in range( steps_per_epoch ):
            chunk = batches[step_index % len( batches )]
            # Gradients are zeroed first: the battle anchor may run its own micro-batch backward passes (--battle-micro).
            optimizer.zero_grad()
            step_value = None  # the strategic term's value when it was micro-batched (its loss is backpropagated already)
            with precision:  # the forward passes and losses; backward and the optimizer step stay outside
                clock = time.time()
                if args.mode == "sft" and ( args.sft_every == 0 or step_index % args.sft_every ):
                    loss = torch.zeros( (), device=device )  # this step without the strategic query batch
                elif args.mode == "sft":
                    def sft_fn( m, part, d ):
                        return smoothed_nll( option_log_probs( m, part, obj_vocab ), [r["target"] for r in part], args.label_smoothing )

                    pending, step_value = micro_step( model, chunk, sft_fn, 1.0, args.micro, device )
                    loss = pending if pending is not None else torch.zeros( (), device=device )
                elif args.mode == "pg":
                    log_probs = option_log_probs( model, chunk, obj_vocab )
                    with torch.no_grad():
                        ref_log_probs = option_log_probs( ref, chunk, obj_vocab )
                    loss = advantage_loss( log_probs, chunk ) + args.kl * kl_to_reference( log_probs, ref_log_probs )
                else:
                    log_probs = option_log_probs( model, chunk, obj_vocab )
                    with torch.no_grad():
                        ref_log_probs = option_log_probs( ref, chunk, obj_vocab )
                    chosen = [r["chosen"] for r in chunk]
                    rejected = [r["rejected"] for r in chunk]
                    margin = args.beta * ( ( picked( log_probs, chosen ) - picked( ref_log_probs, chosen ) )
                                           - ( picked( log_probs, rejected ) - picked( ref_log_probs, rejected ) ) )
                    loss = -F.logsigmoid( margin ).mean()
                total += step_value if step_value is not None else loss.item()
                timing["strategy"] += time.time() - clock
                clock = time.time()
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
                    anchor_batch = random.sample( anchor, min( args.batch, len( anchor ) ) )
                    anchor_fn = distill_loss if args.battle_distill else battle_loss
                    pending, anchor_value = micro_step( model, anchor_batch, anchor_fn, args.battle_weight, args.battle_micro, device )
                    battle_total += anchor_value
                    if pending is not None:
                        loss = loss + pending
                    if checkpointing:
                        model.body.gradient_checkpointing_enable( gradient_checkpointing_kwargs={"use_reentrant": False} )
                timing["battle imitation"] += time.time() - clock
                clock = time.time()
                if grpo_train and args.battle_grpo_weight > 0:
                    if checkpointing:  # the direction decode needs the prefill KV cache
                        model.body.gradient_checkpointing_disable()
                    grpo_term = battle_grpo.grpo_loss( transformer_move_logps, model, ref,
                                                       random.sample( grpo_train, args.battle_grpo_batch ), args.battle_kl, device )
                    battle_total += grpo_term.item()
                    loss = loss + args.battle_grpo_weight * grpo_term
                    if checkpointing:
                        model.body.gradient_checkpointing_enable( gradient_checkpointing_kwargs={"use_reentrant": False} )
                timing["battle grpo"] += time.time() - clock
                clock = time.time()
                if short_train and args.short_weight > 0:
                    short_batch = random.sample( short_train, min( args.short_batch, len( short_train ) ) )
                    short_log_probs = option_log_probs( model, short_batch, obj_vocab )
                    with torch.no_grad():
                        short_ref = option_log_probs( ref, short_batch, obj_vocab )
                    loss = loss + args.short_weight * ( scored_advantage_loss( short_log_probs, short_batch )
                                                        + args.short_kl * kl_to_reference( short_log_probs, short_ref ) )
                timing["short"] += time.time() - clock
                clock = time.time()
                if game_train and args.game_grpo_weight > 0:
                    game_batch = random.sample( game_train, min( args.game_grpo_batch, len( game_train ) ) )
                    game_log_probs = option_log_probs( model, game_batch, obj_vocab )
                    with torch.no_grad():
                        game_ref = option_log_probs( ref, game_batch, obj_vocab )
                    tempered = ss.log_softmax( game_log_probs / args.game_grpo_temperature, dim=1 )
                    loss = loss + args.game_grpo_weight * ( advantage_loss( tempered, game_batch )
                                                            + args.game_grpo_kl * kl_to_reference( game_log_probs, game_ref ) )
                timing["game grpo"] += time.time() - clock
                clock = time.time()
                if ( args.jolt_games or args.jolt_battles ) and step_index % 20 == 0:
                    refresh_jolt()  # what the collectors appended since (complete lines only)
                if args.jolt_weight > 0 and ( jolt_game_train or jolt_battle_train ):
                    import jolt

                    plus = not args.jolt_plain
                    if jolt_game_train:
                        def jolt_game_fn( m, part, d ):
                            return jolt.strategic_loss( m, part, obj_vocab, args.jolt_beta_t, args.jolt_beta_s, plus,
                                                        args.jolt_teacher_anchor_strategy )

                        pending, _ = micro_step( model, random.sample( jolt_game_train, min( args.jolt_game_batch, len( jolt_game_train ) ) ),
                                                 jolt_game_fn, args.jolt_weight, args.micro, device )
                        if pending is not None:
                            loss = loss + pending
                    if jolt_battle_train:
                        if checkpointing:  # the direction decodes need the prefill KV cache
                            model.body.gradient_checkpointing_disable()

                        def jolt_battle_fn( m, part, d ):
                            return jolt.battle_loss( m, part, d, args.jolt_beta_t, args.jolt_beta_s, plus, args.jolt_teacher_anchor )

                        pending, jolt_value = micro_step( model, random.sample( jolt_battle_train, min( args.jolt_battle_batch, len( jolt_battle_train ) ) ),
                                                          jolt_battle_fn, args.jolt_weight, args.battle_micro, device )
                        battle_total += jolt_value
                        if pending is not None:
                            loss = loss + pending
                        if checkpointing:
                            model.body.gradient_checkpointing_enable( gradient_checkpointing_kwargs={"use_reentrant": False} )
                timing["jolt"] += time.time() - clock
                clock = time.time()
                if value_train and args.value_weight > 0 and step_index % args.value_every == 0:
                    if not value_batches:  # batches of similar history length, reshuffled on every pass
                        ordered = sorted( value_train, key=lambda s: ( s[1], batch_rng.random() ) )
                        value_batches = [ordered[start:start + args.batch] for start in range( 0, len( ordered ), args.batch )]
                        batch_rng.shuffle( value_batches )
                    chunk_values = value_batches.pop()

                    def value_fn( m, part, d ):
                        target = torch.tensor( [t["final"] for t, _ in part], dtype=torch.float32, device=d )
                        return F.mse_loss( strategy_value.values( m, part, obj_vocab ), target )

                    pending, value_part = micro_step( model, chunk_values, value_fn, args.value_weight, args.micro, device )
                    value_total += value_part
                    if pending is not None:
                        loss = loss + pending
                timing["value"] += time.time() - clock
            clock = time.time()
            if loss.requires_grad:  # a step whose only term was a micro-batched battle anchor has nothing left here
                loss.backward()
            torch.nn.utils.clip_grad_norm_( model.parameters(), 1.0 )
            optimizer.step()
            timing["backward+step"] += time.time() - clock
            if scheduler is not None:
                scheduler.step()
            steps += 1
            if steps % 50 == 0:
                print( f"  step {steps}/{steps_per_epoch}: loss {total / steps:.4f}, battle {battle_total / steps:.4f}, "
                       f"value {value_total / steps:.4f}, {time.time() - started:.0f}s; forward s/step "
                       f"{ {part: round( seconds / steps, 1 ) for part, seconds in timing.items()} }", flush=True )
            if args.report_minutes and time.time() - last_report >= 60 * args.report_minutes:
                with precision:
                    report( f"epoch {epoch + 1} step {steps}" )
                save_checkpoint( model, args.out )
                model.train()
                last_report = time.time()
        print( f"epoch {epoch + 1}: {args.mode} loss {total / max( steps, 1 ):.4f}, battle {battle_total / max( steps, 1 ):.4f}, "
               f"value {value_total / max( steps, 1 ):.4f}" )
        with precision:
            value_mse = report( f"epoch {epoch + 1}" )
        save_checkpoint( model, args.out )
        if args.keep_epochs:
            save_checkpoint( model, f"{os.path.splitext( args.out )[0]}_e{epoch + 1}.pt" )
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
