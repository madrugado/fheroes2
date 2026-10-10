"""Training for the AlphaZero battle networks on self-play / expert records.

Two architectures (choose with --arch):
  - resnet (default): AzBattleNet, fixed action space (encoding.ACTION_SPACE: 1460 slots incl.
    hero spells), policy = normalized visit
    counts over the legal slots (masked), value = battle outcome.
  - transformer: AzBattleTransformer (HuggingFace Qwen3 body), actions decoded as
    (target cell, direction) with teacher forcing; the direction decode reuses the prefill
    KV-cache, mirroring inference.

Records: rl/data/games.jsonl (MCTS self-play) or rl/data/expert.jsonl.gz (built-in AI); several
files may be given. `--val` holds out whole battles (the records' "battle" key, else the battle
seed) and reports the imitation accuracy on them after training: the share of positions where
the network's most probable legal move is the recorded best move ("exact"), and where it at
least lands in the same action slot ("slot": the spell targets share a slot).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import encoding as enc  # noqa: E402

SKIP_CELL = enc.NUM_CELLS  # the "skip" pseudo-cell in the transformer's cell head


def load_records(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    records = []

    with opener(path, "rt") as f:
        for line in f:
            record = json.loads(line)

            outcome = record.get("outcome")
            if outcome not in ("att", "def", "draw"):
                continue

            legal = record["legal"]
            counts = record["counts"]
            if not legal or len(legal) != len(counts) or sum(counts) <= 0:
                continue

            records.append(record)

    return records


def battle_key(record: dict) -> str:
    """Records of one battle share this key (they are highly correlated: split by battle)."""
    if "battle" in record:
        return str(record["battle"])
    return f"{record.get('seed')}:{record.get('attacker')}:{record.get('defender')}"


def attach_battle_history(records: list[dict]) -> list[dict]:
    """Gives every record's state the battle's earlier expert actions (state["history"],
    transformer_model.battle_history_entry) — the transformer sees the battle so far. Records of a
    battle are consecutive and in order in gen_expert files (checked: 3284 battles, no turn goes
    back), so a new battle key starts an empty history."""
    import transformer_model as tfm

    entries: list[dict] = []
    key = None
    for record in records:
        if battle_key(record) != key:
            entries, key = [], battle_key(record)
        state = record["state"]
        record["state"] = dict(state, history=list(entries))
        best = max(range(len(record["counts"])), key=lambda i: record["counts"][i])
        move = record["legal"][best]
        act, args_ = (move["act"], move["args"]) if isinstance(move, dict) else (move[0], move[1])
        entry = tfm.battle_history_entry(state, act, args_)
        if entry is not None:
            entries.append(entry)
    return records


def split_records(records: list[dict], val_fraction: float) -> tuple[list[dict], list[dict]]:
    """Deterministic train/validation split by battle."""
    if val_fraction <= 0:
        return records, []
    train, val = [], []
    for record in records:
        digest = hashlib.sha1(battle_key(record).encode()).digest()
        (val if digest[0] / 256.0 < val_fraction else train).append(record)
    return train, val


def imitation_accuracy(policy_value, records: list[dict]) -> dict:
    """Top-1 agreement of the policy with the recorded best move over held-out records."""
    exact = slot = value_se = 0.0
    for record in records:
        state = dict(record["state"], legal=record["legal"])
        priors, value = policy_value.evaluate(state)
        predicted = max(range(len(record["legal"])), key=lambda i: priors.get(i, 0.0))
        target = max(range(len(record["counts"])), key=lambda i: record["counts"][i])
        exact += predicted == target

        cells = enc.unit_cells_map(state["units"])
        slot_of = lambda move: enc.action_index(move["act"], move["args"], cells)  # noqa: E731
        slot += slot_of(record["legal"][predicted]) == slot_of(record["legal"][target])
        value_se += (value - enc.value_target(record["outcome"], enc.side_to_move(state), record.get("flee"), record.get("how"))) ** 2

    n = max(len(records), 1)
    return {"positions": len(records), "exact": exact / n, "slot": slot / n, "value_mse": value_se / n}


def build_resnet_samples(records: list[dict]) -> list[tuple]:
    """Converts raw records into resnet training samples:
    (planes, scalars, legal slots, normalized slot counts, value target)."""
    samples = []
    for record in records:
        state = record["state"]
        unit_cells = enc.unit_cells_map(state["units"])

        slot_counts: dict[int, float] = {}
        for move, count in zip(record["legal"], record["counts"]):
            act, args_ = (move["act"], move["args"]) if isinstance(move, dict) else (move[0], move[1])
            slot = enc.action_index(act, args_, unit_cells)
            if slot is None:
                continue
            slot_counts[slot] = slot_counts.get(slot, 0.0) + count

        total = sum(slot_counts.values())
        if total <= 0:
            continue

        mover = enc.side_to_move(state)
        samples.append(
            (
                enc.state_planes(state),
                enc.state_scalars(state),
                list(slot_counts.keys()),
                [c / total for c in slot_counts.values()],
                enc.value_target(record["outcome"], mover, record.get("flee"), record.get("how")),
            )
        )
    return samples


def build_transformer_samples(records: list[dict]) -> tuple[list[tuple], int]:
    """Converts raw records into transformer training samples
    (state, {kind, cell, dir, cells, dirs}, value target); returns (samples, skipped).

    `cells` are the legal first-step tokens (move/attack cells, skip, spells) and `dirs` the legal
    direction sub-indexes for the target cell of an attack: the losses are taken over the legal
    options only, like the ResNet's masked policy and like evaluate() renormalizes."""
    import transformer_model as tfm

    samples = []
    skipped = 0
    for record in records:
        state = record["state"]
        unit_cells = enc.unit_cells_map(state["units"])

        best = max(range(len(record["counts"])), key=lambda i: record["counts"][i])
        move = record["legal"][best]
        act, args_ = (move["act"], move["args"]) if isinstance(move, dict) else (move[0], move[1])

        target = tfm.decompose_action(act, list(args_), unit_cells)
        if target is None or target[0] is None:
            skipped += 1
            continue

        kind, cell, dir_sub = target
        legal_parts = []
        for legal_move in record["legal"]:
            l_act, l_args = (legal_move["act"], legal_move["args"]) if isinstance(legal_move, dict) else (legal_move[0], legal_move[1])
            parts = tfm.decompose_action(l_act, list(l_args), unit_cells)
            if parts is not None:
                legal_parts.append(parts)
        cells = sorted({SKIP_CELL if part[0] == "skip" else part[1] for part in legal_parts})
        dirs = sorted({part[2] for part in legal_parts if part[0] == "attack" and part[1] == cell}) if kind == "attack" else []

        mover = enc.side_to_move(state)
        samples.append((state, {"kind": kind, "cell": cell, "dir": dir_sub, "cells": cells, "dirs": dirs},
                        enc.value_target(record["outcome"], mover, record.get("flee"), record.get("how"))))
    return samples, skipped


def legal_mask(options_per_row: list[list[int] | None], width: int, device) -> torch.Tensor:
    """(rows, width) boolean mask, True for the listed options; a row without a list is all True."""
    mask = torch.zeros(len(options_per_row), width, dtype=torch.bool)
    for row, options in enumerate(options_per_row):
        if options:
            mask[row, options] = True
        else:
            mask[row, :] = True
    return mask.to(device)


def train_resnet(model, records, args, device):
    samples = build_resnet_samples(records)

    print(f"dataset: {len(samples)} positions ({args.arch})")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        model.train()
        random.shuffle(samples)

        total_policy_loss, total_value_loss, batches = 0.0, 0.0, 0
        for start in range(0, len(samples), args.batch):
            batch = samples[start:start + args.batch]
            planes_b = torch.tensor([b[0] for b in batch], dtype=torch.float32, device=device)
            scalars_b = torch.tensor([b[1] for b in batch], dtype=torch.float32, device=device)

            mask_b = torch.zeros(len(batch), enc.ACTION_SPACE, dtype=torch.bool, device=device)
            policy_b = torch.zeros(len(batch), enc.ACTION_SPACE, dtype=torch.float32, device=device)
            for i, (slots, counts) in enumerate(zip((b[2] for b in batch), (b[3] for b in batch))):
                for slot, count in zip(slots, counts):
                    mask_b[i, slot] = True
                    policy_b[i, slot] = count

            values_b = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=device)

            logits, value = model(planes_b, scalars_b, mask_b)

            log_probs = F.log_softmax(logits, dim=1)
            policy_loss = -(policy_b * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(value, values_b)

            loss = policy_loss + value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            batches += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"epoch {epoch + 1}: policy {total_policy_loss / batches:.4f}, value {total_value_loss / batches:.4f}")


def warmup_cosine(total_steps: int, warmup_fraction: float = 0.05, floor: float = 0.1):
    """LR multiplier: linear warmup, then cosine decay to `floor` (transformers are unstable with
    a constant rate from step 0)."""
    warmup = max(1, int(total_steps * warmup_fraction))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return factor


def train_transformer(model, records, args, device):
    samples, skipped = build_transformer_samples(records)

    print(f"dataset: {len(samples)} positions, {skipped} skipped ({args.arch})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(max(1, args.epochs * math.ceil(len(samples) / args.batch))))

    for epoch in range(args.epochs):
        model.train()
        random.shuffle(samples)

        total_cell_loss, total_dir_loss, total_value_loss, batches = 0.0, 0.0, 0.0, 0
        steps, started = math.ceil(len(samples) / args.batch), time.time()
        for start in range(0, len(samples), args.batch):
            batch = samples[start:start + args.batch]
            states = [s for s, _, _ in batch]
            cell_targets = torch.tensor([SKIP_CELL if t["kind"] == "skip" else t["cell"] for _, t, _ in batch],
                                        device=device)
            decode_cells = [t["cell"] if (t["kind"] == "attack" and t["cell"] is not None) else None
                            for _, t, _ in batch]
            dir_targets = torch.tensor([t["dir"] if t["dir"] is not None else 0 for _, t, _ in batch], device=device)
            value_targets = torch.tensor([v for _, _, v in batch], dtype=torch.float32, device=device)

            cell_logits, dir_out, value = model.forward_batch(states, decode_cells)

            # Losses over the legal options only (masked softmax).
            cell_mask = legal_mask([t.get("cells") for _, t, _ in batch], cell_logits.shape[1], device)
            cell_loss = F.cross_entropy(cell_logits.masked_fill(~cell_mask, -1e9), cell_targets)
            value_loss = F.mse_loss(value, value_targets)
            dir_loss = torch.tensor(0.0, device=device)
            if dir_out is not None:
                rows, dir_logits = dir_out
                dir_mask = legal_mask([batch[row][1].get("dirs") for row in rows], dir_logits.shape[1], device)
                dir_loss = F.cross_entropy(dir_logits.masked_fill(~dir_mask, -1e9), dir_targets[rows])

            loss = cell_loss + dir_loss + value_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_cell_loss += cell_loss.item()
            total_dir_loss += dir_loss.item()
            total_value_loss += value_loss.item()
            batches += 1
            if batches % 100 == 0:
                print(f"  step {batches}/{steps}: cell {total_cell_loss / batches:.4f}, dir {total_dir_loss / batches:.4f}, "
                      f"value {total_value_loss / batches:.4f}, {time.time() - started:.0f}s")

        print(f"epoch {epoch + 1}: cell {total_cell_loss / batches:.4f}, dir {total_dir_loss / batches:.4f}, "
              f"value {total_value_loss / batches:.4f}")
        # Checkpoint every epoch: a large model trains for hours.
        save_model(model, args)


def save_model(model, args) -> None:
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.arch == "transformer":
        from transformer_model import save_checkpoint

        save_checkpoint(model, args.out)
    else:
        torch.save(model.state_dict(), args.out)


def main() -> None:
    parser = argparse.ArgumentParser(description="AZ battle network training")
    parser.add_argument("--data", type=str, nargs="+", default=["rl/data/games.jsonl"])
    parser.add_argument("--val", type=float, default=0.1, help="held-out share of battles")
    parser.add_argument("--val-max", type=int, default=0, help="evaluate at most this many held-out positions (0: all)")
    parser.add_argument("--threads", type=int, default=2, help="torch CPU threads (keep the machine usable)")
    parser.add_argument("--arch", choices=["resnet", "transformer"], default="resnet")
    parser.add_argument("--size", choices=["small", "50m", "100m", "200m", "0.5b"], default="small", help="transformer size (transformer_model.PRESETS)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if args.out is None:
        args.out = "rl/models/az_battle_v1.pt" if args.arch == "resnet" else "rl/models/az_battle_tr_v1.pt"

    torch.set_num_threads(args.threads)
    sys.stdout.reconfigure(line_buffering=True)  # progress lines show up in redirected logs

    records = [record for path in args.data for record in load_records(path)]
    if not records:
        print("no usable records in", args.data)
        return
    if args.arch == "transformer":
        attach_battle_history(records)
    records, val_records = split_records(records, args.val)
    print(f"records: {len(records)} train, {len(val_records)} validation")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    if args.arch == "transformer":
        from transformer_model import AzBattleTransformer

        model = AzBattleTransformer(args.size).to(device)
        print(f"transformer parameters: {sum(p.numel() for p in model.parameters())}")
        train_transformer(model, records, args, device)
    else:
        from model import AzBattleNet

        model = AzBattleNet().to(device)
        train_resnet(model, records, args, device)

    if val_records:
        model.eval()
        if args.arch == "transformer":
            evaluator = model
        else:
            from policy_value import ResNetPolicyValue

            evaluator = ResNetPolicyValue(model, device=str(device))
        if args.val_max and len(val_records) > args.val_max:
            val_records = random.Random(0).sample(val_records, args.val_max)
        stats = imitation_accuracy(evaluator, val_records)
        print(f"validation: {stats['positions']} positions, imitation exact {stats['exact']:.3f}, "
              f"slot {stats['slot']:.3f}, value MSE {stats['value_mse']:.3f}")

    save_model(model, args)
    print(f"model saved -> {args.out}")


if __name__ == "__main__":
    main()
