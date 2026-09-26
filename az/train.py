"""Training for the AlphaZero battle networks on self-play / expert records.

Two architectures (choose with --arch):
  - resnet (default): AzBattleNet, fixed 793-slot action space, policy = normalized visit
    counts over the legal slots (masked), value = battle outcome.
  - transformer: AzBattleTransformer (HuggingFace GPT2 body), actions decoded as
    (target cell, direction) with teacher forcing; the direction decode reuses the prefill
    KV-cache, mirroring inference.

Records: az/data/games.jsonl (MCTS self-play) or az/data/expert.jsonl.gz (built-in AI).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys

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


def train_resnet(model, records, args, device):
    planes_list, scalars_list, slots_list, counts_list, values_list = [], [], [], [], []

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

        planes_list.append(enc.state_planes(state))
        scalars_list.append(enc.state_scalars(state))
        slots_list.append(list(slot_counts.keys()))
        counts_list.append([c / total for c in slot_counts.values()])
        values_list.append(enc.value_target(record["outcome"], mover))

    print(f"dataset: {len(planes_list)} positions ({args.arch})")
    samples = list(zip(planes_list, scalars_list, slots_list, counts_list, values_list))

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


def train_transformer(model, records, args, device):
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
        mover = enc.side_to_move(state)
        samples.append((state, {"kind": kind, "cell": cell, "dir": dir_sub},
                        enc.value_target(record["outcome"], mover)))

    print(f"dataset: {len(samples)} positions, {skipped} skipped ({args.arch})")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        random.shuffle(samples)

        total_cell_loss, total_dir_loss, total_value_loss, batches = 0.0, 0.0, 0.0, 0
        for start in range(0, len(samples), args.batch):
            batch = samples[start:start + args.batch]
            states = [s for s, _, _ in batch]
            cell_targets = torch.tensor([SKIP_CELL if t["kind"] == "skip" else t["cell"] for _, t, _ in batch],
                                        device=device)
            decode_cells = [t["cell"] if (t["kind"] == "attack" and t["cell"] is not None) else None
                            for _, t, _ in batch]
            dir_targets = torch.tensor([t["dir"] if t["dir"] is not None else 0 for _, t, _ in batch], device=device)
            value_targets = torch.tensor([v for _, _, v in batch], dtype=torch.float32, device=device)

            cell_logits, dir_out, value = model.forward_batch(states, cell_targets.tolist(), decode_cells,
                                                              dir_targets.tolist(), value_targets)

            cell_loss = F.cross_entropy(cell_logits, cell_targets)
            value_loss = F.mse_loss(value, value_targets)
            dir_loss = torch.tensor(0.0, device=device)
            if dir_out is not None:
                rows, dir_logits = dir_out
                dir_loss = F.cross_entropy(dir_logits, dir_targets[rows])

            loss = cell_loss + dir_loss + value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_cell_loss += cell_loss.item()
            total_dir_loss += dir_loss.item()
            total_value_loss += value_loss.item()
            batches += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"epoch {epoch + 1}: cell {total_cell_loss / batches:.4f}, dir {total_dir_loss / batches:.4f}, "
                  f"value {total_value_loss / batches:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AZ battle network training")
    parser.add_argument("--data", type=str, default="az/data/games.jsonl")
    parser.add_argument("--arch", choices=["resnet", "transformer"], default="resnet")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if args.out is None:
        args.out = "az/models/az_battle_v1.pt" if args.arch == "resnet" else "az/models/az_battle_tr_v1.pt"

    records = load_records(args.data)
    if not records:
        print("no usable records in", args.data)
        return

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    if args.arch == "transformer":
        from transformer_model import AzBattleTransformer

        model = AzBattleTransformer().to(device)
        print(f"transformer parameters: {sum(p.numel() for p in model.parameters())}")
        train_transformer(model, records, args, device)
    else:
        from model import AzBattleNet

        model = AzBattleNet().to(device)
        train_resnet(model, records, args, device)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"model saved -> {args.out}")


if __name__ == "__main__":
    main()
