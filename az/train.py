"""Training for the AlphaZero battle network on self-play records
(az/data/games.jsonl, produced by az/selfplay.py).

Each record: {state, legal, counts, outcome}. Policy target = normalized MCTS
visit counts mapped onto the fixed action space (masked); value target = game
outcome from the side-to-move perspective.

Usage:
    az/.venv/bin/python az/train.py --data az/data/games.jsonl --epochs 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import encoding as enc  # noqa: E402
from model import AzBattleNet  # noqa: E402


def load_dataset(path: str):
    planes_list, scalars_list, slots_list, counts_list, values_list = [], [], [], [], []

    with open(path) as f:
        for line in f:
            record = json.loads(line)
            state = record["state"]
            legal = record["legal"]
            counts = record["counts"]
            outcome = record["outcome"]

            slot_counts: dict[int, float] = {}
            for move, count in zip(legal, counts):
                if isinstance(move, dict):
                    act, args = move["act"], move["args"]
                else:
                    act, args = move[0], move[1]
                slot = enc.action_index(act, args)
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
            values_list.append(enc.value_target(outcome, mover))

    return planes_list, scalars_list, slots_list, counts_list, values_list


def collate(batch):
    planes = torch.tensor([b[0] for b in batch], dtype=torch.float32)
    scalars = torch.tensor([b[1] for b in batch], dtype=torch.float32)

    mask = torch.zeros(len(batch), enc.ACTION_SPACE, dtype=torch.bool)
    policy = torch.zeros(len(batch), enc.ACTION_SPACE, dtype=torch.float32)
    for i, (slots, counts) in enumerate(zip((b[2] for b in batch), (b[3] for b in batch))):
        for slot, count in zip(slots, counts):
            mask[i, slot] = True
            policy[i, slot] = count

    values = torch.tensor([b[4] for b in batch], dtype=torch.float32)
    return planes, scalars, mask, policy, values


def main() -> None:
    parser = argparse.ArgumentParser(description="AZ battle network training")
    parser.add_argument("--data", type=str, default="az/data/games.jsonl")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="az/models/az_battle_v1.pt")
    args = parser.parse_args()

    dataset = load_dataset(args.data)
    if not dataset[0]:
        print("no usable records in", args.data)
        return

    planes, scalars, slots, counts, values = dataset
    print(f"dataset: {len(planes)} positions")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AzBattleNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    samples = list(zip(planes, scalars, slots, counts, values))

    for epoch in range(args.epochs):
        model.train()
        import random

        random.shuffle(samples)

        total_policy_loss, total_value_loss, batches = 0.0, 0.0, 0
        for start in range(0, len(samples), args.batch):
            batch = samples[start:start + args.batch]
            planes_b, scalars_b, mask_b, policy_b, values_b = collate(batch)
            planes_b, scalars_b = planes_b.to(device), scalars_b.to(device)
            mask_b, policy_b, values_b = mask_b.to(device), policy_b.to(device), values_b.to(device)

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

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"model saved -> {args.out}")


if __name__ == "__main__":
    main()
