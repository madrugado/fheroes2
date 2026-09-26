"""Generates a training dataset from the built-in battle AI (expert iteration warm-start).

Drives the battle server: for each battle the engine plays both sides with the built-in
BattlePlanner ("auto" op) and streams one expert record per decision (full state with legal
moves + the built-in AI's action). The generator converts them into the same format as
az/selfplay.py records (policy target = one-hot on the expert action) and writes a
gzipped JSONL file for az/train.py.

If a battle hangs inside the planner (rare army matchups loop forever — the client watchdog
raises after 60 seconds), the engine process is respawned and generation continues.

Usage:
    az/.venv/bin/python az/gen_expert.py --battles 400 --out az/data/expert.jsonl.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine_bridge import BattleEnv  # noqa: E402
from encoding import action_index  # noqa: E402
from selfplay import MONSTER_POOL, random_army  # noqa: E402


def convert_records(expert_records: list[dict], outcome: str) -> list[dict]:
    """Converts streamed expert records into train.py format (one-hot policy target)."""
    result = []
    for record in expert_records:
        legal = record["legal"]
        counts = [0.0] * len(legal)

        unit_cells = {u["u"]: u["i"] for u in record["units"]}
        target = action_index(record["expert"]["act"], record["expert"]["args"], unit_cells)
        if target is None:
            # The built-in AI chose an action outside our action space (spellcast, catapult...).
            continue

        matched = False
        for i, move in enumerate(legal):
            act, args = (move["act"], move["args"]) if isinstance(move, dict) else (move[0], move[1])
            if action_index(act, args, unit_cells) == target:
                counts[i] = 1.0
                matched = True
                break

        if not matched:
            # The built-in AI's move is not in our legal list (v0 legal-move approximation).
            continue

        result.append(
            {
                "state": {"turn": record["turn"], "units": record["units"], "obstacles": record["obstacles"], "cur": record["cur"]},
                "legal": legal,
                "counts": counts,
                "outcome": outcome,
            }
        )
    return result


def play_battle(env: BattleEnv, seed: int, attacker: str, defender: str) -> tuple[list[dict], str | None]:
    """Runs one auto battle; returns (expert records, outcome) — outcome None if unfinished."""
    state = env.new_battle(seed=seed, attacker=attacker, defender=defender)
    if state is None:
        return [], None

    env._send({"op": "auto"})
    expert_records: list[dict] = []
    for _ in range(20000):
        reply = env._read()
        if reply is None:
            return expert_records, None
        if "expert" in reply:
            expert_records.append(reply)
        elif reply.get("result"):
            return expert_records, reply["result"]
        elif reply.get("ev") == "state" and "legal" in reply:
            # The battle hit the server's round cap: unfinished.
            return expert_records, None
    return expert_records, None


def main() -> None:
    parser = argparse.ArgumentParser(description="Expert dataset generator (built-in AI)")
    parser.add_argument("--battles", type=int, default=400)
    parser.add_argument("--map", type=str, default="Arena.mp2")
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--out", type=str, default="az/data/expert.jsonl.gz")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rng = random.Random(args.seed)

    total_records = 0
    skipped = 0
    hangs = 0
    unfinished = 0
    t0 = time.time()

    env: BattleEnv | None = BattleEnv(map_name=args.map)
    out = gzip.open(args.out, "wt")

    try:
        for battle_id in range(args.battles):
            seed = rng.randrange(1 << 30)
            attacker = random_army(rng)
            defender = random_army(rng)

            try:
                expert_records, outcome = play_battle(env, seed, attacker, defender)
            except TimeoutError:
                hangs += 1
                print(f"battle {battle_id}: HANG (planner loop), respawning engine", flush=True)
                try:
                    if env is not None:
                        env.close()
                except Exception:
                    pass
                env = BattleEnv(map_name=args.map)
                continue

            if outcome is None:
                unfinished += 1
                continue

            converted = convert_records(expert_records, outcome)
            skipped += len(expert_records) - len(converted)

            for record in converted:
                record["seed"] = seed
                record["attacker"] = attacker
                record["defender"] = defender
                out.write(json.dumps(record) + "\n")
            total_records += len(converted)

            if (battle_id + 1) % 10 == 0 or battle_id == args.battles - 1:
                elapsed = time.time() - t0
                print(f"battle {battle_id + 1}/{args.battles}: {total_records} records, "
                      f"{skipped} skipped, {unfinished} unfinished, {hangs} hangs, {elapsed:.0f}s", flush=True)
    finally:
        out.close()
        if env is not None:
            env.close()

    print(f"done: {total_records} records, {skipped} skipped, {unfinished} unfinished, {hangs} hangs -> {args.out}")


if __name__ == "__main__":
    main()
