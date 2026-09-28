"""Generates a training dataset from the built-in battle AI (expert iteration warm-start).

Drives the battle server: for each battle the engine plays both sides with the built-in
BattlePlanner ("auto" op) and streams one expert record per decision (full state with legal
moves + the built-in AI's action). The generator converts them into the same format as
rl/selfplay.py records (policy target = one-hot on the expert action) and writes a
gzipped JSONL file for rl/train.py.

Two sources of battles: random monster armies (--battles) and real battles harvested from
seeded playtests (--setups, see rl/harvest_battles.py): heroes with their skills and spells,
sieges, town garrisons. The engine hands out the built-in AI's action as the equal enumerated
legal move (attack fields resolved) and applies one command per decision, so every record's
expert action is one of its legal moves.

If a battle hangs inside the planner (rare army matchups loop forever — the client watchdog
raises after 60 seconds), the engine process is respawned and generation continues.

Usage:
    rl/.venv/bin/python rl/gen_expert.py --battles 400 --out rl/data/expert.jsonl.gz
    rl/.venv/bin/python rl/gen_expert.py --battles 0 --setups rl/data/battles_Battlefi.jsonl \
        --out rl/data/expert_real.jsonl.gz
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

from engine_bridge import BattleEnv, new_battle_from_setup  # noqa: E402
from encoding import action_index  # noqa: E402
from selfplay import MONSTER_POOL, random_army  # noqa: E402


# State fields kept in the training records (the wire state minus the legal moves).
STATE_KEYS = ("turn", "cur", "units", "obstacles", "heroes", "siege")


def convert_records(expert_records: list[dict], outcome: str) -> list[dict]:
    """Converts streamed expert records into train.py format (one-hot policy target)."""
    result = []
    for record in expert_records:
        legal = record["legal"]
        counts = [0.0] * len(legal)

        expert = record["expert"]
        if expert in legal:
            # The engine reports the expert action as the equal legal move.
            counts[legal.index(expert)] = 1.0
        else:
            # Older engines: match by action slot.
            unit_cells = {u["u"]: u["i"] for u in record["units"]}
            target = action_index(expert["act"], expert["args"], unit_cells)
            if target is None:
                # Outside our action space (retreat, surrender, ...).
                continue
            for i, move in enumerate(legal):
                act, args = (move["act"], move["args"]) if isinstance(move, dict) else (move[0], move[1])
                if action_index(act, args, unit_cells) == target:
                    counts[i] = 1.0
                    break
            else:
                continue

        state = {key: record[key] for key in STATE_KEYS if key in record}
        result.append(
            {
                "state": state,
                "legal": legal,
                "counts": counts,
                "outcome": outcome,
            }
        )
    return result


def play_battle(env: BattleEnv, seed: int, attacker: str, defender: str, setup: dict | None = None) -> tuple[list[dict], str | None]:
    """Runs one auto battle (random armies, or a harvested real battle when `setup` is given);
    returns (expert records, outcome) — outcome None if unfinished."""
    if setup is not None:
        state = new_battle_from_setup(env, setup)
    else:
        state = env.new_battle(seed=seed, attacker=attacker, defender=defender)
    if state is None or state.get("ev") != "state":
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


def load_setups(paths: list[str]) -> list[dict]:
    """battle_start events from harvest_battles.py output files (JSON lines)."""
    setups = []
    for path in paths:
        with open(path) as f:
            setups.extend(json.loads(line) for line in f if line.strip())
    return setups


def main() -> None:
    parser = argparse.ArgumentParser(description="Expert dataset generator (built-in AI)")
    parser.add_argument("--battles", type=int, default=400, help="battles between random monster armies")
    parser.add_argument("--setups", type=str, nargs="*", default=[], help="harvested real battles (harvest_battles.py)")
    parser.add_argument("--map", type=str, default="Arena.mp2", help="map of the random battles")
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--out", type=str, default="rl/data/expert.jsonl.gz")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rng = random.Random(args.seed)

    # Jobs: (battle key, map, seed, attacker, defender, setup). A real battle must be rebuilt on
    # the map of its game.
    jobs = []
    for i in range(args.battles):
        jobs.append((f"random:{args.seed}:{i}", args.map, rng.randrange(1 << 30), random_army(rng), random_army(rng), None))
    for setup in load_setups(args.setups):
        key = f"real:{setup.get('map')}:{setup.get('game_seed')}:{setup['bid']}"
        jobs.append((key, setup.get("map", args.map), setup["seed"], "", "", setup))

    total_records = 0
    skipped = 0
    hangs = 0
    unfinished = 0
    t0 = time.time()

    envs: dict[str, BattleEnv] = {}
    out = gzip.open(args.out, "wt")

    try:
        for job_id, (key, map_name, seed, attacker, defender, setup) in enumerate(jobs):
            if map_name not in envs:
                envs[map_name] = BattleEnv(map_name=map_name)
            env = envs[map_name]

            try:
                expert_records, outcome = play_battle(env, seed, attacker, defender, setup)
            except TimeoutError:
                hangs += 1
                print(f"battle {key}: HANG (planner loop), respawning engine", flush=True)
                try:
                    env.close()
                except Exception:
                    pass
                del envs[map_name]
                continue

            if outcome is None:
                unfinished += 1
                continue

            converted = convert_records(expert_records, outcome)
            skipped += len(expert_records) - len(converted)

            for record in converted:
                # "battle" groups the records of one battle (train.py splits validation by it).
                record["battle"] = key
                record["seed"] = seed
                if setup is None:
                    record["attacker"] = attacker
                    record["defender"] = defender
                out.write(json.dumps(record, separators=(",", ":")) + "\n")
            total_records += len(converted)

            if (job_id + 1) % 25 == 0 or job_id == len(jobs) - 1:
                elapsed = time.time() - t0
                print(f"battle {job_id + 1}/{len(jobs)}: {total_records} records, "
                      f"{skipped} skipped, {unfinished} unfinished, {hangs} hangs, {elapsed:.0f}s", flush=True)
    finally:
        out.close()
        for env in envs.values():
            env.close()

    print(f"done: {total_records} records, {skipped} skipped, {unfinished} unfinished, {hangs} hangs -> {args.out}")


if __name__ == "__main__":
    main()
