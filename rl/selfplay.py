"""Self-play runner for the AlphaZero-style battle prototype (phase 1, no NN).

Generates battles, runs MCTS for every decision, records (state, visit counts, outcome)
tuples suitable for later neural network training, and verifies engine determinism by
replaying every finished game.

Usage:
    python3 rl/selfplay.py --battles 4 --sims 16 --out rl/data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine_bridge import BattleEnv  # noqa: E402
from mcts import Mcts  # noqa: E402

# A small pool of common monsters (id -> typical stack size band) for random armies.
MONSTER_POOL = [
    (1, 30, 100), (2, 30, 100), (4, 20, 60), (5, 15, 50), (12, 30, 80), (13, 15, 40),
    (16, 10, 30), (21, 10, 30), (22, 8, 20), (24, 5, 15), (27, 10, 25), (33, 5, 12),
    (36, 3, 10), (40, 4, 12), (44, 2, 8), (49, 4, 10), (54, 2, 6), (57, 2, 5), (66, 1, 4),
]


def random_army(rng: random.Random) -> str:
    """Builds a random 3..5 stack army string 'mon x count, ...'."""
    picks = rng.sample(MONSTER_POOL, rng.randint(3, 5))
    return ",".join(f"{mon}x{rng.randint(lo, hi)}" for mon, lo, hi in picks)


def state_hash(state: dict) -> str:
    payload = json.dumps(
        {
            "cur": state.get("cur"),
            "units": sorted((json.dumps(u, sort_keys=True) for u in state["units"])),
            "turn": state.get("turn"),
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def play_one(env: BattleEnv, sims: int, seed: int, attacker: str, defender: str, rng: random.Random, mcts_factory=None) -> tuple[dict, list[dict], list[str]]:
    """Plays one full battle; returns (final state, training records, action trace)."""
    state = env.new_battle(seed=seed, attacker=attacker, defender=defender)
    if state is None or "legal" not in state:
        raise RuntimeError(f"failed to start battle: {state}")

    mcts = mcts_factory()
    records: list[dict] = []
    trace: list[str] = []

    max_moves = 400
    moves_done = 0
    while state is not None and not state.get("result") and moves_done < max_moves:
        legal, counts = mcts.run(state, sims)
        if not legal:
            break

        total = sum(counts)
        if total == 0 or rng.random() < 0.25:
            # Temperature sampling for exploration (AZ-style, simplified).
            choice = rng.randrange(len(legal))
        else:
            choice = max(range(len(legal)), key=lambda i: counts[i])

        records.append(
            {
                "state_hash": state_hash(state),
                "state": {"turn": state["turn"], "units": state["units"], "obstacles": state["obstacles"], "cur": state["cur"]},
                "legal": legal,
                "counts": counts,
            }
        )

        act, args = legal[choice]
        trace.append(json.dumps({"act": act, "args": args}))
        state = env.action(act, args)
        moves_done += 1

    if state is None:
        raise RuntimeError("engine closed the connection")

    outcome = state.get("result")
    if outcome is None:
        # The move cap was hit before the battle ended: no outcome, records are not usable.
        return state, [], trace

    for record in records:
        record["outcome"] = outcome

    return state, records, trace


def verify_determinism(env: BattleEnv, seed: int, attacker: str, defender: str, trace: list[str]) -> bool:
    """Resets the battle and replays the action trace; the result must be identical."""
    state = env.new_battle(seed=seed, attacker=attacker, defender=defender)
    for line in trace:
        if state.get("result"):
            break
        cmd = json.loads(line)
        state = env.action(cmd["act"], cmd["args"])
    return bool(state and state.get("result"))


def load_policy_value(model_path: str, arch: str, device: str):
    """Loads a trained checkpoint as a policy/value object implementing .evaluate(state)."""
    import torch

    if arch == "transformer":
        from transformer_model import load_checkpoint

        model = load_checkpoint(model_path, device)
        model.eval()
        return model

    from model import AzBattleNet
    from policy_value import ResNetPolicyValue

    from transformer_model import grow_rows

    model = AzBattleNet()
    model.load_state_dict(grow_rows(model.state_dict(), torch.load(model_path, map_location=device)))
    model.to(device)
    model.eval()
    return ResNetPolicyValue(model, device)


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaZero battle prototype: self-play runner")
    parser.add_argument("--battles", type=int, default=4)
    parser.add_argument("--sims", type=int, default=16)
    parser.add_argument("--out", type=str, default="rl/data")
    parser.add_argument("--map", type=str, default="Arena.mp2")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--model", type=str, default=None, help="trained network for search guidance")
    parser.add_argument("--arch", choices=["resnet", "transformer"], default="resnet")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--att", type=str, default=None, help="attacker stacks, e.g. 13x30")
    parser.add_argument("--def", dest="def_army", type=str, default=None, help="defender stacks")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "games.jsonl")

    rng = random.Random(args.seed)
    env = BattleEnv(map_name=args.map)

    policy_value = None
    if args.model:
        policy_value = load_policy_value(args.model, args.arch, args.device)
        print(f"model loaded: {args.model} ({args.arch}) on {args.device}")

    def mcts_factory():
        return Mcts(env, policy_value=policy_value, rng=rng)

    t0 = time.time()
    total_records = 0
    try:
        with open(out_path, "a") as out:
            for battle_id in range(args.battles):
                seed = rng.randrange(1 << 30)
                attacker = args.att or random_army(rng)
                defender = args.def_army or random_army(rng)

                state, records, trace = play_one(env, args.sims, seed, attacker, defender, rng, mcts_factory)
                # Determinism check: replay the exact same game through the same engine.
                ok = verify_determinism(env, seed, attacker, defender, trace)

                if not ok:
                    print(f"battle {battle_id}: DETERMINISM CHECK FAILED", file=sys.stderr)
                    continue

                for record in records:
                    record["seed"] = seed
                    record["attacker"] = attacker
                    record["defender"] = defender
                    out.write(json.dumps(record) + "\n")
                total_records += len(records)

                print(f"battle {battle_id}: winner={state['result']} moves={len(trace)} records={len(records)}")
    finally:
        env.close()

    print(f"done in {time.time() - t0:.1f}s, {total_records} records -> {out_path}")


if __name__ == "__main__":
    main()
