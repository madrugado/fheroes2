"""Runner for strategic-layer experiments: full games where hero target choices are
made by a Python policy (see strategy_policies.py): greedy (best built-in value), random,
builtin (always skip -> the engine's own choice), tempo (value/distance-aware). Records
decision traces for future neural network training.

Usage:
    python3 az/strategy_run.py --policy greedy --playthroughs 1 --days 10 --map 2kings.mp2
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy_env import StrategyEnv  # noqa: E402
from strategy_policies import STRATEGY_POLICIES, make_strategy_policy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Strategic layer runner")
    parser.add_argument("--policy", choices=list(STRATEGY_POLICIES), default="greedy")
    parser.add_argument("--playthroughs", type=int, default=1)
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--map", type=str, default="2kings.mp2")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=str, default="az/data")
    args = parser.parse_args()

    policy = make_strategy_policy(args.policy, random.Random(args.seed))

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"strategy_{args.policy}.jsonl")

    env = StrategyEnv(map_name=args.map, days=args.days, playthroughs=args.playthroughs)
    records: list[dict] = []

    t0 = time.time()
    try:
        for item in env.run(policy, on_decision=lambda record: records.append(record)):
            if item["type"] == "game_end":
                print(f"game_end: playthrough {item.get('playthrough')} day {item.get('day')} results={item.get('results')}")
    finally:
        env.close()

    with open(out_path, "w") as out:
        for record in records:
            out.write(json.dumps(record) + "\n")

    print(f"done in {time.time() - t0:.1f}s: {len(records)} decisions -> {out_path}")


if __name__ == "__main__":
    main()
