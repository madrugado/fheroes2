"""Gate runner: the AlphaZero battle engine vs the built-in BattlePlanner.

Plays battles where one side is driven by our MCTS (with an optional trained policy/value
network) and the other side by the built-in battle AI via the battle server "suggest" op.
Sides alternate between battles for fairness; every battle is determinism-checked by replay.

Examples:
    # pure MCTS (heuristic evaluation) vs built-in AI:
    az/.venv/bin/python az/gate.py --battles 20 --sims 32

    # net-guided MCTS vs built-in AI:
    az/.venv/bin/python az/gate.py --battles 40 --sims 32 --model az/models/az_battle_expert_v1.pt

    # on real battles harvested from games (heroes, spells, sieges; az/harvest_battles.py):
    az/.venv/bin/python az/gate.py --setups az/data/battles_Thechaos.jsonl --battles 40 --sims 32

With --setups the battles are real game setups, usually lopsided (a hero against neutrals), so
the raw win rate says little: every (setup, side) is also played built-in vs built-in, and the
report compares our result with that baseline (better / equal / worse by outcome, then by the
share of our side's creatures still alive).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import encoding as enc  # noqa: E402
from engine_bridge import BattleEnv, new_battle_from_setup  # noqa: E402
from mcts import Mcts  # noqa: E402
from selfplay import load_policy_value, random_army  # noqa: E402

MAX_MOVES = 400


def play_gate_battle(env: BattleEnv, our_side: str | None, sims: int, seed: int, attacker: str, defender: str,
                     rng: random.Random, model=None, setup: dict | None = None) -> tuple[str | None, int, dict | None]:
    """Plays one battle (random armies, or a real `setup`); our_side None = built-in on both sides.
    Returns (winner, moves, final state)."""
    state = new_battle_from_setup(env, setup) if setup is not None else env.new_battle(seed=seed, attacker=attacker, defender=defender)
    if state is None or "legal" not in state:
        raise RuntimeError(f"failed to start battle: {state}")

    mcts = Mcts(env, policy_value=model, rng=rng, root_noise=0.0) if sims > 0 else None
    trace: list[tuple[int, list[int]]] = []

    moves = 0
    while state is not None and not state.get("result") and moves < MAX_MOVES:
        if not state.get("legal"):
            break

        mover = enc.side_to_move(state)
        if mover == our_side and mcts is not None:
            _, counts = mcts.run(state, sims)
            choice = max(range(len(state["legal"])), key=lambda i: counts[i])
            act, args = (state["legal"][choice]["act"], state["legal"][choice]["args"])
        elif mover == our_side:
            raise RuntimeError("our side needs MCTS (sims must be >= 1)")
        else:
            suggestion = env.suggest()
            expert = suggestion.get("expert")
            if expert is None:
                break  # the built-in AI has no action (should not happen mid-battle)
            act, args = expert["act"], expert["args"]

        trace.append((act, list(args)))
        state = env.action(act, args)
        moves += 1

    if state is None:
        raise RuntimeError("engine closed the connection")

    # Determinism check: replay the whole action path from the battle root in one roundtrip.
    env.reset()  # clean main line, so the replay applies exactly this battle's trace
    replayed = env.replay(trace)
    if not replayed or replayed.get("result") != state.get("result"):
        print("gate battle: DETERMINISM CHECK FAILED", file=sys.stderr)

    return state.get("result"), moves, state


def survivors(root: dict, final: dict | None, side: str) -> float:
    """Share of `side`'s creatures (by count) alive at the end."""
    start = sum(u["q"] for u in root["units"] if u["side"] == side)
    end = sum(u["q"] for u in (final or {}).get("units", []) if u["side"] == side)
    return end / start if start else 0.0


def outcome_score(winner: str | None, side: str) -> int:
    return 1 if winner == side else (0 if winner in ("draw", None) else -1)


def run_setups(args, model) -> None:
    """Paired gate on real battles: our MCTS vs built-in, compared with built-in vs built-in."""
    setups = []
    for path in args.setups:
        with open(path) as f:
            setups.extend(json.loads(line) for line in f if line.strip())
    rng = random.Random(args.seed)
    rng.shuffle(setups)

    envs: dict[str, BattleEnv] = {}
    verdicts = {"better": 0, "equal": 0, "worse": 0}
    wins = baseline_wins = played = 0
    t0 = time.time()
    try:
        for setup in setups[:args.battles]:
            map_name = setup.get("map", args.map)
            env = envs.setdefault(map_name, BattleEnv(map_name=map_name))
            root = new_battle_from_setup(env, setup)
            if root is None or "legal" not in root:
                continue
            for side in ("att", "def"):
                base_winner, _, base_final = play_gate_battle(env, None, args.sims, 0, "", "", rng, model, setup)
                winner, moves, final = play_gate_battle(env, side, args.sims, 0, "", "", rng, model, setup)
                ours = (outcome_score(winner, side), survivors(root, final, side))
                base = (outcome_score(base_winner, side), survivors(root, base_final, side))
                verdict = "better" if ours > base else ("worse" if ours < base else "equal")
                verdicts[verdict] += 1
                wins += winner == side
                baseline_wins += base_winner == side
                played += 1
                print(f"battle {setup.get('map')}:{setup.get('game_seed')}:{setup['bid']} as {side}: {verdict} "
                      f"(ours {winner}, alive {ours[1]:.2f}; built-in {base_winner}, alive {base[1]:.2f}; {moves} moves)", flush=True)
    finally:
        for env in envs.values():
            env.close()

    played = max(played, 1)
    print(f"\npaired gate in {time.time() - t0:.1f}s over {played} (setup, side) pairs: "
          f"better {verdicts['better']}, equal {verdicts['equal']}, worse {verdicts['worse']}; "
          f"win rate ours {wins / played:.0%} vs built-in {baseline_wins / played:.0%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AZ battle engine vs built-in BattlePlanner")
    parser.add_argument("--battles", type=int, default=20)
    parser.add_argument("--sims", type=int, default=32)
    parser.add_argument("--map", type=str, default="Arena.mp2")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", type=str, default=None, help="trained network checkpoint")
    parser.add_argument("--arch", choices=["resnet", "transformer"], default="resnet")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--setups", type=str, nargs="*", default=[], help="real battles (harvest_battles.py)")
    args = parser.parse_args()

    model = None
    if args.model:
        model = load_policy_value(args.model, args.arch, args.device)
        print(f"model loaded: {args.model} ({args.arch}) on {args.device}")

    if args.setups:
        run_setups(args, model)
        return

    rng = random.Random(args.seed)
    env = BattleEnv(map_name=args.map)

    stats = {"wins": 0, "losses": 0, "draws": 0}
    per_side = {"att": {"wins": 0, "battles": 0}, "def": {"wins": 0, "battles": 0}}
    total_moves = 0

    t0 = time.time()
    try:
        for battle_id in range(args.battles):
            our_side = "att" if battle_id % 2 == 0 else "def"
            seed = rng.randrange(1 << 30)
            attacker = random_army(rng)
            defender = random_army(rng)

            winner, moves, _ = play_gate_battle(env, our_side, args.sims, seed, attacker, defender, rng, model)
            total_moves += moves
            per_side[our_side]["battles"] += 1

            if winner == our_side:
                stats["wins"] += 1
                per_side[our_side]["wins"] += 1
            elif winner == "draw" or winner is None:
                stats["draws"] += 1
            else:
                stats["losses"] += 1

            print(f"battle {battle_id}: our side={our_side}, winner={winner}, moves={moves}", flush=True)
    finally:
        env.close()

    played = max(args.battles, 1)
    print(f"\ngate result in {time.time() - t0:.1f}s: "
          f"wins {stats['wins']}/{played} ({stats['wins'] / played:.0%}), "
          f"losses {stats['losses']}/{played}, draws {stats['draws']}/{played}, "
          f"avg moves {total_moves / played:.1f}")
    for side, info in per_side.items():
        if info["battles"]:
            print(f"  as {side}: {info['wins']}/{info['battles']} wins")


if __name__ == "__main__":
    main()
