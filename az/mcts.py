"""Pure MCTS for the battle prototype (phase 1: no neural network).

The search runs against the live engine: to evaluate a node, the battle is reset to its
initial position and the action path from the root is replayed (the engine is deterministic).
The leaf value is a heuristic material-strength estimate; move priors are uniform.

Once the neural value/policy network exists, only `evaluate_leaf` and the priors change.
"""

from __future__ import annotations

import math
import random

# Rough per-monster strength table is not needed in phase 1: stack "value" is estimated as
# count * monster tier approximation passed in via the state itself (quantity + hp left).

ACT_MOVE = 0
ACT_ATTACK = 1
ACT_SKIP = 8


def _stack_value(unit: dict) -> float:
    """Cheap material proxy: count times per-unit HP pool factor."""
    # monster id is used only as a hash-spread factor until a real table exists
    tier = 1.0 + (unit["mon"] % 7) * 0.35
    return unit["q"] * tier


def evaluate_state(state: dict, side: str) -> float:
    """Heuristic value in [-1, 1] from the perspective of `side`."""
    att = sum(_stack_value(u) for u in state["units"] if u["side"] == "att")
    dfd = sum(_stack_value(u) for u in state["units"] if u["side"] == "def")
    if side == "def":
        att, dfd = dfd, att
    total = att + dfd
    if total == 0:
        return 0.0
    return (att - dfd) / total


class _Node:
    __slots__ = ("path", "act", "args", "parent", "children", "visits", "value_sum")

    def __init__(self, path: tuple, act: int | None, args: list | None, parent: "_Node | None"):
        self.path = path  # tuple of (act, args) from the battle root
        self.act = act
        self.args = args
        self.parent = parent
        self.children: list[_Node] = []
        self.visits = 0
        self.value_sum = 0.0


class Mcts:
    def __init__(self, env: BattleEnv, c_puct: float = 1.4, rng: random.Random | None = None):
        self.env = env
        self.c_puct = c_puct
        self.rng = rng or random.Random()

    def _replay(self, root_state: dict, path: tuple) -> dict:
        """Resets the engine to the battle root and replays the given action path.

        The whole path is applied inside the engine in one roundtrip (batched "replay" op).
        """
        return self.env.replay(list(path))

    def run(self, root_state: dict, num_simulations: int) -> tuple[list[tuple[int, list[int]]], list[float]]:
        """Runs the search from the given state; returns (legal moves, visit counts)."""
        legal = [(m["act"], tuple(m["args"])) for m in root_state["legal"]]
        if not legal:
            return [], []

        side = root_state["units"][0]["side"] if root_state["units"] else "att"
        # The active unit's side is the side to move.
        cur_uid = root_state["cur"]
        for u in root_state["units"]:
            if u["u"] == cur_uid:
                side = u["side"]
                break

        root = _Node((), None, None, None)

        for _ in range(num_simulations):
            node = root

            # Selection: descend the tree while it has children.
            while node.children:
                total = sum(child.visits for child in node.children)
                best, best_score = None, -1e9
                for child in node.children:
                    prior = 1.0 / len(node.children)
                    exploit = 0.0 if child.visits == 0 else child.value_sum / child.visits
                    explore = self.c_puct * prior * math.sqrt(total) / (1 + child.visits)
                    score = exploit + explore
                    if score > best_score:
                        best, best_score = child, score
                node = best

            # Expansion and evaluation.
            state = self._replay(root_state, node.path)
            if state.get("result"):
                value = 1.0 if state["result"] == side else (-1.0 if state["result"] != "draw" else 0.0)
            else:
                value = evaluate_state(state, side)
                legal = [(m["act"], tuple(m["args"])) for m in state["legal"]]
                # Back up the value, expand children (path from the root).
                for act, args in legal:
                    node.children.append(_Node(node.path + ((act, args),), act, args, node))

            while node is not None:
                node.visits += 1
                node.value_sum += value
                value = -value  # two-player zero-sum perspective flip
                node = node.parent

        counts = [0.0] * len(legal)
        index = {move: i for i, move in enumerate(legal)}
        for child in root.children:
            key = (child.act, child.args)
            if key in index:
                counts[index[key]] = child.visits

        return legal, counts
