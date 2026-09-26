"""MCTS for the battle prototype (phase 3: optional neural network guidance).

The search runs against the live engine: to evaluate a node, the battle is reset to its
initial position and the action path from the root is replayed (the engine is deterministic;
the whole path is applied inside the engine with one batched "replay" roundtrip).

With a model (az/model.py, trained by az/train.py):
  - move priors come from the policy head (masked by legality, Dirichlet noise at the root);
  - leaf value comes from the value head (perspective of the side to move).
Without a model: uniform priors + a material-strength heuristic value.
"""

from __future__ import annotations

import math
import random

import encoding as enc

ACT_MOVE = 0
ACT_ATTACK = 1
ACT_SKIP = 8


def _stack_value(unit: dict) -> float:
    """Cheap material proxy: count times a per-monster tier factor."""
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
    __slots__ = ("path", "act", "args", "parent", "children", "visits", "value_sum", "prior")

    def __init__(self, path: tuple, act: int | None, args: tuple | None, parent: "_Node | None", prior: float):
        self.path = path  # tuple of (act, args) from the battle root
        self.act = act
        self.args = args
        self.parent = parent
        self.children: list[_Node] = []
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior


class Mcts:
    def __init__(self, env, model=None, device: str = "cpu", c_puct: float = 1.4,
                 rng: random.Random | None = None, root_noise: float = 0.25, dirichlet_alpha: float = 1.0):
        self.env = env
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.rng = rng or random.Random()
        self.root_noise = root_noise
        self.dirichlet_alpha = dirichlet_alpha
        self._torch = None
        if model is not None:
            import torch

            self._torch = torch

    def _evaluate(self, state: dict) -> tuple[dict[int, float], float]:
        """Evaluates a leaf: returns (slot -> prior for the legal moves, value for the side to move)."""
        legal = [(m["act"], tuple(m["args"])) for m in state["legal"]]
        slots = [enc.action_index(act, list(args)) for act, args in legal]
        slots = [s for s in slots if s is not None]

        mover = enc.side_to_move(state)

        if self.model is None:
            prior = 1.0 / max(len(slots), 1)
            return {s: prior for s in slots}, evaluate_state(state, mover)

        planes = enc.state_planes(state)
        scalars = enc.state_scalars(state)
        torch = self._torch

        with torch.no_grad():
            p_logits, value = self.model(
                torch.tensor([planes], dtype=torch.float32, device=self.device),
                torch.tensor([scalars], dtype=torch.float32, device=self.device),
            )
            mask = torch.zeros(enc.ACTION_SPACE, dtype=torch.bool, device=self.device)
            for s in slots:
                mask[s] = True
            p_logits = p_logits[0].masked_fill(~mask, -1e9)
            probs = torch.softmax(p_logits, dim=0)

        priors = {s: float(probs[s]) for s in slots}
        return priors, float(value[0])

    def _replay(self, path: tuple) -> dict:
        """Resets the engine to the battle root and replays the given action path (one roundtrip)."""
        return self.env.replay(list(path))

    def run(self, root_state: dict, num_simulations: int) -> tuple[list[tuple[int, list[int]]], list[float]]:
        """Runs the search from the given state; returns (legal moves, visit counts)."""
        legal = [(m["act"], tuple(m["args"])) for m in root_state["legal"]]
        if not legal:
            return [], []

        side = enc.side_to_move(root_state)
        root = _Node((), None, None, None, 0.0)

        for sim in range(num_simulations):
            node = root

            # Selection: descend the tree while it has children.
            while node.children:
                total = sum(child.visits for child in node.children)
                best, best_score = None, -1e9
                for child in node.children:
                    exploit = 0.0 if child.visits == 0 else child.value_sum / child.visits
                    explore = self.c_puct * child.prior * math.sqrt(total) / (1 + child.visits)
                    score = exploit + explore
                    if score > best_score:
                        best, best_score = child, score
                node = best

            # Expansion and evaluation.
            state = self._replay(node.path)
            if state.get("result"):
                value = 1.0 if state["result"] == side else (-1.0 if state["result"] != "draw" else 0.0)
            else:
                priors, value = self._evaluate(state)
                leaf_legal = [(m["act"], tuple(m["args"])) for m in state["legal"]]
                for (act, args) in leaf_legal:
                    slot = enc.action_index(act, list(args))
                    prior = priors.get(slot, 1.0 / max(len(leaf_legal), 1))

                    if node is root and self.root_noise > 0:
                        # Dirichlet exploration noise at the root (AlphaZero-style).
                        noise = self.rng.gammavariate(self.dirichlet_alpha, 1.0)
                        prior = (1 - self.root_noise) * prior + self.root_noise * noise

                    node.children.append(_Node(node.path + ((act, args),), act, args, node, prior))

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
