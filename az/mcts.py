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
    __slots__ = ("path", "act", "args", "parent", "children", "visits", "value_sum", "prior", "snap_id")

    def __init__(self, path: tuple, act: int | None, args: tuple | None, parent: "_Node | None", prior: float):
        self.path = path  # tuple of (act, args) from the battle root
        self.act = act
        self.args = args
        self.parent = parent
        self.children: list[_Node] = []
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior
        self.snap_id = 0  # battle-server snapshot of the materialized state (0 = none)


class Mcts:
    def __init__(self, env, policy_value=None, c_puct: float = 1.4,
                 rng: random.Random | None = None, root_noise: float = 0.25, dirichlet_alpha: float = 1.0,
                 use_snapshots: bool = True):
        """policy_value: optional object with .evaluate(state) -> (priors by legal move
        index, value in [-1, 1]). Without it the search uses uniform priors and the
        material-strength heuristic value.

        With an engine that supports battle-server snapshots (snapshot_save/snapshot_restore,
        i.e. the live BattleEnv) the state of each visited node is materialized by restoring
        the parent snapshot and applying one action — instead of replaying the whole path from
        the battle root per simulation. The caller must keep the engine paused exactly at
        root_state for the duration of run()."""

        self.env = env
        self.policy_value = policy_value
        self.c_puct = c_puct
        self.rng = rng or random.Random()
        self.root_noise = root_noise
        self.dirichlet_alpha = dirichlet_alpha
        self._use_snapshots = use_snapshots and hasattr(env, "snapshot_restore") and hasattr(env, "snapshot_save")
        self._snap_counter = 0

    def _next_snap_id(self) -> int:
        self._snap_counter += 1
        return self._snap_counter

    def _evaluate(self, state: dict) -> tuple[dict[int, float], float]:
        """Evaluates a leaf: returns (prior per legal move index, value for the side to move)."""
        if self.policy_value is None:
            legal_count = len(state["legal"])
            mover = enc.side_to_move(state)
            prior = 1.0 / max(legal_count, 1)
            return {i: prior for i in range(legal_count)}, evaluate_state(state, mover)

        return self.policy_value.evaluate(state)

    def _materialize(self, node: "_Node") -> dict:
        """Returns the engine state for the given node, materializing its snapshot on the way."""
        if not self._use_snapshots:
            return self.env.replay(list(node.path))

        snap_id = self._next_snap_id()
        state = self.env.snapshot_restore(node.parent.snap_id, path=[(node.act, list(node.args))], save_as=snap_id)
        node.snap_id = snap_id
        return state

    def run(self, root_state: dict, num_simulations: int) -> tuple[list[tuple[int, list[int]]], list[float]]:
        """Runs the search from the given state; returns (legal moves, visit counts)."""
        legal = [(m["act"], tuple(m["args"])) for m in root_state["legal"]]
        if not legal:
            return [], []

        side = enc.side_to_move(root_state)
        root = _Node((), None, None, None, 0.0)

        if self._use_snapshots:
            # Canonicalize the engine position first: a previous replay-based search may have
            # left the engine paused at an arbitrary search node. The search root must be the
            # main-line end (the same contract the replay-based materialization relies on).
            self.env.replay([])
            root.snap_id = self._next_snap_id()
            self.env.snapshot_save(root.snap_id)

        try:
            for _sim in range(num_simulations):
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
                state = root_state if node is root else self._materialize(node)
                if state.get("result"):
                    value = 1.0 if state["result"] == side else (-1.0 if state["result"] != "draw" else 0.0)
                else:
                    priors, value = self._evaluate(state)
                    leaf_legal = [(m["act"], tuple(m["args"])) for m in state["legal"]]
                    uniform = 1.0 / max(len(leaf_legal), 1)
                    for i, (act, args) in enumerate(leaf_legal):
                        prior = priors.get(i, uniform)

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
        finally:
            if self._use_snapshots:
                self.env.snapshots_free()

        counts = [0.0] * len(legal)
        index = {move: i for i, move in enumerate(legal)}
        for child in root.children:
            key = (child.act, child.args)
            if key in index:
                counts[index[key]] = child.visits
        return legal, counts
