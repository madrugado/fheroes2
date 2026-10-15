"""Unit tests for rl/battle_grpo.py helpers and the ResNet share head / MCTS score mode they feed."""

import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from battle_grpo import group_advantages, score_parts, value_targets  # noqa: E402
from mcts import Mcts, score_value  # noqa: E402
from model import AzBattleNet  # noqa: E402
from transformer_model import grow_rows  # noqa: E402


def units(att_str, def_str):
    return [{"u": 1, "side": "att", "q": 1, "str": att_str}, {"u": 2, "side": "def", "q": 1, "str": def_str}]


def test_group_advantages_center_and_skip_flat_groups():
    adv = group_advantages({"scores": [[1.0, 1.0], [0.0, 0.0]]})
    assert abs(sum(adv)) < 1e-9 and adv[0] > 0 > adv[1]
    assert group_advantages({"scores": [[0.5], [0.5]]}) is None
    assert group_advantages({"scores": [[0.5]]}) is None


def test_score_parts_are_the_parts_of_move_score():
    root = {"units": units(100, 50)}
    final = {"ev": "state", "result": "att", "units": units(40, 0)}
    assert score_parts(root, final, "att") == [1.0, 0.4, 0.0]
    assert score_parts(root, final, "def") == [-1.0, 0.0, 0.4]
    assert score_parts(root, None, "att") is None


def test_value_targets_use_the_builtin_moves_rollouts():
    record = {"options": [3, 5], "expert": 5, "parts": [[[1, 1, 0]], [[1, 0.5, 0], [-1, 0, 0.5], None]]}
    assert value_targets(record) == [0.0, 0.25, 0.25]
    assert value_targets(dict(record, expert=7)) is None
    assert value_targets({"options": [1], "expert": 1}) is None


def test_resnet_share_head_and_loading_an_older_checkpoint():
    model = AzBattleNet()
    old = {k: v for k, v in model.state_dict().items() if not k.startswith("share_")}
    fresh = AzBattleNet()
    fresh.load_state_dict(grow_rows(fresh.state_dict(), old))  # strict: the missing share head keeps its init
    import encoding as enc
    planes = torch.zeros(2, enc.NUM_PLANES, 9, 11)
    scalars = torch.zeros(2, enc.NUM_SCALARS)
    fresh.eval()
    _, value, shares = fresh(planes, scalars, shares=True)
    assert value.shape == (2,) and shares.shape == (2, 2)
    assert ((shares > 0) & (shares < 1)).all()


class ScoreEnv:
    """The attacker moves twice: MOVE then ATTACK wins keeping all its strength; SKIP then SKIP wins too but loses
    most of the attacker's strength. Only the score (outcome + strength) tells the lines apart."""

    def replay(self, path):
        path = list(path)
        state = {"turn": 1, "obstacles": [], "cur": 1, "units": units(100, 100)}
        if any(act == 1 for act, _ in path):
            return dict(state, legal=[], result="att", units=units(100, 0))
        if len(path) >= 2:
            return dict(state, legal=[], result="att", units=units(10, 0))
        if not path:
            return dict(state, legal=[{"act": 0, "args": [1, 1]}, {"act": 8, "args": [1]}])
        if path[0][0] == 0:
            return dict(state, legal=[{"act": 1, "args": [1, 2, -1, 6, 4]}])
        return dict(state, legal=[{"act": 8, "args": [1]}])


class FlatNet:
    def evaluate_parts(self, state):
        n = len(state["legal"])
        return {i: 1.0 / n for i in range(n)}, 0.0, [1.0, 1.0]


def test_mcts_score_mode_prefers_the_cheaper_win():
    env = ScoreEnv()
    mcts = Mcts(env, policy_value=FlatNet(), rng=random.Random(1), root_noise=0.0, score_mode=True)
    legal, counts = mcts.run(env.replay(()), num_simulations=20)
    move_index = next(i for i, (act, _) in enumerate(legal) if act == 0)
    assert counts[move_index] == max(counts)
    assert score_value(1.0, 1.0, 0.0) == 1.0 and score_value(-1.0, 0.0, 1.0) == -1.0
