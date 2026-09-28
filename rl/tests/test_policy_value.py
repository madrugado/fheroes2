"""Unit tests for rl/policy_value.py (torch required)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pytest

torch = pytest.importorskip("torch")

from model import AzBattleNet  # noqa: E402
from policy_value import ResNetPolicyValue  # noqa: E402


def make_state():
    return {
        "turn": 3,
        "cur": 1,
        "obstacles": [40],
        "units": [
            {"u": 1, "side": "att", "mon": 13, "q": 30, "hpl": 10, "i": 0, "ti": -1, "sp": 2, "shots": 8, "moved": 0},
            {"u": 2, "side": "def", "mon": 22, "q": 20, "hpl": 20, "i": 6, "ti": -1, "sp": 2, "shots": 0, "moved": 0},
        ],
        "legal": [
            {"act": 0, "args": [1, 1]},             # MOVE -> slot 1
            {"act": 1, "args": [1, 2, -1, -1, 0]},  # ranged ATTACK at unit 2 (cell 6)
            {"act": 2, "args": [0]},                # SPELLCAST without a valid spell: unmappable
            {"act": 8, "args": [1]},                # SKIP
            {"act": 2, "args": [6, 1]},             # Fireball on cell 6 ...
            {"act": 2, "args": [7, 1]},             # ... and on cell 7: one shared spell slot
        ],
    }


def test_resnet_policy_value_interface():
    model = AzBattleNet()
    model.eval()
    pv = ResNetPolicyValue(model)

    priors, value = pv.evaluate(make_state())

    assert set(priors) == {0, 1, 2, 3, 4, 5}
    assert priors[2] == 0.0  # unmappable move gets no mass
    assert priors[4] > 0.0 and abs(priors[4] - priors[5]) < 1e-9  # the targets split the slot
    assert all(p >= 0 for p in priors.values())
    assert abs(sum(priors.values()) - 1.0) < 1e-5
    assert -1.0 <= value <= 1.0
