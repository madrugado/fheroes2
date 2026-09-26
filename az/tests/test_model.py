"""Unit tests for az/model.py (torch required)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pytest

torch = pytest.importorskip("torch")

import encoding as enc  # noqa: E402
from model import AzBattleNet  # noqa: E402


def test_forward_shapes():
    model = AzBattleNet()
    planes = torch.zeros(2, enc.NUM_PLANES, enc.BOARD_H, enc.BOARD_W)
    scalars = torch.zeros(2, enc.NUM_SCALARS)
    mask = torch.ones(2, enc.ACTION_SPACE, dtype=torch.bool)

    logits, value = model(planes, scalars, mask)
    assert logits.shape == (2, enc.ACTION_SPACE)
    assert value.shape == (2,)
    assert value.abs().max() <= 1.0  # tanh head


def test_masked_policy_excludes_illegal_slots():
    model = AzBattleNet()
    model.eval()

    planes = torch.zeros(1, enc.NUM_PLANES, enc.BOARD_H, enc.BOARD_W)
    scalars = torch.zeros(1, enc.NUM_SCALARS)
    mask = torch.zeros(1, enc.ACTION_SPACE, dtype=torch.bool)
    mask[0, 5] = True   # one MOVE
    mask[0, enc.SKIP_INDEX] = True  # SKIP

    logits, _ = model(planes, scalars, mask)
    probs = torch.softmax(logits, dim=1)

    # All probability mass is on the two legal slots.
    assert abs(probs[0, 5].item() + probs[0, enc.SKIP_INDEX].item() - 1.0) < 1e-5
    # Masked slots get ~zero probability (logits are -1e9 there).
    assert probs[0, 6].item() < 1e-12


def test_value_head_matches_outcome_direction():
    """Sanity: the value head can be trained toward a constant label."""
    model = AzBattleNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    planes = torch.zeros(4, enc.NUM_PLANES, enc.BOARD_H, enc.BOARD_W)
    scalars = torch.zeros(4, enc.NUM_SCALARS)
    targets = torch.ones(4)

    first_loss = None
    for _ in range(200):
        _, value = model(planes, scalars)
        loss = torch.nn.functional.mse_loss(value, targets)
        if first_loss is None:
            first_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    _, value = model(planes, scalars)
    assert value.mean().item() > 0.5, "the value head must move toward the target"
    assert loss.item() < first_loss
