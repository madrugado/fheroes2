"""Adapters exposing the unified policy/value interface for MCTS:

    evaluate(state) -> (priors keyed by legal move index, value in [-1, 1])
"""

from __future__ import annotations

import torch

import encoding as enc


class ResNetPolicyValue:
    """Wraps AzBattleNet (fixed 793-slot action space)."""

    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device

    @torch.no_grad()
    def evaluate(self, state: dict):
        legal = state["legal"]
        unit_cells = enc.unit_cells_map(state["units"])

        slots = []
        for move in legal:
            act, args = (move["act"], move["args"]) if isinstance(move, dict) else (move[0], move[1])
            slots.append(enc.action_index(act, list(args), unit_cells))

        planes = enc.state_planes(state)
        scalars = enc.state_scalars(state)

        logits, value = self.model(
            torch.tensor([planes], dtype=torch.float32, device=self.device),
            torch.tensor([scalars], dtype=torch.float32, device=self.device),
        )

        mask = torch.zeros(enc.ACTION_SPACE, dtype=torch.bool, device=self.device)
        for slot in slots:
            if slot is not None:
                mask[slot] = True
        logits = logits[0].masked_fill(~mask, -1e9)
        probs = torch.softmax(logits, dim=0)

        priors = {}
        for i, slot in enumerate(slots):
            priors[i] = float(probs[slot]) if slot is not None else 0.0

        return priors, float(value[0])
