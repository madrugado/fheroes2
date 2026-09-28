"""Small AlphaZero-style ResNet for battles: policy (fixed action space: encoding.ACTION_SPACE,
1460 slots incl. hero spells, masked by legality) and value (tanh scalar). ~2.7M parameters
(mostly the policy layer), runs comfortably on CPU/MPS for prototype-scale self-play."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import ACTION_SPACE, NUM_PLANES, NUM_SCALARS

FILTERS = 64
BLOCKS = 4


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(x + out)


class AzBattleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Conv2d(NUM_PLANES, FILTERS, 3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(FILTERS)
        self.blocks = nn.Sequential(*[ResBlock(FILTERS) for _ in range(BLOCKS)])

        self.policy_conv = nn.Conv2d(FILTERS, 16, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(16)
        self.policy_fc = nn.Linear(16 * 9 * 11, ACTION_SPACE)

        self.value_conv = nn.Conv2d(FILTERS, 8, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * 9 * 11 + NUM_SCALARS, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, planes: torch.Tensor, scalars: torch.Tensor, mask: torch.Tensor | None = None):
        """planes: (B, NUM_PLANES, 9, 11); scalars: (B, NUM_SCALARS);
        mask: (B, ACTION_SPACE) boolean, True = legal. Returns (policy_logits, value)."""
        x = F.relu(self.bn_input(self.input(planes)))
        x = self.blocks(x)

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = self.policy_fc(p.flatten(1))
        if mask is not None:
            p = p.masked_fill(~mask, -1e9)

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = torch.cat([v.flatten(1), scalars], dim=1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v.squeeze(1)
