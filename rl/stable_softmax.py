"""Log-softmax and cross-entropy that stay accurate for near-certain predictions.

torch's log_softmax computes z - max - log(sum exp(z - max)). At the argmax that is
-log(1 + s) with s = sum of the other exp(z_j - max), and 1 + s rounds to 1 once s falls
below half an ulp (fp32 ~6e-8, bf16 ~4e-3): the log-probability of a confident answer
comes out as 0 or with a large relative error, and the gradient no longer sums to zero.
Here the argmax is excluded from the sum and log p(argmax) = -softplus(logsumexp(others)),
accurate down to exp underflow; the other entries are shifted by the same exact term.

Everything is computed in float32: under CPU bf16 autocast torch keeps log_softmax in
bf16 (only cross_entropy is upcast), which quantizes log-probabilities near -17 to steps
of 0.125 — the scale of DPO/GRPO log-ratios.
"""

import torch
import torch.nn.functional as F


def _excluding(shifted: torch.Tensor, excluded: torch.Tensor, dim: int) -> torch.Tensor:
    """logsumexp over `dim` without the excluded entry. A finite sentinel instead of -inf: a row
    with nothing else left gets a huge negative value (softplus -> 0) and finite gradients."""
    return torch.logsumexp(shifted.masked_fill(excluded, torch.finfo(shifted.dtype).min), dim=dim, keepdim=True)


def log_softmax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    z = z.float()
    i_max = z.argmax(dim=dim, keepdim=True)
    is_max = torch.zeros_like(z, dtype=torch.bool).scatter(dim, i_max, True)
    shifted = (z - z.gather(dim, i_max)).masked_fill(is_max, 0.0)
    return shifted - F.softplus(_excluding(shifted, is_max, dim))


def softmax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return log_softmax(z, dim).exp()


def cross_entropy(z: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """-log softmax(z)[target] over dim 1 of (N, C) logits: softplus of minus the target's margin
    over the other logits."""
    z = z.float()
    index = target.view(-1, 1)
    is_target = torch.zeros_like(z, dtype=torch.bool).scatter(1, index, True)
    shifted = z - z.gather(1, index)
    loss = F.softplus(_excluding(shifted, is_target, 1)).squeeze(1)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss
