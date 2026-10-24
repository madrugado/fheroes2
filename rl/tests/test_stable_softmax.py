import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import stable_softmax as ss


def reference(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.log_softmax(z.double(), dim=dim)


def confident_logits() -> torch.Tensor:
    # argmax at 0 with others far below: torch fp32 rounds 1 + sum(exp) and loses log p(argmax)
    return torch.tensor([[0.0, -17.0, -17.0, -17.0, -17.0, -17.0],
                         [3.0, -20.0, -25.0, -30.0, -40.0, -60.0],
                         [1.0, 0.5, -0.5, -2.0, -4.0, -8.0]])


def test_log_softmax_matches_float64_on_random_logits():
    torch.manual_seed(0)
    z = torch.randn(64, 37) * 5
    assert torch.allclose(ss.log_softmax(z, 1).double(), reference(z, 1), rtol=1e-6, atol=1e-5)


def test_log_prob_of_a_confident_argmax_keeps_its_relative_precision():
    z = confident_logits()
    exact = reference(z, 1)[:, 0]
    ours = ss.log_softmax(z, 1)[:, 0].double()
    assert torch.allclose(ours, exact, rtol=1e-5, atol=0)
    # torch's own fp32 result is off by tens of percent here (the reason for this module)
    assert not torch.allclose(F.log_softmax(z, 1)[:, 0].double(), exact, rtol=1e-2, atol=0)


def test_gradients_sum_to_zero_like_the_exact_softmax():
    z = confident_logits().requires_grad_()
    (-ss.log_softmax(z, 1)[:, 0]).sum().backward()
    exact = (torch.softmax(z.detach().double(), 1) - F.one_hot(torch.zeros(3, dtype=torch.long), 6)).float()
    assert torch.allclose(z.grad, exact, rtol=1e-5, atol=1e-30)
    scale = z.grad.double().abs().amax(1)
    assert (z.grad.double().sum(1).abs() <= 1e-6 * scale).all()
    # torch: on the first row the sum is ~70% of the largest gradient
    z.grad = None
    (-F.log_softmax(z, 1)[0, 0]).backward()
    assert z.grad[0].double().sum().abs() > 0.1 * z.grad[0].double().abs().max()


@pytest.mark.parametrize("target", [0, 1, 5])
def test_cross_entropy_matches_float64_and_grads(target):
    z = confident_logits().requires_grad_()
    t = torch.full((3,), target, dtype=torch.long)
    loss = ss.cross_entropy(z, t, reduction="none")
    exact = -reference(z.detach(), 1)[:, target]
    assert torch.allclose(loss.double(), exact, rtol=1e-5, atol=0)
    loss.sum().backward()
    exact_grad = torch.softmax(z.detach().double(), 1) - F.one_hot(t, 6).double()
    assert torch.allclose(z.grad.double(), exact_grad, rtol=1e-5, atol=1e-30)


def test_cross_entropy_reductions_match_torch_on_ordinary_logits():
    torch.manual_seed(1)
    z, t = torch.randn(16, 9), torch.randint(0, 9, (16,))
    assert torch.allclose(ss.cross_entropy(z, t), F.cross_entropy(z, t), atol=1e-6)
    assert torch.allclose(ss.cross_entropy(z, t, reduction="sum"), F.cross_entropy(z, t, reduction="sum"), atol=1e-5)


def test_masked_entries_ties_and_single_options_stay_finite():
    z = torch.tensor([[2.0, -1e9, -1e9], [1.0, 1.0, -1e9], [-1e9, 0.5, -1e9]], requires_grad=True)
    lp = ss.log_softmax(z, 1)
    assert lp[0, 0].item() == 0.0 and lp[2, 1].item() == 0.0
    assert torch.allclose(lp[1, :2], torch.full((2,), -0.6931472))
    assert (lp[:, :][z.detach() < -1e8] < -1e8).all()
    (lp[0, 0] + lp[1, 0] + lp[2, 1] + ss.cross_entropy(z, torch.tensor([0, 1, 1]))).backward()
    assert torch.isfinite(z.grad).all()
    one = torch.tensor([[3.0]], requires_grad=True)
    ss.log_softmax(one, 1).sum().backward()
    assert ss.log_softmax(one.detach(), 1).item() == 0.0 and one.grad.item() == 0.0


def test_bf16_input_is_computed_in_float32():
    z = torch.tensor([[0.0, -17.0, -16.875, -12.0]]).bfloat16()
    out = ss.log_softmax(z, 1)
    assert out.dtype == torch.float32
    assert torch.allclose(out.double(), reference(z.float(), 1), rtol=1e-6, atol=0)
    assert ss.softmax(z, 1).dtype == torch.float32


def test_dim_zero_and_softmax_sum_to_one():
    torch.manual_seed(2)
    z = torch.randn(7) * 30
    assert torch.allclose(ss.log_softmax(z, 0).double(), reference(z, 0), rtol=1e-6, atol=1e-4)
    assert abs(ss.softmax(z, 0).double().sum().item() - 1.0) < 1e-6
