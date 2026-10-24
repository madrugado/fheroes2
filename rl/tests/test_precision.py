import copy
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import precision  # noqa: E402
import strategy_net  # noqa: E402
from model import AzBattleNet  # noqa: E402
from selfplay import load_policy_value  # noqa: E402
from transformer_model import AzBattleTransformer, save_checkpoint  # noqa: E402

from .test_strategy_net import ARMY, CONTEXT, TARGET  # noqa: E402
from .test_transformer import history_state, make_state  # noqa: E402


def max_gap(a: dict, b: dict) -> float:
    return max(abs(a[k] - b[k]) for k in a)


def test_the_precision_switch(monkeypatch):
    monkeypatch.setenv("FHEROES2_PRECISION", "fp32")
    assert not precision.use_bf16("cpu", requested=False)
    assert not precision.use_bf16("cpu", requested=True)
    monkeypatch.setenv("FHEROES2_PRECISION", "bf16")
    assert precision.use_bf16("cpu") and not precision.use_bf16("mps")
    monkeypatch.setenv("FHEROES2_PRECISION", "auto")
    assert precision.use_bf16("cpu", requested=True)
    assert precision.use_bf16("cpu") == precision.fast_bf16()


def test_bf16_transformer_evaluate_agrees_with_fp32():
    torch.manual_seed(0)
    fp32 = AzBattleTransformer().eval()
    bf16 = precision.for_inference(copy.deepcopy(fp32), requested=True)
    assert precision.is_bf16(bf16) and not precision.is_bf16(fp32)
    for state in (make_state(), history_state(4)):
        p32, v32 = fp32.evaluate(state)
        p16, v16 = bf16.evaluate(state)
        assert set(p16) == set(p32) and abs(sum(p16.values()) - 1.0) < 1e-5
        assert max_gap(p32, p16) < 0.02 and abs(v32 - v16) < 0.02


def test_the_battle_loader_follows_the_switch(tmp_path, monkeypatch):
    path = tmp_path / "t.pt"
    save_checkpoint(AzBattleTransformer(), str(path))
    monkeypatch.setenv("FHEROES2_PRECISION", "bf16")
    assert precision.is_bf16(load_policy_value(str(path), "transformer", "cpu"))
    monkeypatch.setenv("FHEROES2_PRECISION", "fp32")
    assert not precision.is_bf16(load_policy_value(str(path), "transformer", "cpu"))
    monkeypatch.setenv("FHEROES2_PRECISION", "bf16")
    resnet = tmp_path / "r.pt"
    torch.save(AzBattleNet().state_dict(), str(resnet))
    assert not precision.is_bf16(load_policy_value(str(resnet), "resnet", "cpu").model)  # the ResNet stays fp32


def test_bf16_strategic_policy_agrees_with_fp32_and_stays_deterministic(tmp_path, monkeypatch):
    monkeypatch.setenv("FHEROES2_PRECISION", "auto")
    path = tmp_path / "m.pt"
    torch.manual_seed(0)
    save_checkpoint(AzBattleTransformer(), str(path))
    monkeypatch.setenv("FHEROES2_PRECISION", "fp32")
    fp32 = strategy_net.NetStrategyPolicy(str(path))
    monkeypatch.setenv("FHEROES2_PRECISION", "bf16")
    policies = [strategy_net.NetStrategyPolicy(str(path)), strategy_net.NetStrategyPolicy(str(path)),
                strategy_net.NetStrategyPolicy(str(path), cache=False)]
    assert all(p.bf16 and precision.is_bf16(p.model) for p in policies) and not fp32.bf16
    for day in range(1, 5):
        for policy in (fp32, *policies):
            policy.observe_turn(dict(CONTEXT, t=day))
        for kind, event in (("target", dict(TARGET, t=day)), ("army", dict(ARMY, t=day))):
            ref = fp32.probabilities(kind, event)
            policies[1].prefix_cache.chunks.clear()
            policies[1]._answers.clear()
            warm, cold, plain = (p.probabilities(kind, event) for p in policies)
            assert warm == cold  # bf16 keeps the cached path's determinism: a cold cache gives the same answer
            for probs in (warm, plain):
                assert max(abs(x - y) for x, y in zip(probs, ref)) < 0.02
            for policy in (fp32, *policies):
                policy.record(kind, event, 0)
