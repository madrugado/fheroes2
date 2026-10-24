"""Inference precision of the transformer: bf16 WEIGHTS on CPUs with native bf16 (avx512_bf16 / amx_bf16), fp32 elsewhere.
The battle ResNet stays fp32 (trained in fp32; bf16 is no faster for it).

Measured on the training server (EPYC 9654, 1 thread, 200m body, 2026-10-24): inference of a 40-token
query fp32 35 ms, bf16 autocast 42 ms (it re-casts every weight on each call), bf16 weights 17 ms;
the real 200m strategic net over 176 game queries: fp32 233 ms, autocast 176 ms, bf16 weights 116 ms, argmax
unchanged in all of them (probability gap <= 0.04);
fp16 and fp8 are slower than fp32 there (no avx512_fp16; fp8 _scaled_mm has no fast CPU kernel nor
a backward). Training keeps fp32 master weights under autocast: a bf16 weight cannot absorb an
update of lr x grad (8 mantissa bits).

FHEROES2_PRECISION=fp32|bf16 overrides the CPU check; `requested=True` (the --bf16 flags) forces bf16.
"""

import contextlib
import functools
import os

import torch


@functools.lru_cache(maxsize=1)
def fast_bf16() -> bool:
    try:
        with open("/proc/cpuinfo") as f:
            flags = f.read()
    except OSError:
        return False
    return " avx512_bf16" in flags or " amx_bf16" in flags


def use_bf16(device: str = "cpu", requested: bool = False) -> bool:
    if str(device) != "cpu":
        return False
    mode = os.environ.get("FHEROES2_PRECISION", "auto")
    if mode == "fp32":
        return False
    return requested or mode == "bf16" or fast_bf16()


def for_inference(model: torch.nn.Module, device: str = "cpu", requested: bool = False) -> torch.nn.Module:
    """The model in eval mode, its weights in bf16 when use_bf16()."""
    model.eval()
    if use_bf16(device, requested):
        model.to(torch.bfloat16)
    return model


def is_bf16(model: torch.nn.Module) -> bool:
    return next(model.parameters()).dtype == torch.bfloat16


def autocast_for(model: torch.nn.Module):
    """Autocast for a bf16-weight model: casts the fp32 inputs (token features, planes); the weight
    casts are no-ops. A null context for fp32 models."""
    if is_bf16(model):
        return torch.autocast("cpu", dtype=torch.bfloat16)
    return contextlib.nullcontext()
