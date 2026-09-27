"""Unit tests for az/train.py record loading and sample building (no engine required)."""

import gzip
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pytest

torch = pytest.importorskip("torch")

import encoding as enc  # noqa: E402
import train  # noqa: E402


def make_record(outcome="att", counts=None, legal=None, units=None, cur=1, turn=3):
    units = units if units is not None else [
        {"u": 1, "side": "att", "mon": 13, "q": 30, "hpl": 10, "i": 0, "ti": -1, "sp": 2, "shots": 8, "moved": 0},
        {"u": 2, "side": "def", "mon": 22, "q": 20, "hpl": 20, "i": 6, "ti": -1, "sp": 2, "shots": 0, "moved": 0},
    ]
    legal = legal if legal is not None else [
        {"act": 0, "args": [1, 1]},
        {"act": 1, "args": [0, -1, -1, 2, 1]},
        {"act": 8, "args": [1]},
    ]
    counts = counts if counts is not None else [2.0, 6.0, 2.0]
    return {"state": {"turn": turn, "cur": cur, "units": units, "obstacles": []},
            "legal": legal, "counts": counts, "outcome": outcome}


def test_load_records_filters(tmp_path):
    good = make_record()
    records = [
        good,
        make_record(outcome="ongoing"),      # unknown outcome
        make_record(counts=[1.0, 2.0]),      # counts/legal length mismatch
        make_record(counts=[0.0, 0.0, 0.0]),  # no visit mass
    ]

    path = tmp_path / "records.jsonl"
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    loaded = train.load_records(str(path))
    assert loaded == [good]


def test_load_records_gzip(tmp_path):
    path = tmp_path / "records.jsonl.gz"
    with gzip.open(path, "wt") as f:
        f.write(json.dumps(make_record()) + "\n")

    loaded = train.load_records(str(path))
    assert len(loaded) == 1


def test_build_resnet_samples():
    samples = train.build_resnet_samples([make_record(outcome="def")])

    assert len(samples) == 1
    planes, scalars, slots, counts, value = samples[0]
    assert len(planes) == enc.NUM_PLANES
    assert len(scalars) == enc.NUM_SCALARS
    assert slots == [1, enc.ATTACK_BASE + 6 * enc.ATTACK_SLOTS + enc.RANGED_DIR, enc.SKIP_INDEX]
    assert abs(sum(counts) - 1.0) < 1e-9
    assert value == -1.0  # outcome "def" from the attacker's perspective


def test_build_resnet_samples_drops_unmappable():
    record = make_record(legal=[{"act": 0, "args": [1, 1]}, {"act": 2, "args": [0]}], counts=[3.0, 1.0])
    samples = train.build_resnet_samples([record])

    assert len(samples) == 1
    _, _, slots, counts, _ = samples[0]
    assert slots == [1]  # a SPELLCAST of Spell::NONE contributes nothing, the rest is renormalized
    assert counts == [1.0]


def test_build_resnet_samples_skips_empty_after_mapping():
    record = make_record(legal=[{"act": 2, "args": [0]}], counts=[1.0])
    assert train.build_resnet_samples([record]) == []


def test_build_transformer_samples():
    # The best move (count 6) is the ranged attack at unit 2 (cell 6).
    record = make_record(counts=[0.0, 6.0, 0.0], outcome="att")
    samples, skipped = train.build_transformer_samples([record])

    assert skipped == 0
    assert len(samples) == 1
    state, target, value = samples[0]
    # The legal options (masked losses): cells 1 (move), 6 (attack), 99 (skip); one direction.
    assert target == {"kind": "attack", "cell": 6, "dir": enc.RANGED_DIR, "cells": [1, 6, 99], "dirs": [enc.RANGED_DIR]}
    assert value == 1.0
    assert state["cur"] == 1


def test_build_transformer_samples_move_and_skip():
    move = make_record(counts=[4.0, 0.0, 0.0])
    skip = make_record(counts=[0.0, 0.0, 1.0], cur=2)
    samples, skipped = train.build_transformer_samples([move, skip])

    assert skipped == 0
    assert samples[0][1] == {"kind": "move", "cell": 1, "dir": None, "cells": [1, 6, 99], "dirs": []}
    assert samples[1][1]["kind"] == "skip" and samples[1][1]["cell"] is None
    assert samples[1][2] == -1.0  # defender to move, attacker won


def test_build_transformer_samples_skips_unmappable():
    record = make_record(legal=[{"act": 2, "args": [0]}], counts=[1.0])
    samples, skipped = train.build_transformer_samples([record])

    assert samples == []
    assert skipped == 1


def test_split_records_keeps_battles_together():
    records = [dict(make_record(legal=[{"act": 8, "args": [1]}], counts=[1.0]), battle=f"b{i % 20}") for i in range(200)]
    fit, val = train.split_records(records, 0.3)
    assert len(fit) + len(val) == 200 and fit and val
    assert not {r["battle"] for r in fit} & {r["battle"] for r in val}
    assert train.split_records(records, 0.0) == (records, [])


def test_imitation_accuracy_counts_exact_and_slot_hits():
    legal = [{"act": 8, "args": [1]}, {"act": 2, "args": [6, 1]}, {"act": 2, "args": [7, 1]}]
    records = [make_record(legal=legal, counts=[0.0, 0.0, 1.0]), make_record(legal=legal, counts=[1.0, 0.0, 0.0])]

    class SpellLover:
        # Prefers the first Fireball target: wrong target (slot hit only) and wrong move.
        def evaluate(self, state):
            return {0: 0.1, 1: 0.5, 2: 0.4}, 0.0

    stats = train.imitation_accuracy(SpellLover(), records)
    assert stats["positions"] == 2
    assert stats["exact"] == 0.0
    assert stats["slot"] == 0.5


def test_warmup_cosine_schedule():
    factor = train.warmup_cosine(1000)
    assert factor(0) < 0.05 and abs(factor(49) - 1.0) < 1e-9  # 5% warmup
    assert factor(500) < 1.0 and abs(factor(999) - 0.1) < 1e-3  # cosine down to the floor


def test_legal_mask_rows():
    mask = train.legal_mask([[1, 3], None, []], 5, "cpu")
    assert mask.tolist() == [[False, True, False, True, False], [True] * 5, [True] * 5]
