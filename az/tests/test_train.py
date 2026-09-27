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
    assert slots == [1, enc.ATTACK_BASE + 6 * 7 + enc.RANGED_DIR, enc.SKIP_INDEX]
    assert abs(sum(counts) - 1.0) < 1e-9
    assert value == -1.0  # outcome "def" from the attacker's perspective


def test_build_resnet_samples_drops_unmappable():
    record = make_record(legal=[{"act": 0, "args": [1, 1]}, {"act": 2, "args": [1]}], counts=[3.0, 1.0])
    samples = train.build_resnet_samples([record])

    assert len(samples) == 1
    _, _, slots, counts, _ = samples[0]
    assert slots == [1]  # SPELLCAST contributes nothing, the rest is renormalized
    assert counts == [1.0]


def test_build_resnet_samples_skips_empty_after_mapping():
    record = make_record(legal=[{"act": 2, "args": [1]}], counts=[1.0])
    assert train.build_resnet_samples([record]) == []


def test_build_transformer_samples():
    # The best move (count 6) is the ranged attack at unit 2 (cell 6).
    record = make_record(counts=[0.0, 6.0, 0.0], outcome="att")
    samples, skipped = train.build_transformer_samples([record])

    assert skipped == 0
    assert len(samples) == 1
    state, target, value = samples[0]
    assert target == {"kind": "attack", "cell": 6, "dir": enc.RANGED_DIR}
    assert value == 1.0
    assert state["cur"] == 1


def test_build_transformer_samples_move_and_skip():
    move = make_record(counts=[4.0, 0.0, 0.0])
    skip = make_record(counts=[0.0, 0.0, 1.0], cur=2)
    samples, skipped = train.build_transformer_samples([move, skip])

    assert skipped == 0
    assert samples[0][1] == {"kind": "move", "cell": 1, "dir": None}
    assert samples[1][1] == {"kind": "skip", "cell": None, "dir": None}
    assert samples[1][2] == -1.0  # defender to move, attacker won


def test_build_transformer_samples_skips_unmappable():
    record = make_record(legal=[{"act": 2, "args": [1]}], counts=[1.0])
    samples, skipped = train.build_transformer_samples([record])

    assert samples == []
    assert skipped == 1
