"""Unit tests for az/gen_expert.py record conversion (no engine required)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import encoding as enc  # noqa: E402
from gen_expert import convert_records  # noqa: E402


def make_expert_record(expert, legal, cur=1, turn=3):
    units = [
        {"u": 1, "side": "att", "mon": 13, "q": 30, "hpl": 10, "i": 0, "ti": -1, "sp": 2, "shots": 8, "moved": 0},
        {"u": 2, "side": "def", "mon": 22, "q": 20, "hpl": 20, "i": 6, "ti": -1, "sp": 2, "shots": 0, "moved": 0},
    ]
    return {"turn": turn, "cur": cur, "units": units, "obstacles": [], "legal": legal, "expert": expert}


def test_convert_records_maps_expert_action():
    legal = [
        {"act": 0, "args": [1, 1]},
        {"act": 1, "args": [0, -1, -1, 2, 1]},  # ranged shot at unit 2
        {"act": 8, "args": [1]},
    ]
    records = convert_records([make_expert_record({"act": 1, "args": [0, -1, -1, 2, 1]}, legal)], "att")

    assert len(records) == 1
    record = records[0]
    assert record["counts"] == [0.0, 1.0, 0.0]
    assert record["outcome"] == "att"
    assert set(record["state"]) == {"turn", "units", "obstacles", "cur"}
    assert record["state"]["cur"] == 1
    assert record["legal"] == legal
    assert enc.action_index(1, [0, -1, -1, 2, 1], {2: 6}) == enc.ATTACK_BASE + 6 * 7 + enc.RANGED_DIR


def test_convert_records_skips_out_of_space_and_unmatched():
    legal = [{"act": 0, "args": [1, 1]}, {"act": 8, "args": [1]}]
    # SPELLCAST is outside the fixed action space entirely.
    out_of_space = make_expert_record({"act": 2, "args": [1]}, legal)
    # MOVE to cell 42 is mappable but absent from the legal list (v0 enumeration gap).
    unmatched = make_expert_record({"act": 0, "args": [42, 1]}, legal)

    assert convert_records([out_of_space, unmatched], "def") == []
