"""Unit tests for rl/game_compare.py (paired real-game comparison of two arms)."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import game_compare  # noqa: E402


def write(path, pairs, map_name="Battlefi.mp2"):
    with open(path, "w") as f:
        json.dump({"summary": {"map": map_name}, "pairs": pairs}, f)


def test_arms_are_paired_by_map_seed_and_color(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    write(a / "r.json", [{"seed": s, "color": c, "hero": 0.1, "duel": 0.0, "str": 100, "k": 0, "outcome": 0}
                         for s in (1, 2, 3) for c in ("Blue", "Red")])
    # Arm B: +0.2 hero on every shared pair; one extra pair (seed 4) has no partner in A.
    write(b / "r.json", [{"seed": s, "color": c, "hero": 0.3, "duel": 0.5, "str": 50, "k": -1, "outcome": 0}
                         for s in (1, 2, 3, 4) for c in ("Blue", "Red")])
    report = game_compare.compare(game_compare.load_arm([str(a)]), game_compare.load_arm([str(b)]))
    assert report["pairs"] == 6
    mean, low, high, n = report["hero"]["b-a"]
    assert n == 6 and abs(mean - 0.2) < 1e-9 and abs(low - 0.2) < 1e-9 and abs(high - 0.2) < 1e-9
    assert abs(report["str"]["b-a"][0] + 50) < 1e-9 and abs(report["k"]["a"][0]) < 1e-9


def test_the_bootstrap_resamples_whole_seeds():
    values = {("m", 1, "Blue"): 1.0, ("m", 1, "Red"): 1.0, ("m", 2, "Blue"): -1.0, ("m", 2, "Red"): -1.0}
    mean, low, high, n = game_compare.cluster_ci(values)
    assert n == 4 and mean == 0.0
    assert low == -1.0 and high == 1.0  # two clusters: every resample is all +1, all -1 or half and half
