"""Unit tests for az/selfplay.py game driver with a scripted fake engine (no engine required)."""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import selfplay  # noqa: E402
from selfplay import play_one, random_army, state_hash, verify_determinism  # noqa: E402

DECISION = {
    "turn": 1,
    "cur": 1,
    "obstacles": [],
    "units": [
        {"u": 1, "side": "att", "mon": 1, "q": 10, "hpl": 5, "i": 0, "ti": -1, "sp": 3, "shots": 0, "moved": 0},
        {"u": 2, "side": "def", "mon": 2, "q": 10, "hpl": 5, "i": 6, "ti": -1, "sp": 3, "shots": 0, "moved": 0},
    ],
    "legal": [
        {"act": 0, "args": [1, 1]},
        {"act": 1, "args": [1, 2, -1, 6, 4]},
        {"act": 8, "args": [1]},
    ],
}


class ScriptedEnv:
    """Fake battle engine: the first action ends the battle with the attacker winning."""

    def __init__(self, terminate=True):
        self.terminate = terminate
        self.actions = 0

    def new_battle(self, seed, attacker, defender):
        self.actions = 0
        return dict(DECISION)

    def action(self, act, args):
        self.actions += 1
        if self.terminate:
            terminal = dict(DECISION)
            terminal["legal"] = []
            terminal["result"] = "att"
            return terminal
        return dict(DECISION)

    def close(self):
        pass


class FirstMoveMcts:
    """Stub search: uniform counts over the legal moves."""

    def run(self, state, sims):
        return state["legal"], [1.0] * len(state["legal"])


def test_play_one_records_outcome():
    env = ScriptedEnv()

    state, records, trace = play_one(env, sims=4, seed=7, attacker="1x10", defender="2x5",
                                     rng=random.Random(0), mcts_factory=FirstMoveMcts)

    assert state["result"] == "att"
    assert len(records) == 1 and len(trace) == 1
    record = records[0]
    assert record["outcome"] == "att"
    assert record["counts"] == [1.0, 1.0, 1.0]
    assert set(record) == {"state_hash", "state", "legal", "counts", "outcome"}
    assert len(record["state_hash"]) == 12


def test_play_one_without_outcome_drops_records():
    env = ScriptedEnv(terminate=False)  # never terminates: the move cap kicks in

    _, records, _ = play_one(env, sims=4, seed=7, attacker="1x10", defender="2x5",
                             rng=random.Random(0), mcts_factory=FirstMoveMcts)

    assert records == []


def test_verify_determinism_replays_the_trace():
    env = ScriptedEnv()
    _, _, trace = play_one(env, sims=4, seed=7, attacker="1x10", defender="2x5",
                           rng=random.Random(0), mcts_factory=FirstMoveMcts)

    assert verify_determinism(env, 7, "1x10", "2x5", trace)


def test_random_army_shape():
    army = random_army(random.Random(1))
    pool = {mon for mon, _, _ in selfplay.MONSTER_POOL}
    parts = army.split(",")

    assert 3 <= len(parts) <= 5
    for part in parts:
        mon, count = part.split("x")
        assert int(mon) in pool
        assert int(count) > 0


def test_state_hash_stability():
    assert state_hash(DECISION) == state_hash(dict(DECISION))
    assert state_hash(DECISION) != state_hash({**DECISION, "turn": 2})
    # unit order must not matter
    shuffled = dict(DECISION, units=list(reversed(DECISION["units"])))
    assert state_hash(DECISION) == state_hash(shuffled)
