"""Integration test for the battle server protocol (requires the ./fheroes2 binary and
game data; skipped automatically when the binary is missing).

Covers the stateless protocol contract: one state reply per operation, legal moves at
decision points, batched replay determinism, and the reset semantics.
"""

import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from engine_bridge import BattleEnv  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BINARY = os.path.join(REPO_ROOT, "fheroes2")
MAP_NAME = "Arena.mp2"

pytestmark = pytest.mark.skipif(not os.path.exists(BINARY), reason="fheroes2 binary not built")


@pytest.fixture()
def env():
    env = BattleEnv(binary=BINARY, map_name=MAP_NAME)
    yield env
    env.close()


def test_new_returns_root_decision_point(env):
    state = env.new_battle(seed=42, attacker="13x30,21x25", defender="22x20,40x10")

    assert state is not None and state["ev"] == "state"
    assert state["turn"] >= 1
    assert state["cur"] != -1
    assert len(state["units"]) == 4
    assert len(state["legal"]) > 0
    assert "result" not in state or state["result"] is None


def test_action_advances_to_next_decision_point(env):
    state = env.new_battle(seed=42, attacker="13x30,21x25", defender="22x20,40x10")
    move = state["legal"][0]

    nxt = env.action(move["act"], move["args"])

    assert nxt is not None and nxt["ev"] == "state"
    assert nxt["cur"] != -1
    assert len(nxt["legal"]) > 0


def test_replay_is_deterministic(env):
    env.new_battle(seed=42, attacker="13x30,21x25", defender="22x20,40x10")

    trace = []
    state = env.reset()
    for _ in range(20):
        if state.get("result"):
            break
        move = state["legal"][0]
        trace.append((move["act"], tuple(move["args"])))
        state = env.action(move["act"], move["args"])

    replay_a = env.replay(trace)
    replay_b = env.replay(trace)

    assert replay_a == replay_b, "identical replays must produce identical states"


def test_replay_matches_per_step_play(env):
    """The batched replay of a path must equal the state reached by playing it step by step."""
    env.new_battle(seed=1234, attacker="13x30,21x25", defender="22x20,40x10")

    trace = []
    state = env.reset()
    for _ in range(12):
        if state.get("result"):
            break
        move = state["legal"][0]
        trace.append((move["act"], tuple(move["args"])))
        state = env.action(move["act"], move["args"])

    if state.get("result"):
        pytest.skip("battle ended within 12 moves, nothing to compare")

    replayed = env.replay(trace)
    assert replayed["units"] == state["units"]
    assert replayed["cur"] == state["cur"]
    assert replayed["turn"] == state["turn"]


def test_reset_returns_to_root(env):
    env.new_battle(seed=42, attacker="13x30,21x25", defender="22x20,40x10")
    state = env.reset()
    move = state["legal"][0]
    env.action(move["act"], move["args"])

    root = env.reset()
    assert root["turn"] >= 1
    assert root["cur"] != -1
    assert "result" not in root or root["result"] is None

    # The main line is gone: the same first move leads to the same state again.
    move = root["legal"][0]
    again = env.action(move["act"], move["args"])
    env.action(root["legal"][0]["act"], root["legal"][0]["args"])
    root2 = env.reset()
    move = root2["legal"][0]
    again2 = env.action(move["act"], move["args"])
    assert again2["units"] == again["units"]


def test_auto_streams_expert_records(env):
    env.new_battle(seed=7, attacker="13x30,21x25", defender="22x20,40x8")

    env._send({"op": "auto"})
    experts = 0
    outcome = None
    for _ in range(5000):
        reply = env._read()
        assert reply is not None, "engine closed the connection"
        if "expert" in reply:
            experts += 1
            assert len(reply["legal"]) > 0
        elif reply.get("result"):
            outcome = reply["result"]
            break
        else:
            break

    assert experts > 0
    assert outcome in ("att", "def", "draw")
