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

    # A clean main line, so the replay applies exactly the trace (the search replay semantics:
    # the batched path is applied on top of the current main line).
    env.reset()
    replay_a = env.replay(trace)
    replay_b = env.replay(trace)

    assert replay_a == replay_b, "identical replays must produce identical states"
    assert replay_a["turn"] == state["turn"]


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

    # Clear the main line so the replay applies exactly the walked path.
    env.reset()
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


def test_snapshot_restore_matches_replay(env):
    """Snapshot restore must reproduce exactly the state a full replay from the battle root
    produces, for every prefix point of a random action path."""
    import random

    env.new_battle(seed=42, attacker="13x30,21x25", defender="22x20,40x10")
    state = env.reset()
    rng = random.Random(1234)

    path = []
    finals = [state]  # state after each prefix of the path

    # Snapshot 100+k must capture the state after k actions: taken while the engine is
    # paused exactly at that point (snapshot ids map 1:1 onto finals indexes).
    env.snapshot_save(100)

    for step in range(12):
        if state.get("result") or not state.get("legal"):
            break
        move = rng.choice(state["legal"])
        path.append((move["act"], list(move["args"])))
        state = env.action(move["act"], move["args"])
        finals.append(state)
        env.snapshot_save(101 + step)

    assert len(path) >= 2, "the test needs a battle that survives at least two actions"
    final = finals[-1]

    # Restoring a snapshot must return the exact state stored in it.
    for step in range(len(path) + 1):
        restored = env.snapshot_restore(100 + step)
        assert restored == finals[step], f"restore to step {step} must match the walked state"

    # Restore + suffix must continue identically to the full main line.
    for step in range(len(path)):
        restored = env.snapshot_restore(100 + step, path=path[step:])
        assert restored == final, f"restore at step {step} + suffix must reach the final state"

    # Snapshots survive restores (no arena rebuild between them).
    assert env.snapshot_restore(100) == finals[0]

    # Full replay from the battle root (clean main line + the whole path) must agree with the
    # walked final state. Snapshots (owned by the battle server) survive the arena rebuild.
    env.reset()
    replayed = env.replay(path)
    assert replayed == final
    assert env.snapshot_restore(100) == finals[0]

    # Explicit free must make snapshots unresolvable; the reply carries the current state,
    # which is the root state (snapshot 100 was restored just above).
    env.snapshot_save(7)
    reply = env.snapshots_free()
    assert reply is not None and reply["ev"] == "state" and reply == finals[0]
    assert env.snapshot_restore(7) == {"ev": "error", "what": "unknown snapshot id"}

    # A new battle (different setup) invalidates any previously stored snapshots.
    env.snapshot_save(9)
    env.new_battle(seed=43, attacker="13x30,21x25", defender="22x20,40x10")
    assert env.snapshot_restore(9) == {"ev": "error", "what": "unknown snapshot id"}


def test_suggest_returns_builtin_action_without_applying(env):
    import encoding as enc

    env.new_battle(seed=42, attacker="13x30,21x25", defender="22x20,40x10")
    state = env.reset()

    suggested = env.suggest()

    assert suggested is not None and suggested["ev"] == "state"
    # The pre-decision state must be unchanged: suggest does not apply anything.
    assert suggested["cur"] == state["cur"]
    assert suggested["units"] == state["units"]
    assert suggested["legal"] == state["legal"]

    expert = suggested["expert"]
    assert expert is not None
    unit_cells = {u["u"]: u["i"] for u in state["units"]}
    slot = enc.action_index(expert["act"], list(expert["args"]), unit_cells)
    assert slot is not None, "the built-in action must be inside the fixed action space"

    # The suggested slot must correspond to one of the enumerated legal moves.
    legal_slots = set()
    for move in state["legal"]:
        mapped = enc.action_index(move["act"], list(move["args"]), unit_cells)
        if mapped is not None:
            legal_slots.add(mapped)
    assert slot in legal_slots, "the built-in action must be present in the legal move list"

    # And applying it through the normal action op advances the battle.
    nxt = env.action(expert["act"], expert["args"])
    assert nxt is not None and nxt["ev"] == "state"
