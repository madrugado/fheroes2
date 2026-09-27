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
    # Extend the main line further (with a move legal at THAT point), then reset again.
    extended = env.action(again["legal"][0]["act"], again["legal"][0]["args"])
    assert extended["ev"] == "state"
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


# --- "new" op extensions for real-battle replication (az/battle_agent.py) ---


def unit_cells(state, side):
    return sorted(u["i"] for u in state["units"] if u["side"] == side)


def test_new_places_stacks_by_explicit_army_slot(env):
    """'slot:mon x count' puts the stack into that army slot; board positions derive from it."""
    first = env.new_battle(seed=42, attacker="0:13x10", defender="22x20")
    last = env.new_battle(seed=42, attacker="4:13x10", defender="22x20")

    assert len(first["units"]) == len(last["units"]) == 2
    assert unit_cells(first, "att") != unit_cells(last, "att")

    # The plain format keeps the first-free-slot behavior: '13x10' == '0:13x10'.
    plain = env.new_battle(seed=42, attacker="13x10", defender="22x20")
    assert unit_cells(plain, "att") == unit_cells(first, "att")


def test_new_formation_flags_change_positions(env):
    stacks = "0:13x10,1:21x10,2:22x10"
    spread = env.new_battle(seed=42, attacker=stacks, defender="40x10", spread_att=True)
    grouped = env.new_battle(seed=42, attacker=stacks, defender="40x10", spread_att=False)

    assert unit_cells(spread, "att") != unit_cells(grouped, "att")
    # The defender formation is independent of the attacker flag.
    assert unit_cells(spread, "def") == unit_cells(grouped, "def")


def test_world_seed_is_applied_and_zero_restores_the_pinned_default(env):
    """Obstacles derive from the world seed; 'wseed' 0/absent must mean the pinned default,
    NOT the seed of the previous 'new' (datasets must not depend on the op history)."""
    default = env.new_battle(seed=42, attacker="13x10", defender="22x20", tile=408)["obstacles"]

    other = None
    for world_seed in range(1, 40):
        obstacles = env.new_battle(seed=42, attacker="13x10", defender="22x20", tile=408, world_seed=world_seed)["obstacles"]
        if obstacles != default:
            other = obstacles
            break
    assert other is not None, "no world seed changed the obstacles on the test tile"

    assert env.new_battle(seed=42, attacker="13x10", defender="22x20", tile=408)["obstacles"] == default
    assert env.new_battle(seed=42, attacker="13x10", defender="22x20", tile=408, world_seed=0)["obstacles"] == default


def test_malformed_stack_tokens_are_skipped(env):
    """A garbage token must not crash the engine (std::stoi throws): it is skipped."""
    state = env.new_battle(seed=42, attacker="a:13x10,13x10,zz,5:x", defender="22x20")

    assert state is not None and state["ev"] == "state"
    assert len([u for u in state["units"] if u["side"] == "att"]) == 1


# Wide units (cavalry 8, wolf 15, centaur 30, green dragon 36) and archers (archer 2, elf 24,
# centaur 30): the cases where geometric move candidates used to diverge from the engine's
# command validation.
WIDE_AND_ARCHER_SETUPS = [
    (42, "8x10,2x20,15x10", "30x10,36x2,24x15"),
    (7, "0:15x12,2:30x8,4:2x25", "1:8x10,3:24x20"),
]


@pytest.mark.parametrize("seed,attacker,defender", WIDE_AND_ARCHER_SETUPS)
def test_every_legal_move_is_accepted_by_the_engine(env, seed, attacker, defender):
    """Regression: EnumerateLegalMoves offered moves ApplyAction*() rejects (MOVE to a cell that
    is not the head of a reachable wide-unit position, melee of non-blocked archers, shots of
    blocked archers). Release silently dropped them (the state did not change), Debug asserted.
    Every legal move must change the state when applied from a snapshot of the decision point."""
    import random

    rng = random.Random(seed)
    state = env.new_battle(seed=seed, attacker=attacker, defender=defender)
    checked = 0

    for _ in range(12):
        if state is None or state.get("result") or not state.get("legal"):
            break

        env.snapshot_save(1)
        before = (state["turn"], state["cur"], state["units"])
        for move in state["legal"]:
            after = env.snapshot_restore(1, [(move["act"], tuple(move["args"]))])
            assert after is not None, f"engine died on legal move {move}"
            assert (after["turn"], after["cur"], after["units"]) != before, f"legal move {move} was not applied"
            checked += 1

        env.snapshot_restore(1)  # back to the decision point before extending the main line
        move = rng.choice(state["legal"])
        state = env.action(move["act"], move["args"])

    assert checked > 100


# --- illegal commands are rejected, never silently dropped ---


def illegal_move(state):
    """A MOVE for the unit to move that is not in the legal list (its own head cell)."""
    unit = next(u for u in state["units"] if u["u"] == state["cur"])
    # CommandType::MOVE = 0; values are stored in REVERSE constructor order: [dst, uid].
    move = (0, (unit["i"], state["cur"]))
    assert {"act": move[0], "args": list(move[1])} not in state["legal"]
    return move


def test_illegal_action_is_rejected_and_the_main_line_is_kept(env):
    root = env.new_battle(seed=42, attacker="8x10,2x20", defender="30x10,24x15")
    first = root["legal"][0]
    after_first = env.action(first["act"], first["args"])

    act, args = illegal_move(after_first)
    assert env.action(act, list(args)) == {"ev": "error", "what": "illegal action"}

    # A command for another unit is illegal too.
    other = next(u["u"] for u in after_first["units"] if u["u"] != after_first["cur"])
    assert env.action(8, [other])["ev"] == "error"  # CommandType::SKIP = 8

    # The main line still ends after the first move: the next legal move applies normally and
    # matches a clean main line with the same two moves.
    second = after_first["legal"][0]
    via_errors = env.action(second["act"], second["args"])

    env.reset()
    env.action(first["act"], first["args"])
    clean = env.action(second["act"], second["args"])
    assert via_errors == clean


def test_illegal_command_in_replay_or_restore_path_is_an_error(env):
    root = env.new_battle(seed=42, attacker="8x10,2x20", defender="30x10,24x15")
    legal = (root["legal"][0]["act"], tuple(root["legal"][0]["args"]))
    illegal = illegal_move(root)

    assert env.replay([legal])["ev"] == "state"
    assert env.replay([illegal]) == {"ev": "error", "what": "illegal action"}

    env.snapshot_save(1)
    assert env.snapshot_restore(1, [illegal], save_as=2) == {"ev": "error", "what": "illegal action"}
    # The failed restore must not have stored a snapshot.
    assert env.snapshot_restore(2)["ev"] == "error"
    assert env.snapshot_restore(1, [legal])["ev"] == "state"


@pytest.mark.parametrize("seed,attacker,defender", WIDE_AND_ARCHER_SETUPS + [(42, "13x30,21x25", "22x20,40x10")])
def test_real_legal_moves_map_to_distinct_action_indexes(env, seed, attacker, defender):
    """Regression for the wire-order decoding bug: with the real engine, distinct legal moves
    must get distinct action indexes (46 legal moves used to collapse onto 2 indexes)."""
    import encoding as enc

    state = env.new_battle(seed=seed, attacker=attacker, defender=defender)
    for _ in range(6):
        if state.get("result") or not state.get("legal"):
            break
        cells = enc.unit_cells_map(state["units"])
        slots = [enc.action_index(m["act"], m["args"], cells) for m in state["legal"]]
        assert None not in slots
        assert len(set(slots)) == len(slots), "two legal moves share an action index"
        move = state["legal"][-1]
        state = env.action(move["act"], move["args"])



@pytest.mark.parametrize("seed,attacker,defender", WIDE_AND_ARCHER_SETUPS)
def test_main_line_snapshot_fast_path_matches_the_full_replay(env, seed, attacker, defender):
    """Main-line ops restore the main-line-end snapshot and apply only the new commands; the
    result must equal replaying the whole main line from the battle root ("full":1), at every
    step of a long random game, including search replays on top of the main line."""
    import random

    rng = random.Random(seed)
    state = env.new_battle(seed=seed, attacker=attacker, defender=defender)
    steps = 0
    while state.get("legal") and not state.get("result") and steps < 60:
        move = rng.choice(state["legal"])
        probe = [(move["act"], tuple(move["args"]))]
        assert env.replay(probe) == env.replay(probe, full=True)
        assert env.replay([]) == env.replay([], full=True) == state
        state = env.action(move["act"], move["args"])
        steps += 1
    assert steps >= 20


def test_replica_does_not_write_the_parent_ai_log(tmp_path, monkeypatch):
    """A real game with FHEROES2_AI_LOG spawns this server as its MCTS replica: the replica's
    battle_start/battle_action events must not be appended to the game's log."""
    log_path = tmp_path / "ai.jsonl"
    monkeypatch.setenv("FHEROES2_AI_LOG", str(log_path))

    replica = BattleEnv(binary=BINARY, map_name=MAP_NAME)
    try:
        state = replica.new_battle(seed=42, attacker="13x30,21x25", defender="22x20,40x10")
        for _ in range(5):
            if not state or state.get("result") or not state.get("legal"):
                break
            move = state["legal"][0]
            state = replica.action(move["act"], move["args"])
    finally:
        replica.close()

    assert not log_path.exists() or log_path.stat().st_size == 0


@pytest.mark.parametrize("hero", [
    {"ahid": 5, "ahero": "zz"},          # not hex
    {"ahid": 5, "ahero": "0a0"},         # odd length
    {"ahid": 5, "ahero": "00"},          # truncated serialization
    {"ahid": 9999, "ahero": "0000"},     # no such hero in the world
])
def test_bad_commander_is_an_error_and_the_server_stays_usable(env, hero):
    """A commander that cannot be restored answers an error (no silent commander-less battle);
    the next battle works normally."""
    env._send({"op": "new", "seed": 3, "att": "13x10", "def": "22x10", **hero})
    reply = env._read()
    if hero["ahero"] in ("zz", "0a0"):
        # Undecodable hex means "no commander": the battle is set up with the stacks.
        assert reply["ev"] == "state" and "legal" in reply
    else:
        assert reply == {"ev": "error", "what": "bad battle setup"}

    state = env.new_battle(seed=3, attacker="13x10", defender="22x10")
    assert state["ev"] == "state" and "legal" in state


def test_army_colors_are_accepted(env):
    """Real neutral armies have color 0 (PlayerColor::NONE); the battle must still be valid."""
    state = env.new_battle(seed=3, attacker="13x10", defender="22x10", color_att=4, color_def=0)
    assert state["ev"] == "state" and "legal" in state
    assert {u["side"] for u in state["units"]} == {"att", "def"}
