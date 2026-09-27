"""Unit tests for az/encoding.py (no engine required)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import encoding as enc


def make_state(units, obstacles=None, cur=1, turn=5):
    return {"turn": turn, "units": units, "obstacles": obstacles or [], "cur": cur}


def test_action_index_move():
    assert enc.action_index(0, [42, 7]) == 42
    assert enc.action_index(0, [0, 7]) == 0
    assert enc.action_index(0, [98, 7]) == 98
    assert enc.action_index(0, [99, 7]) is None
    assert enc.action_index(0, [-1, 7]) is None


def test_action_index_attack_directions():
    # melee: dir must be one of the hex direction flags
    for flag in (1, 2, 4, 8, 16, 32):
        slot = enc.action_index(1, [flag, 40, -1, 2, 1])
        assert slot == enc.ATTACK_BASE + 40 * 7 + enc._DIR_FLAGS.index(flag)

    # ranged: dir == 0 with explicit target cell
    assert enc.action_index(1, [0, 40, -1, 2, 1]) == enc.ATTACK_BASE + 40 * 7 + enc.RANGED_DIR

    # shot with omitted target cell: resolved through the unit map
    unit_cells = {2: 40}
    assert enc.action_index(1, [0, -1, -1, 2, 1], unit_cells) == enc.ATTACK_BASE + 40 * 7 + enc.RANGED_DIR
    # same, with the engine's dir == -1 convention
    assert enc.action_index(1, [-1, -1, -1, 2, 1], unit_cells) == enc.ATTACK_BASE + 40 * 7 + enc.RANGED_DIR
    # cannot resolve without the unit map
    assert enc.action_index(1, [0, -1, -1, 2, 1]) is None

    # melee with dir == -1: target resolved via the unit map, direction derived from moveCell
    # (target unit stands at cell 14; cell 13 is adjacent LEFT of it)
    slot = enc.action_index(1, [-1, -1, 13, 2, 1], {2: 14})
    assert slot == enc.ATTACK_BASE + 14 * 7 + enc._DIR_FLAGS.index(4)

    # unknown directions must not crash and must not silently alias
    assert enc.action_index(1, [64, 40, -1, 2, 1]) is None


def test_direction_between_parity():
    # even row (row 0): RIGHT is +1, TOP_RIGHT is -10, BOTTOM_RIGHT is +12
    assert enc.direction_between(0, 1) == 4
    assert enc.direction_between(1, 0) == 32
    # odd row (row 1): TOP_RIGHT keeps the column, so 12 (row 1, col 1) -> 1 (row 0, col 1)
    assert enc.direction_between(12, 1) == 2
    # even row: BOTTOM_RIGHT from (row 0, col 1) lands on (row 1, col 2) = cell 13
    assert enc.direction_between(1, 13) == 8
    # non-adjacent cells have no direction
    assert enc.direction_between(0, 50) is None


def test_action_index_skip_and_unknown():
    assert enc.action_index(8, [5]) == enc.SKIP_INDEX
    assert enc.action_index(2, [1, 3, 40]) is None  # SPELLCAST is outside the v0 space
    assert enc.action_index(5, []) is None  # CATAPULT


def test_action_space_size():
    assert enc.ACTION_SPACE == enc.NUM_CELLS + enc.NUM_CELLS * (enc.NUM_DIRS + 1) + 1
    assert enc.SKIP_INDEX == enc.ACTION_SPACE - 1


def test_state_planes_shape_and_content():
    state = make_state(
        units=[
            {"u": 1, "side": "att", "mon": 13, "q": 30, "hpl": 10, "i": 0, "ti": -1, "sp": 2, "shots": 8, "moved": 0},
            {"u": 2, "side": "def", "mon": 22, "q": 20, "hpl": 20, "i": 13, "ti": 14, "sp": 2, "shots": 0, "moved": 0},
        ],
        obstacles=[50],
        cur=1,
    )
    planes = enc.state_planes(state)

    assert len(planes) == enc.NUM_PLANES
    assert all(len(row) == enc.BOARD_H for row in planes)
    assert all(len(cell) == enc.BOARD_W for row in planes for cell in row)

    # active unit marker at cell 0
    assert planes[0][0][0] == 1.0
    # attacker occupancy at cell 0, defender at 13 and 14 (wide tail)
    assert planes[1][0][0] == 1.0
    assert planes[2][1][2] == 1.0
    assert planes[2][1][3] == 1.0
    # wide tail marker
    assert planes[3][1][3] == 1.0
    # obstacle
    assert planes[4][4][6] == 1.0
    # ranged marker for the attacker stack (shots > 0)
    assert planes[9][0][0] == 1.0
    # no defender ranged marker
    assert planes[10][1][2] == 0.0


def test_state_scalars():
    state = make_state(
        units=[
            {"u": 1, "side": "att", "mon": 1, "q": 5, "hpl": 1, "i": 0, "ti": -1, "sp": 1, "shots": 0, "moved": 0},
            {"u": 2, "side": "def", "mon": 2, "q": 5, "hpl": 1, "i": 1, "ti": -1, "sp": 1, "shots": 0, "moved": 0},
        ],
        turn=100,
    )
    turn_n, att, dfd = enc.state_scalars(state)
    assert abs(turn_n - 0.5) < 1e-9
    assert att == 1 / 7
    assert dfd == 1 / 7


def test_side_to_move():
    state = make_state(
        units=[
            {"u": 1, "side": "att", "mon": 1, "q": 5, "hpl": 1, "i": 0, "ti": -1, "sp": 1, "shots": 0, "moved": 0},
            {"u": 2, "side": "def", "mon": 2, "q": 5, "hpl": 1, "i": 1, "ti": -1, "sp": 1, "shots": 0, "moved": 0},
        ],
        cur=2,
    )
    assert enc.side_to_move(state) == "def"
    state["cur"] = 1
    assert enc.side_to_move(state) == "att"


def test_legal_slots_order_and_dedup():
    legal = [
        {"act": 0, "args": [5, 1]},
        {"act": 0, "args": [5, 1]},   # duplicate slot
        {"act": 8, "args": [1]},
        {"act": 2, "args": [1]},      # unmappable -> dropped
        (0, [9, 7]),                  # tuple form
    ]
    assert enc.legal_slots(legal) == [5, enc.SKIP_INDEX, 9]


def test_side_to_move_fallback():
    # cur does not match any unit (e.g. battle already over) -> attacker by convention.
    assert enc.side_to_move(make_state(units=[], cur=99)) == "att"


def test_direction_between_bounds():
    assert enc.direction_between(-1, 5) is None
    assert enc.direction_between(5, enc.NUM_CELLS) is None


def test_value_target():
    assert enc.value_target("att", "att") == 1.0
    assert enc.value_target("att", "def") == -1.0
    assert enc.value_target("draw", "att") == 0.0


def test_wire_args_are_decoded_in_reverse_constructor_order():
    """Regression: the engine stores/serializes Command values in REVERSE constructor order;
    decoding them as constructor order mapped every MOVE of a unit onto MOVE_BASE + uid."""
    assert enc.ctor_args([42, 7]) == [7, 42]  # MOVE wire [dst, uid] -> (uid, dst)
    uid = 7
    slots = {enc.action_index(0, [cell, uid]) for cell in (10, 11, 12)}
    assert slots == {10, 11, 12}
    # ATTACK wire [dir, tgt, moveCell, targetUID, uid]: target cell 40 from direction RIGHT (4).
    assert enc.action_index(1, [4, 40, 39, 2, uid]) == enc.ATTACK_BASE + 40 * 7 + enc._DIR_FLAGS.index(4)
