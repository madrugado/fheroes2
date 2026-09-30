"""Unit tests for rl/encoding.py (no engine required)."""

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
        assert slot == enc.ATTACK_BASE + 40 * enc.ATTACK_SLOTS + enc._DIR_FLAGS.index(flag)

    # ranged: dir == 0 with explicit target cell
    assert enc.action_index(1, [0, 40, -1, 2, 1]) == enc.ATTACK_BASE + 40 * enc.ATTACK_SLOTS + enc.RANGED_DIR

    # shot with omitted target cell: resolved through the unit map
    unit_cells = {2: 40}
    assert enc.action_index(1, [0, -1, -1, 2, 1], unit_cells) == enc.ATTACK_BASE + 40 * enc.ATTACK_SLOTS + enc.RANGED_DIR
    # same, with the engine's dir == -1 convention
    assert enc.action_index(1, [-1, -1, -1, 2, 1], unit_cells) == enc.ATTACK_BASE + 40 * enc.ATTACK_SLOTS + enc.RANGED_DIR
    # cannot resolve without the unit map
    assert enc.action_index(1, [0, -1, -1, 2, 1]) is None

    # melee with dir == -1: target resolved via the unit map, direction derived from moveCell
    # (target unit stands at cell 14; cell 13 is adjacent LEFT of it)
    slot = enc.action_index(1, [-1, -1, 13, 2, 1], {2: 14})
    assert slot == enc.ATTACK_BASE + 14 * enc.ATTACK_SLOTS + enc._DIR_FLAGS.index(4)

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
    assert enc.action_index(5, []) is None  # CATAPULT


def test_action_index_spellcast_uses_the_spell_slot():
    # Wire order is reversed: SPELLCAST(spell, cell) is [cell, spell], Teleport is [dst, src, spell].
    assert enc.action_index(2, [40, 3]) == enc.SPELL_BASE + 3
    assert enc.action_index(2, [-1, 3]) == enc.SPELL_BASE + 3  # mass spell, no target
    assert enc.action_index(2, [60, 40, 5]) == enc.SPELL_BASE + 5  # Teleport
    assert enc.action_index(2, [40, 0]) is None  # Spell::NONE
    assert enc.action_index(2, [40, enc.NUM_SPELLS]) is None
    assert enc.action_index(2, []) is None


def test_action_space_size():
    assert enc.SKIP_INDEX == enc.NUM_CELLS + enc.NUM_CELLS * enc.ATTACK_SLOTS
    assert enc.SPELL_BASE == enc.SKIP_INDEX + 1
    assert enc.ACTION_SPACE == enc.SPELL_BASE + enc.NUM_SPELLS


def test_state_scalars_include_commanders():
    state = {"turn": 20, "units": [], "heroes": [{"side": "def", "sp": 30, "cast": 1}]}
    scalars = enc.state_scalars(state)
    assert len(scalars) == enc.NUM_SCALARS
    assert scalars[3:] == [0.0, 0.0, 0.0, 1.0, 0.3, 1.0]
    # Old records without the "heroes" field: no commanders.
    assert enc.state_scalars({"turn": 1, "units": []})[3:] == [0.0] * 6


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
    turn_n, att, dfd = enc.state_scalars(state)[:3]
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
        {"act": 2, "args": [0]},      # unmappable -> dropped
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
    assert enc.action_index(1, [4, 40, 39, 2, uid]) == enc.ATTACK_BASE + 40 * enc.ATTACK_SLOTS + enc._DIR_FLAGS.index(4)


def test_wide_attacker_tail_strikes_have_their_own_slots():
    # Cell 12 hits cell 23 with BOTTOM_RIGHT (flag 8). From head 12 it is a head strike; a wide
    # unit whose head moves to 11 (tail on 12) delivers the same blow from its tail.
    head_strike = enc.action_index(1, [8, 23, -1, 3, 5], {5: 12})
    tail_strike = enc.action_index(1, [8, 23, 11, 3, 5], {5: 40})
    assert head_strike == enc.ATTACK_BASE + 23 * enc.ATTACK_SLOTS + enc._DIR_FLAGS.index(8)
    assert tail_strike == head_strike + enc.TAIL_DIR_OFFSET
    assert enc.neighbor_cell(12, 8) == 23 and enc.neighbor_cell(23, 1) == 12
    assert enc.neighbor_cell(0, 32) is None  # off the board


def test_battle_tokens_carry_every_stack_and_both_commanders():
    state = {"turn": 4, "cur": 1, "obstacles": [],
             "units": [{"u": 1, "side": "att", "mon": 13, "q": 30, "hpl": 150, "i": 12, "ti": 13, "sp": 5, "shots": 8, "moved": 1},
                       {"u": 2, "side": "def", "mon": 22, "q": 1, "hpl": 7, "i": 40, "ti": -1, "sp": 3, "shots": 0, "moved": 0}],
             "heroes": [{"side": "def", "sp": 25, "cast": 1}]}
    rows = enc.battle_tokens(state)
    assert len(rows) == enc.NUM_CELLS + 2 and all(len(row) == enc.BATTLE_TOKEN_W for row in rows)
    # A wide stack is seen on its head and its tail cell: the creature and the exact numbers.
    for cell in (12, 13):
        assert rows[cell][enc.MON_COL] == 14.0  # id + 1
        assert rows[cell][enc.UNIT_COLS] == enc.unit_features(state["units"][0])
    assert rows[40][enc.MON_COL] == 23.0
    # Health is not capped at 100: a 150 hp creature differs from a 100 hp one.
    hurt = dict(state["units"][0], hpl=100)
    assert enc.unit_features(hurt) != enc.unit_features(state["units"][0])
    assert rows[0][enc.MON_COL] == 0.0 and rows[0][enc.UNIT_COLS] == [0.0] * enc.NUM_UNIT_FEATURES
    attacker, defender = rows[enc.NUM_CELLS], rows[enc.NUM_CELLS + 1]
    assert attacker[enc.HERO_COLS] == [1.0, 0.0, 0.0, 0.0, 0.0, 4 / 50]  # no commander
    assert defender[enc.HERO_COLS] == [0.0, 1.0, 1.0, 0.25, 1.0, 4 / 50]  # spell points, cast this round
