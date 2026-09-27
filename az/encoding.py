"""State and action encoding for the AlphaZero battle network.

Board: 11x9 hex grid (Battle::Board), cells indexed row-major (y * 11 + x).

Input planes (11 channels x 9 x 11), documented in az/README.md. Scalars are
appended as a small feature vector to the value trunk.

Fixed action space (866):
  [0, 99)      MOVE      -> destination head-cell
  [99, 792)    ATTACK    -> 99 target cells x 7 (6 melee directions + 1 ranged)
  792          SKIP
  [793, 866)   SPELLCAST -> spell id (the hero's spell; the target is not encoded: all legal
               targets of one spell share the slot and split its probability)

An engine ATTACK command is (attUID, defUID, moveCell, targetCell, dir); the fixed
slot encodes (targetCell, dir) — for wide targets two legal moves may collapse into
one slot, the first one wins (v0 approximation).
"""

import math

BOARD_W = 11
BOARD_H = 9
NUM_CELLS = BOARD_W * BOARD_H
NUM_DIRS = 6  # hex directions, see Battle::CellDirection (values 1,2,4,8,16,32)

NUM_PLANES = 11
NUM_SCALARS = 9  # turn, unit counts, and per side: has commander, spell points, cast this round

MOVE_BASE = 0
ATTACK_BASE = NUM_CELLS  # 99
RANGED_DIR = 6
SKIP_INDEX = NUM_CELLS + NUM_CELLS * (NUM_DIRS + 1)  # 792
SPELL_BASE = SKIP_INDEX + 1  # 793
NUM_SPELLS = 73  # Spell::SPELL_COUNT (spell.h)
ACTION_SPACE = SPELL_BASE + NUM_SPELLS  # 866
SPELLCAST = 2  # Battle::CommandType::SPELLCAST

# Engine CellDirection flags in a fixed order; the direction value stored in the ATTACK
# command args is this flag value.
_DIR_FLAGS = [1, 2, 4, 8, 16, 32]

# Hex neighbor offsets (parity-aware, see Board::GetIndexDirection): row parity (0 = even,
# 1 = odd) -> flag -> (dcol, drow).
_DIR_OFFSETS = {
    1: {0: (0, -1), 1: (-1, -1)},    # TOP_LEFT
    2: {0: (1, -1), 1: (0, -1)},     # TOP_RIGHT
    4: {0: (1, 0), 1: (1, 0)},       # RIGHT
    8: {0: (1, 1), 1: (0, 1)},       # BOTTOM_RIGHT
    16: {0: (0, 1), 1: (-1, 1)},     # BOTTOM_LEFT
    32: {0: (-1, 0), 1: (-1, 0)},    # LEFT
}


def direction_between(from_cell: int, to_cell: int) -> int | None:
    """Engine direction flag leading from from_cell to the adjacent to_cell, or None."""
    if not (0 <= from_cell < NUM_CELLS and 0 <= to_cell < NUM_CELLS):
        return None

    from_row, from_col = divmod(from_cell, BOARD_W)
    to_row, to_col = divmod(to_cell, BOARD_W)
    parity = from_row % 2

    for flag, offsets in _DIR_OFFSETS.items():
        dcol, drow = offsets[parity]
        if from_col + dcol == to_col and from_row + drow == to_row:
            return flag

    return None


def ctor_args(args) -> list[int]:
    """Engine wire args -> constructor-order parameters.

    `Battle::Command` stores its values in REVERSE constructor order and the engine serializes
    them as stored: MOVE is `[dst, uid]`, ATTACK is `[dir, tgt, moveCell, targetUID, uid]`,
    SKIP is `[uid]` (see battle_command.h). Every decoder must go through this function —
    reading the wire order as constructor order collapses all MOVEs of a unit onto one action
    index (a bug that went unnoticed until 2026-09-27).
    """
    return list(reversed(list(args)))


def action_index(act: int, args: list[int], unit_cells: dict[int, int] | None = None) -> int | None:
    """Maps an engine command (act, wire-order args) to the fixed action index, or None.

    ATTACK commands may omit the target cell and/or the direction (<= 0) — in that case the target cell is resolved through unit_cells
    (uid -> head cell) and the direction is derived from the two cells.
    """
    args = ctor_args(args)

    if act == 0 and len(args) >= 2:  # MOVE: (uid, cell)
        cell = args[1]
        if 0 <= cell < NUM_CELLS:
            return MOVE_BASE + cell
        return None

    if act == 1 and len(args) >= 5:  # ATTACK: (uid, targetUID, moveCell, targetCell, dir)
        _, target_uid, move_cell, target_cell, direction = args[:5]

        if target_cell < 0:
            target_cell = unit_cells.get(target_uid) if unit_cells else None
        if target_cell is None or not (0 <= target_cell < NUM_CELLS):
            return None

        if direction in _DIR_FLAGS:
            return ATTACK_BASE + target_cell * (NUM_DIRS + 1) + _DIR_FLAGS.index(direction)

        if direction <= 0:
            # No explicit direction: a shot or an in-place attack targets the unit directly.
            if move_cell >= 0:
                derived = direction_between(move_cell, target_cell)
                if derived is not None:
                    return ATTACK_BASE + target_cell * (NUM_DIRS + 1) + _DIR_FLAGS.index(derived)
            return ATTACK_BASE + target_cell * (NUM_DIRS + 1) + RANGED_DIR

        return None

    if act == 8:  # SKIP
        return SKIP_INDEX
    if act == SPELLCAST and len(args) >= 1:  # SPELLCAST: (spell, target...)
        spell = args[0]
        if 0 < spell < NUM_SPELLS:
            return SPELL_BASE + spell
        return None
    return None


def unit_cells_map(units: list[dict]) -> dict[int, int]:
    """uid -> head cell for all live units (used to resolve target-less ATTACK args)."""
    return {unit["u"]: unit["i"] for unit in units}


def legal_slots(legal_moves: list[dict]) -> list[int]:
    """Fixed-slot indices for the engine's legal move list (order preserved, duplicates collapsed)."""
    slots: list[int] = []
    seen: set[int] = set()
    for move in legal_moves:
        if isinstance(move, dict):
            act, args = move["act"], move["args"]
        else:
            act, args = move[0], move[1]
        slot = action_index(act, args)
        if slot is not None and slot not in seen:
            seen.add(slot)
            slots.append(slot)
    return slots


def side_to_move(state: dict) -> str:
    cur = state.get("cur", -1)
    for unit in state["units"]:
        if unit["u"] == cur:
            return unit["side"]
    return "att"


def state_planes(state: dict) -> list[list[list[float]]]:
    """Returns NUM_PLANES x BOARD_H x BOARD_W float planes for the battle state."""
    planes = [[[0.0] * BOARD_W for _ in range(BOARD_H)] for _ in range(NUM_PLANES)]

    def put(channel: int, cell: int, value: float) -> None:
        if 0 <= cell < NUM_CELLS:
            planes[channel][cell // BOARD_W][cell % BOARD_W] = value

    cur_uid = state.get("cur", -1)

    for unit in state["units"]:
        head, tail = unit["i"], unit.get("ti", -1)
        side_is_att = unit["side"] == "att"
        qty_channel = 5 if side_is_att else 6
        hp_channel = 7 if side_is_att else 8
        ranged_channel = 9 if side_is_att else 10

        strength = math.log2(unit["q"] + 1) / 8.0
        hp_frac = min(unit["hpl"] / 100.0, 1.0)

        for cell in {head, tail}:
            if cell < 0:
                continue
            put(1 if side_is_att else 2, cell, 1.0)
            put(qty_channel, cell, strength)
            put(hp_channel, cell, hp_frac)
            if unit["shots"] > 0:
                put(ranged_channel, cell, 1.0)
            if unit["u"] == cur_uid:
                put(0, cell, 1.0)

        if tail >= 0:
            put(3, tail, 1.0)

    for cell in state.get("obstacles", []):
        put(4, cell, 1.0)

    return planes


def state_scalars(state: dict) -> list[float]:
    n_att = sum(1 for u in state["units"] if u["side"] == "att")
    n_def = sum(1 for u in state["units"] if u["side"] == "def")
    scalars = [state.get("turn", 0) / 200.0, n_att / 7.0, n_def / 7.0]

    # Commanders ("heroes" in the state; absent in old records): presence, spell points and
    # whether the side already cast a spell this round.
    heroes = {hero["side"]: hero for hero in state.get("heroes", [])}
    for side in ("att", "def"):
        hero = heroes.get(side)
        if hero is None:
            scalars += [0.0, 0.0, 0.0]
        else:
            scalars += [1.0, hero.get("sp", 0) / 100.0, float(hero.get("cast", 0))]
    return scalars


def value_target(outcome: str, mover_side: str) -> float:
    """Value in [-1, 1] from the side-to-move perspective."""
    if outcome == "draw":
        return 0.0
    win = (outcome == mover_side)
    return 1.0 if win else -1.0
