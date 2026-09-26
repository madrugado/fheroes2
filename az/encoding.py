"""State and action encoding for the AlphaZero battle network.

Board: 11x9 hex grid (Battle::Board), cells indexed row-major (y * 11 + x).

Input planes (11 channels x 9 x 11), documented in az/README.md. Scalars are
appended as a small feature vector to the value trunk.

Fixed action space (793):
  [0, 99)      MOVE      -> destination head-cell
  [99, 792)    ATTACK    -> 99 target cells x 7 (6 melee directions + 1 ranged)
  792          SKIP

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
NUM_SCALARS = 3

MOVE_BASE = 0
ATTACK_BASE = NUM_CELLS  # 99
RANGED_DIR = 6
ACTION_SPACE = NUM_CELLS + NUM_CELLS * (NUM_DIRS + 1) + 1  # 793
SKIP_INDEX = ACTION_SPACE - 1

# Engine CellDirection flags in a fixed order; the direction value stored in the ATTACK
# command args is this flag value.
_DIR_FLAGS = [1, 2, 4, 8, 16, 32]


def action_index(act: int, args: list[int]) -> int | None:
    """Maps an engine command (act, args) to the fixed action index, or None."""
    if act == 0 and len(args) >= 2:  # MOVE: (uid, cell)
        cell = args[1]
        if 0 <= cell < NUM_CELLS:
            return MOVE_BASE + cell
        return None
    if act == 1 and len(args) >= 5:  # ATTACK: (uid, targetUID, moveCell, targetCell, dir)
        target_cell, direction = args[3], args[4]
        if not (0 <= target_cell < NUM_CELLS):
            return None
        if direction == 0:  # ranged shot
            return ATTACK_BASE + target_cell * (NUM_DIRS + 1) + RANGED_DIR
        if direction in _DIR_FLAGS:
            return ATTACK_BASE + target_cell * (NUM_DIRS + 1) + _DIR_FLAGS.index(direction)
        return None
    if act == 8:  # SKIP
        return SKIP_INDEX
    return None


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
    return [state.get("turn", 0) / 200.0, n_att / 7.0, n_def / 7.0]


def value_target(outcome: str, mover_side: str) -> float:
    """Value in [-1, 1] from the side-to-move perspective."""
    if outcome == "draw":
        return 0.0
    win = (outcome == mover_side)
    return 1.0 if win else -1.0
