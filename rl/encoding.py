"""State and action encoding for the AlphaZero battle network.

Board: 11x9 hex grid (Battle::Board), cells indexed row-major (y * 11 + x).

Input planes (11 channels x 9 x 11), documented in rl/README.md. Scalars are
appended as a small feature vector to the value trunk.

Fixed action space (1460):
  [0, 99)       MOVE      -> destination head-cell
  [99, 1386)    ATTACK    -> 99 target cells x 13: 6 melee directions struck from the attacker's
                head cell, 6 struck from its tail cell (wide units), 1 ranged
  1386          SKIP
  [1387, 1460)  SPELLCAST -> spell id (the hero's spell; the target is not encoded: all legal
                targets of one spell share the slot and split its probability)

An engine ATTACK command is (attUID, defUID, moveCell, targetCell, dir): `dir` points from the
attacking cell to the target cell, so (targetCell, dir) fixes the attacking cell; whether that is
the head or the tail of a wide attacker tells two otherwise equal slots apart (a wide unit can
reach the same attacking cell with its head from one position and with its tail from another).
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
TAIL_DIR_OFFSET = NUM_DIRS  # tail-cell strikes: sub-slots [6, 12)
RANGED_DIR = 2 * NUM_DIRS  # 12
ATTACK_SLOTS = 2 * NUM_DIRS + 1  # 13 sub-slots per target cell
SKIP_INDEX = NUM_CELLS + NUM_CELLS * ATTACK_SLOTS  # 1386
SPELL_BASE = SKIP_INDEX + 1  # 1387
NUM_SPELLS = 73  # Spell::SPELL_COUNT (spell.h)
RETREAT_INDEX = SPELL_BASE + NUM_SPELLS  # 1460: the commander retreats (Battle::CommandType::RETREAT)
SURRENDER_INDEX = RETREAT_INDEX + 1  # 1461: the commander surrenders (pays gold, keeps the army)
ACTION_SPACE = SURRENDER_INDEX + 1  # 1462 (1460 before retreat/surrender became actions, 2026-10-10)
SPELLCAST = 2  # Battle::CommandType::SPELLCAST
RETREAT = 6  # Battle::CommandType::RETREAT (no args)
SURRENDER = 7  # Battle::CommandType::SURRENDER (no args)

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


# Opposite direction flags (the direction from the target back to the attacking cell).
_REFLECT = {1: 8, 8: 1, 2: 16, 16: 2, 4: 32, 32: 4}


def neighbor_cell(cell: int, flag: int) -> int | None:
    """The neighbor of `cell` in direction `flag`, or None off the board."""
    row, col = divmod(cell, BOARD_W)
    dcol, drow = _DIR_OFFSETS[flag][row % 2]
    ncol, nrow = col + dcol, row + drow
    if 0 <= ncol < BOARD_W and 0 <= nrow < BOARD_H:
        return nrow * BOARD_W + ncol
    return None


def attack_parts(args: list[int], unit_cells: dict[int, int] | None = None) -> tuple[int, int] | None:
    """(target cell, attack sub-slot in [0, ATTACK_SLOTS)) of an ATTACK in constructor order
    (uid, targetUID, moveCell, targetCell, dir), or None when it cannot be resolved.

    Omitted fields (<= 0) are resolved like the engine does for the built-in AI: the target cell
    through unit_cells (uid -> head cell), the direction from the move cell. A strike whose
    attacking cell is not the attacker's head (move cell, or its current head from unit_cells)
    is a wide unit's tail strike."""
    uid, target_uid, move_cell, target_cell, direction = args[:5]

    if target_cell < 0:
        target_cell = unit_cells.get(target_uid) if unit_cells else None
    if target_cell is None or not (0 <= target_cell < NUM_CELLS):
        return None

    if direction in _DIR_FLAGS:
        head = move_cell if move_cell >= 0 else (unit_cells.get(uid) if unit_cells else None)
        attacking_cell = neighbor_cell(target_cell, _REFLECT[direction])
        tail = head is not None and attacking_cell is not None and attacking_cell != head
        return target_cell, _DIR_FLAGS.index(direction) + (TAIL_DIR_OFFSET if tail else 0)

    if direction <= 0:
        # No explicit direction: a shot or an in-place attack targets the unit directly.
        if 0 <= move_cell < NUM_CELLS:
            derived = direction_between(move_cell, target_cell)
            if derived is not None:
                return target_cell, _DIR_FLAGS.index(derived)
        return target_cell, RANGED_DIR

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
        parts = attack_parts(args, unit_cells)
        if parts is None:
            return None
        target_cell, sub = parts
        return ATTACK_BASE + target_cell * ATTACK_SLOTS + sub

    if act == 8:  # SKIP
        return SKIP_INDEX
    if act == RETREAT:
        return RETREAT_INDEX
    if act == SURRENDER:
        return SURRENDER_INDEX
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


# Transformer battle tokens (battle_tokens): one row per board cell, then one per commander (the
# attacker's, the defender's). Columns: the cell planes (state_planes), the features of the stack on
# the cell, the stack's creature id + 1 (0: no unit; the network embeds it), the commander features.
NUM_UNIT_FEATURES = 6  # count (log, linear), health of the top creature (log), speed, shots, moved
NUM_MONSTER_IDS = 127  # Monster::MONSTER_COUNT is 72; ids above the table are clipped
NUM_HERO_FEATURES = 6  # attacker's / defender's, present, spell points, cast this round, turn
PLANE_COLS = slice(0, NUM_PLANES)
UNIT_COLS = slice(NUM_PLANES, NUM_PLANES + NUM_UNIT_FEATURES)
MON_COL = NUM_PLANES + NUM_UNIT_FEATURES
HERO_COLS = slice(MON_COL + 1, MON_COL + 1 + NUM_HERO_FEATURES)
BATTLE_TOKEN_W = MON_COL + 1 + NUM_HERO_FEATURES


def monster_token(monster_id) -> float:
    """The creature's row of the network's creature embedding (mon_embed): Monster::GetID() + 1, 0 =
    no creature. The ONE encoding of a creature — battle units ("mon") and strategic army stacks
    (strategy_net._mon_slots) both go through it, so a creature is the same vector in both."""
    return float(min(max(int(monster_id), 0), NUM_MONSTER_IDS - 1) + 1)


def unit_features(unit: dict) -> list[float]:
    """What the battle screen shows of a stack beyond its cell: the exact count, the health of its
    top creature (no cap: creatures have 1 to 250+ hit points), speed, shots left, moved."""
    return [math.log2(unit["q"] + 1) / 10.0, min(unit["q"], 1000) / 1000.0, math.log2(unit["hpl"] + 1) / 10.0,
            unit.get("sp", 0) / 10.0, min(unit.get("shots", 0), 32) / 32.0, float(unit.get("moved", 0))]


def battle_tokens(state: dict) -> list[list[float]]:
    """(NUM_CELLS + 2) x BATTLE_TOKEN_W rows: every cell with its planes and the stack standing on it
    (head and tail cells of a wide unit alike), then the attacker's and the defender's commander:
    present, spell points, whether it cast this round (both are visible in the battle's hero dialog)."""
    planes = state_planes(state)
    rows = [[planes[channel][cell // BOARD_W][cell % BOARD_W] for channel in range(NUM_PLANES)] + [0.0] * (BATTLE_TOKEN_W - NUM_PLANES)
            for cell in range(NUM_CELLS)]
    for unit in state["units"]:
        features = unit_features(unit)
        monster = monster_token(unit.get("mon", 0))
        for cell in {unit["i"], unit.get("ti", -1)}:
            if 0 <= cell < NUM_CELLS:
                rows[cell][UNIT_COLS] = features
                rows[cell][MON_COL] = monster
    heroes = {hero["side"]: hero for hero in state.get("heroes", [])}
    for index, side in enumerate(("att", "def")):
        row = [0.0] * BATTLE_TOKEN_W
        hero = heroes.get(side)
        row[HERO_COLS] = [float(index == 0), float(index == 1), float(hero is not None),
                          hero.get("sp", 0) / 100.0 if hero else 0.0, float(hero.get("cast", 0)) if hero else 0.0,
                          state.get("turn", 0) / 50.0]
        rows.append(row)
    return rows


# The loser of a battle that ends by the commander's escape keeps something (user decision 2026-10-10): a
# retreating hero survives with his skills, artifacts and experience (the army is lost), a surrendering hero
# also keeps his army (he pays gold for it).
RETREAT_CREDIT = 0.4
SURRENDER_CREDIT = 0.8


def value_target(outcome: str, mover_side: str, flee: str | None = None, how: str | None = None) -> float:
    """Value in [-1, 1] from the side-to-move perspective: +1 won, -1 lost, 0 draw; a loss by the side's own
    retreat / surrender (battle state "flee" = that side, "how") is worth -1 + RETREAT_CREDIT /
    SURRENDER_CREDIT, a win over a fleeing enemy as much less (zero-sum)."""
    if outcome == "draw" or outcome is None:
        return 0.0
    value = 1.0 if outcome == mover_side else -1.0
    if flee is not None:
        credit = SURRENDER_CREDIT if how == "surrender" else RETREAT_CREDIT
        value += credit if flee == mover_side else -credit
    return value
