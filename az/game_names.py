"""Human-readable names for the numeric ids of the strategic protocol (map object types, monsters,
races), for text agents (grpo_env.py).

The names are parsed once from the engine headers (`MP2::MapObjectType` in maps/mp2.h,
`Monster::MonsterType` in monster/monster.h), so they cannot drift from the engine; without the
sources the lookups fall back to "object #n" / "monster #n".
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "fheroes2")

RACES = {1: "Knight", 2: "Barbarian", 4: "Sorceress", 8: "Warlock", 16: "Wizard", 32: "Necromancer",
         64: "Multi", 128: "Random"}

RESOURCES = ("wood", "mercury", "ore", "sulfur", "crystal", "gems", "gold")


def _enum_body(text: str, name: str) -> str:
    start = text.find(f"enum {name}")
    if start < 0:
        return ""
    open_brace = text.find("{", start)
    close_brace = text.find("};", open_brace)
    return text[open_brace + 1:close_brace] if open_brace >= 0 and close_brace >= 0 else ""


def _entries(body: str):
    """(NAME, value expression or None) per enumerator; comments stripped."""
    body = re.sub(r"//[^\n]*", "", body)
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    for item in body.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, expr = item.partition("=")
        yield name.strip(), (expr.strip() or None)


def _evaluate(expr: str, known: dict[str, int]) -> int | None:
    total = 0
    for term in expr.split("+"):
        term = term.strip()
        if re.fullmatch(r"-?(0x[0-9a-fA-F]+|\d+)", term):
            total += int(term, 0)
        elif term in known:
            total += known[term]
        else:
            return None
    return total


def _pretty(identifier: str, prefixes: tuple[str, ...] = ()) -> str:
    for prefix in prefixes:
        if identifier.startswith(prefix):
            identifier = identifier[len(prefix):]
            break
    return identifier.replace("_", " ").lower().capitalize()


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


@lru_cache(maxsize=1)
def object_names() -> dict[int, str]:
    body = _enum_body(_read(os.path.join(SRC, "maps", "mp2.h")), "MapObjectType")
    values: dict[str, int] = {}
    names: dict[int, str] = {}
    for name, expr in _entries(body):
        value = _evaluate(expr, values) if expr is not None else None
        if value is None:
            continue
        values[name] = value
        # The first enumerator of a value wins (later ones are aliases like OBJ_ACTION_OBJECT_TYPE).
        names.setdefault(value, _pretty(name, ("OBJ_NON_ACTION_", "OBJ_ACTION_", "OBJ_")))
    return names


@lru_cache(maxsize=1)
def monster_names() -> dict[int, str]:
    body = _enum_body(_read(os.path.join(SRC, "monster", "monster.h")), "MonsterType")
    names: dict[int, str] = {}
    value = -1
    for name, expr in _entries(body):
        value = _evaluate(expr, {}) if expr is not None else value + 1
        if value is None:
            break
        names[value] = _pretty(name)
    return names


def object_name(obj: int) -> str:
    return object_names().get(obj, f"object #{obj}")


def monster_name(mon: int) -> str:
    return monster_names().get(mon, f"monster #{mon}")


def race_name(race: int) -> str:
    return RACES.get(race, f"race #{race}")


def format_resources(res, skip_zero: bool = True) -> str:
    parts = [f"{name} {amount}" for name, amount in zip(RESOURCES, res or []) if amount or not skip_zero]
    return ", ".join(parts) if parts else "nothing"
