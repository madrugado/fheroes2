"""Bridge to the fheroes2 headless battle server (see rl/README.md and battle_server.cpp)."""

import json
import os
import select
import subprocess
import time

# Variables of the parent process that must never reach a battle-server child.
CHILD_ENV_BLOCKLIST = ("FHEROES2_AI_LOG", "FHEROES2_BATTLE_AGENT", "FHEROES2_STRATEGY_SERVER")


def format_stacks( stacks: list ) -> str:
    """[slot, monId, count] rows of a battle_start army -> 'slot:mon x count' CSV for "new"."""
    return ",".join( f"{slot}:{mon}x{count}" for slot, mon, count in stacks )


def hero_spec( army: dict ) -> tuple[int, str] | None:
    """(hero id, hex serialization) of a battle_start army, None without a commander."""
    if "hid" not in army or not army.get( "hero" ):
        return None
    return army["hid"], army["hero"]


def new_battle_from_setup( env, setup: dict, seed: int | None = None ) -> dict | None:
    """Rebuilds a real battle from its "battle_start" event (see rl/README.md, "Real-battle
    integration"): stacks, tile, world seed, formations, colors, commander heroes and the castle.
    `seed` overrides the battle seed (a different random stream on the same setup). `env` must
    have loaded the map of the real game."""
    att, dfd = setup["att"], setup["def"]
    return env.new_battle(
        seed=setup["seed"] if seed is None else seed,
        attacker=format_stacks( att["stacks"] ),
        defender=format_stacks( dfd["stacks"] ),
        tile=setup["tile"],
        world_seed=setup["wseed"],
        spread_att=bool( att["spread"] ),
        spread_def=bool( dfd["spread"] ),
        color_att=att.get( "c" ),
        color_def=dfd.get( "c" ),
        hero_att=hero_spec( att ),
        hero_def=hero_spec( dfd ),
        castle=setup.get( "castle" ),
        garrison=bool( dfd.get( "garrison" ) ),
    )


class BattleEnv:
    """Synchronous JSON-lines client for one battle-server process."""

    def __init__(self, binary: str = "./fheroes2", map_name: str | None = None):
        env = dict(os.environ)
        # The server is often a replica spawned next to a real game (battle_agent.py --policy
        # mcts): an inherited AI log would get the replica's battle events appended, and the agent
        # / strategic channel flags belong to the real engine only.
        for name in CHILD_ENV_BLOCKLIST:
            env.pop(name, None)
        env["FHEROES2_BATTLE_SERVER"] = "1"
        if map_name:
            env["FHEROES2_AUTO_PLAYTEST_MAP"] = map_name

        self._buffer: bytes | None = None

        self.proc = subprocess.Popen(
            [binary],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=False,
            env=env,
            bufsize=0,
        )

    def _send(self, obj: dict) -> None:
        self.proc.stdin.write((json.dumps(obj, separators=(",", ":")) + "\n").encode())
        self.proc.stdin.flush()

    def _read(self) -> dict | None:
        """Reads one JSON line with a hard 60-second cap.

        The engine may hang mid-line (e.g. an infinite loop inside the planner), so the line is
        assembled from raw timed reads instead of a blocking readline().
        """
        if self._buffer is None:
            self._buffer = b""

        deadline = time.monotonic() + 60.0
        fd = self.proc.stdout.fileno()
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("battle server did not reply within 60 seconds")

            ready, _, _ = select.select([fd], [], [], min(remaining, 1.0))
            if not ready:
                continue

            chunk = os.read(fd, 65536)
            if not chunk:
                line, self._buffer = self._buffer, b""
                return json.loads(line) if line.strip() else None
            self._buffer += chunk

        line, self._buffer = self._buffer.split(b"\n", 1)
        return json.loads(line)

    def new_battle( self, seed: int, attacker: str, defender: str, tile: int = -1, world_seed: int | None = None,
                    spread_att: bool | None = None, spread_def: bool | None = None,
                    color_att: int | None = None, color_def: int | None = None,
                    hero_att: tuple[int, str] | None = None, hero_def: tuple[int, str] | None = None,
                    castle: str | None = None, garrison: bool = False ) -> dict | None:
        """Starts a battle; attacker/defender are 'monsterIdx x count' CSV strings.

        Real-battle replication (rl/battle_agent.py) additionally supports stacks with explicit
        army slots ('0:13x30,2:21x24'), the world seed of the real game (obstacle placement
        derives from it), the battle formation and color (PlayerColor value, neutral = 0) of
        both armies, and the commander heroes as (hero id, hex save-game serialization) from
        "battle_start" — a hero's side then fights with the hero's own army (the stacks of that
        side are ignored), and for battles on a castle/town tile the castle's hex serialization
        (`castle`; `garrison=True`: the defenders are its garrison). The engine must load the same
        map as the real game."""
        obj: dict = {"op": "new", "seed": seed, "att": attacker, "def": defender, "tile": tile}
        if world_seed is not None:
            obj["wseed"] = world_seed
        if spread_att is not None:
            obj["sat"] = 1 if spread_att else 0
        if spread_def is not None:
            obj["sdf"] = 1 if spread_def else 0
        if color_att is not None:
            obj["acol"] = color_att
        if color_def is not None:
            obj["dcol"] = color_def
        if hero_att is not None:
            obj["ahid"], obj["ahero"] = hero_att
        if hero_def is not None:
            obj["dhid"], obj["dhero"] = hero_def
        if castle:
            obj["castle"] = castle
        if garrison:
            obj["dgar"] = 1

        self._send( obj )
        return self._read()

    def action(self, act: int, args: list[int]) -> dict | None:
        """Applies an action for the current unit; returns the next state/reply."""
        self._send({"op": "action", "act": act, "args": args})
        return self._read()

    def reset(self) -> dict | None:
        """Restores the battle to its initial position (replay-based search support)."""
        self._send({"op": "reset"})
        return self._read()

    def replay(self, path, full: bool = False) -> dict | None:
        """Applies the action path on top of the main line inside the engine (one roundtrip).

        Returns the state at the pause point (if the path ends mid-battle, with legal moves)
        or the final state with the result. `full=True` forces the engine to replay the whole
        main line from the battle root instead of restoring its main-line-end snapshot (the
        reference path for tests; identical result, slower).
        """
        obj = {
            "op": "replay",
            "acts": [act for act, _ in path],
            "lens": [len(args) for _, args in path],
            "args": [v for _, args in path for v in args],
        }
        if full:
            obj["full"] = 1
        self._send(obj)
        return self._read()

    def snapshot_save(self, snap_id: int) -> dict | None:
        """Stores the current pause-point state under the given id (battle server snapshots)."""
        self._send({"op": "snap", "id": snap_id})
        return self._read()

    def snapshot_restore(self, snap_id: int, path=(), save_as: int = 0, rollout: bool = False) -> dict | None:
        """Restores the snapshot and applies the optional action path suffix from it, saving the
        resulting state under save_as (if non-zero) — one roundtrip per search-tree node.

        rollout=True: then the built-in AI plays both sides to the end of the battle and the
        final state is returned (the engine is left there: restore a snapshot before the next
        search operation)."""
        obj = {
            "op": "restore",
            "id": snap_id,
            "save_as": save_as,
            "acts": [act for act, _ in path],
            "lens": [len(args) for _, args in path],
            "args": [v for _, args in path for v in args],
        }
        if rollout:
            obj["rollout"] = 1
        self._send(obj)
        return self._read()

    def snapshots_free(self) -> dict | None:
        """Releases all stored snapshots of the current battle."""
        self._send({"op": "snap_free"})
        return self._read()

    def suggest(self) -> dict | None:
        """Asks the built-in battle AI for its action at the current decision point (the reply
        carries the pre-decision state plus the "expert" field); nothing is applied."""
        self._send({"op": "suggest"})
        return self._read()

    def quit(self) -> None:
        try:
            self._send({"op": "quit"})
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        self.proc.wait(timeout=10)

    def close(self) -> None:
        if self.proc.poll() is None:
            self.quit()
