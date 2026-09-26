"""Bridge to the fheroes2 headless battle server (see az/README.md and battle_server.cpp)."""

import json
import os
import select
import subprocess
import time


class BattleEnv:
    """Synchronous JSON-lines client for one battle-server process."""

    def __init__(self, binary: str = "./fheroes2", map_name: str | None = None):
        env = dict(os.environ)
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

    def new_battle(self, seed: int, attacker: str, defender: str, tile: int = -1) -> dict | None:
        """Starts a battle; attacker/defender are 'monsterIdx x count' CSV strings."""
        self._send({"op": "new", "seed": seed, "att": attacker, "def": defender, "tile": tile})
        return self._read()

    def action(self, act: int, args: list[int]) -> dict | None:
        """Applies an action for the current unit; returns the next state/reply."""
        self._send({"op": "action", "act": act, "args": args})
        return self._read()

    def reset(self) -> dict | None:
        """Restores the battle to its initial position (replay-based search support)."""
        self._send({"op": "reset"})
        return self._read()

    def replay(self, path) -> dict | None:
        """Resets the battle and applies the whole action path inside the engine (one roundtrip).

        Returns the state at the pause point (if the path ends mid-battle, with legal moves)
        or the final state with the result.
        """
        self._send(
            {
                "op": "replay",
                "acts": [act for act, _ in path],
                "lens": [len(args) for _, args in path],
                "args": [v for _, args in path for v in args],
            }
        )
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
