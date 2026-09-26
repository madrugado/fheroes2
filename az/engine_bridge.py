"""Bridge to the fheroes2 headless battle server (see az/README.md and battle_server.cpp)."""

import json
import os
import subprocess


class BattleEnv:
    """Synchronous JSON-lines client for one battle-server process."""

    def __init__(self, binary: str = "./fheroes2", map_name: str | None = None):
        env = dict(os.environ)
        env["FHEROES2_BATTLE_SERVER"] = "1"
        if map_name:
            env["FHEROES2_AUTO_PLAYTEST_MAP"] = map_name

        self.proc = subprocess.Popen(
            [binary],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
            bufsize=1,
        )

    def _send(self, obj: dict) -> None:
        self.proc.stdin.write(json.dumps(obj, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()

    def _read(self) -> dict | None:
        line = self.proc.stdout.readline()
        if not line:
            return None
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
