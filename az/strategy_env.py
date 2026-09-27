"""Environment for the strategic layer: drives full fheroes2 games where the
strategic target choice is delegated to a Python policy over JSON lines
(see az/README.md and src/fheroes2/ai/ai_decision.cpp).

The game runs in the autonomous playtest mode (all players are AI-controlled);
at every hero target decision the engine sends the candidate list and blocks
until this process replies with a pick.
"""

from __future__ import annotations

import json
import os
import select
import subprocess


class StrategyEnv:
    def __init__(self, binary: str = "./fheroes2", map_name: str = "Arena.mp2", days: int = 30, playthroughs: int = 1):
        env = dict(os.environ)
        env["FHEROES2_STRATEGY_SERVER"] = "1"
        env["FHEROES2_AUTO_PLAYTEST"] = str(playthroughs)
        env["FHEROES2_AUTO_PLAYTEST_DAYS"] = str(days)
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
        self.last_turn_context: dict | None = None

    def run(self, policy, on_decision=None) -> list[dict]:
        """Runs the game(s) to completion. `policy(decision_event) -> chosen candidate dict | None`.

        Returns the list of game_end events. If `on_decision` is provided, it is called with
        (decision_event, chosen_candidate) for every decision (e.g. for recording traces).
        """
        summaries: list[dict] = []
        buffered_records: list[dict] = []

        assert self.proc.stdout is not None and self.proc.stdin is not None

        stdout = self.proc.stdout
        while True:
            if not select.select([stdout], [], [], 60.0)[0]:
                raise TimeoutError("game did not produce output within 60 seconds")
            line = stdout.readline()
            if not line:
                break
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue

            kind = ev.get("ev")
            if kind == "turn_context":
                self.last_turn_context = ev
                if hasattr(policy, "observe_turn"):
                    policy.observe_turn(ev)
            elif kind == "decision":
                chosen = policy(ev)
                buffered_records.append(
                    {
                        "t": ev.get("t"),
                        "h": ev.get("h"),
                        "from": ev.get("from"),
                        "cands": ev.get("cands"),
                        "chosen": chosen.get("i") if chosen else None,
                    }
                )

                if chosen is not None:
                    self._send({"op": "pick", "h": ev["h"], "i": chosen["i"]})
                else:
                    self._send({"op": "skip"})
            elif kind == "game_end":
                outcome = {"winner_states": ev.get("results"), "day": ev.get("day")}
                for record in buffered_records:
                    record["outcome"] = outcome
                    if on_decision is not None:
                        on_decision(record)
                buffered_records.clear()
                summaries.append({"type": "game_end", **ev})

        return summaries

    def _send(self, obj: dict) -> None:
        self.proc.stdin.write(json.dumps(obj, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self._send({"op": "quit"})
            except BrokenPipeError:
                pass
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
