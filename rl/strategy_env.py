"""Environment for the strategic layer: drives full fheroes2 games where the
strategic target choice is delegated to a Python policy over JSON lines
(see rl/README.md and src/fheroes2/ai/ai_decision.cpp).

The game runs in the autonomous playtest mode (all players are AI-controlled);
at every hero target decision the engine sends the candidate list and blocks
until this process replies with a pick.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from line_reader import READ_TIMEOUT, LineReader  # noqa: E402
from strategy_policies import STRATEGIC_QUERIES, attach_build_result, strategic_reply  # noqa: E402


class StrategyEnv:
    def __init__(self, binary: str = "./fheroes2", map_name: str = "Arena.mp2", days: int = 30, playthroughs: int = 1,
                 seed: int | None = None, niceness: int = 10, ai_log: str | None = None,
                 report_days: list[int] | None = None, reseed: tuple[int, int] | None = None):
        env = dict(os.environ)
        env.pop("FHEROES2_AI_LOG", None)
        if ai_log:
            # The AI event log of this game (battle_start/battle_end with commanders and strengths).
            env["FHEROES2_AI_LOG"] = ai_log
        # A stray battle-agent flag would make the engine block on battle queries nobody answers.
        env.pop("FHEROES2_BATTLE_AGENT", None)
        env.pop("FHEROES2_AUTO_PLAYTEST_SEED", None)
        if seed is not None:
            # Reproducible playthroughs: equal seeds + equal agent choices = equal games.
            env["FHEROES2_AUTO_PLAYTEST_SEED"] = str(seed)
        env["FHEROES2_STRATEGY_SERVER"] = "1"
        env.pop("FHEROES2_REPORT_DAYS", None)
        if report_days:
            # "day_report" events (every player's stats + strongest hero) at the start of these days.
            env["FHEROES2_REPORT_DAYS"] = ",".join(str(day) for day in report_days)
        env.pop("FHEROES2_RESEED", None)
        if reseed:
            # (day, salt): other luck from that day on, the game before it unchanged.
            env["FHEROES2_RESEED"] = f"{reseed[0]}:{reseed[1]}"
        env["FHEROES2_AUTO_PLAYTEST"] = str(playthroughs)
        env["FHEROES2_AUTO_PLAYTEST_DAYS"] = str(days)
        env["FHEROES2_AUTO_PLAYTEST_MAP"] = map_name

        # Background priority: benchmarks and rollout labeling run many engines, the machine
        # must stay responsive. `nice` execs the binary with the same argv[0] path, so the
        # engine still finds its data next to it.
        self.proc = subprocess.Popen(
            ["nice", "-n", str(niceness), binary] if niceness else [binary],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=False,
            env=env,
            bufsize=0,
        )
        # Byte-level reader: select() + buffered readline() deadlocks when a query arrives in
        # the same chunk as the previous event (see line_reader.py).
        self._reader = LineReader(self.proc.stdout.fileno())
        self.read_timeout = READ_TIMEOUT
        self.last_turn_context: dict | None = None
        self.day_reports: list[dict] = []

    def run(self, policy, on_decision=None) -> list[dict]:
        """Runs the game(s) to completion, answering every strategic query with `policy` (see
        strategy_policies.strategic_reply: hero targets, building, hiring).

        Returns the list of game_end events. If `on_decision` is provided, it is called with the
        record of every strategic choice (kind target/build/hire, with the game outcome attached).
        """
        summaries: list[dict] = []
        buffered_records: list[dict] = []

        assert self.proc.stdout is not None and self.proc.stdin is not None

        while True:
            line = self._reader.read_line(self.read_timeout)
            if line is None:
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
            elif kind in STRATEGIC_QUERIES:
                reply, record = strategic_reply(policy, ev)
                buffered_records.append(record)
                self._send(reply)
            elif kind == "build_result":
                attach_build_result(buffered_records, ev)
            elif kind == "day_report":
                self.day_reports.append(ev)
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
        self.proc.stdin.write((json.dumps(obj, separators=(",", ":")) + "\n").encode())
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
