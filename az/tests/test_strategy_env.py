"""Unit tests for az/strategy_env.py game loop with a scripted fake engine process."""

import io
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from strategy_env import StrategyEnv  # noqa: E402


class FakeProc:
    """Stands in for subprocess.Popen: stdout replays a fixed script, stdin captures replies."""

    def __init__(self, script):
        read_fd, write_fd = os.pipe()
        payload = "".join(json.dumps(line) + "\n" for line in script) + "not json at all\n"
        os.write(write_fd, payload.encode())
        os.close(write_fd)  # EOF after the script
        self.stdout = io.TextIOWrapper(os.fdopen(read_fd, "rb"), encoding="utf-8")
        self.stdin = io.StringIO()

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


def make_env(script):
    env = object.__new__(StrategyEnv)  # bypass __init__: no real engine process
    env.proc = FakeProc(script)
    env.last_turn_context = None
    return env


def test_run_records_decisions_with_outcome():
    results = [{"c": 0, "s": 2, "d": 12}]
    script = [
        {"ev": "turn_context", "t": 1, "res": [0] * 7, "castles": [], "heroes": []},
        {"ev": "decision", "t": 1, "h": 5, "from": 100, "cands": [{"i": 10, "v": 3.0}]},
        {"ev": "decision", "t": 1, "h": 6, "from": 100, "cands": [{"i": 20, "v": 1.0}]},
        {"ev": "game_end", "playthrough": 0, "day": 12, "results": results},
    ]
    env = make_env(script)
    decisions = []

    def policy(ev):
        return ev["cands"][0] if ev["h"] == 5 else None

    summaries = env.run(policy, on_decision=decisions.append)

    assert len(summaries) == 1 and summaries[0]["day"] == 12
    assert env.last_turn_context["t"] == 1

    assert len(decisions) == 2
    assert decisions[0]["chosen"] == 10
    assert decisions[1]["chosen"] is None  # policy declined -> skip
    assert all(d["outcome"] == {"winner_states": results, "day": 12} for d in decisions)

    replies = [json.loads(line) for line in env.proc.stdin.getvalue().splitlines()]
    assert replies == [{"op": "pick", "h": 5, "i": 10}, {"op": "skip"}]


def test_close_is_safe_after_eof():
    env = make_env([{"ev": "game_end", "day": 1, "results": []}])
    env.run(lambda ev: None)

    env.close()  # must not raise even though the fake process already finished its output
