"""Unit tests for az/strategy_env.py game loop with a scripted fake engine process."""

import io
import json
import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from line_reader import LineReader  # noqa: E402
from strategy_env import StrategyEnv  # noqa: E402


class FakeProc:
    """Stands in for subprocess.Popen: stdout replays a fixed script, stdin captures replies."""

    def __init__(self, script):
        read_fd, write_fd = os.pipe()
        payload = "".join(json.dumps(line) + "\n" for line in script) + "not json at all\n"
        os.write(write_fd, payload.encode())
        os.close(write_fd)  # EOF after the script
        self.stdout = os.fdopen(read_fd, "rb", buffering=0)
        self.stdin = io.BytesIO()

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


def make_env(script, proc=None):
    env = object.__new__(StrategyEnv)  # bypass __init__: no real engine process
    env.proc = proc or FakeProc(script)
    env._reader = LineReader(env.proc.stdout.fileno())
    env.read_timeout = 60.0
    env.last_turn_context = None
    return env


def replies(env):
    return [json.loads(line) for line in env.proc.stdin.getvalue().decode().splitlines()]


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

    assert replies(env) == [{"op": "pick", "h": 5, "i": 10}, {"op": "skip"}]


def test_close_is_safe_after_eof():
    env = make_env([{"ev": "game_end", "day": 1, "results": []}])
    env.run(lambda ev: None)

    env.close()  # must not raise even though the fake process already finished its output


def test_run_feeds_turn_context_to_context_aware_policies():
    class ContextPolicy:
        def __init__(self):
            self.contexts = []

        def observe_turn(self, ev):
            self.contexts.append(ev["t"])

        def __call__(self, ev):
            return None

    script = [
        {"ev": "turn_context", "t": 1, "heroes": []},
        {"ev": "turn_context", "t": 2, "heroes": []},
        {"ev": "game_end", "day": 2, "results": []},
    ]
    policy = ContextPolicy()
    make_env(script).run(policy)

    assert policy.contexts == [1, 2]


class InteractiveProc:
    """A fake engine that behaves like the real one: it sends turn_context + decision in ONE
    chunk and then blocks until the agent replies, keeping stdout open meanwhile."""

    def __init__(self):
        out_read, self._out_write = os.pipe()
        in_read, in_write = os.pipe()
        self.stdout = os.fdopen(out_read, "rb", buffering=0)
        self.stdin = os.fdopen(in_write, "wb", buffering=0)
        self._stdin_read = os.fdopen(in_read, "rb", buffering=0)
        self.reply = None
        self._thread = threading.Thread(target=self._engine, daemon=True)
        self._thread.start()

    def _engine(self):
        chunk = json.dumps({"ev": "turn_context", "t": 1, "p": "Blue", "heroes": []}) + "\n"
        chunk += json.dumps({"ev": "decision", "t": 1, "p": "Blue", "h": 5, "from": 0, "cands": [{"i": 10, "v": 1.0, "d": 1}]}) + "\n"
        os.write(self._out_write, chunk.encode())
        self.reply = self._stdin_read.readline()  # blocks like the engine's getline()
        os.write(self._out_write, (json.dumps({"ev": "game_end", "day": 1, "results": []}) + "\n").encode())
        os.close(self._out_write)

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


def test_query_in_the_same_chunk_as_the_previous_event_is_answered():
    """Regression: select() + buffered readline() swallowed the decision into the Python buffer
    and then waited for more data forever while the engine waited for our reply."""
    proc = InteractiveProc()
    env = make_env(None, proc=proc)
    env.read_timeout = 5.0  # the old reader would hit this timeout

    summaries = env.run(lambda ev: ev["cands"][0])

    assert json.loads(proc.reply) == {"op": "pick", "h": 5, "i": 10}
    assert len(summaries) == 1
