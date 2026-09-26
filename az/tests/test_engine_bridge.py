"""Unit tests for az/engine_bridge.py robustness paths (fake binary instead of the engine).

The 60s watchdog TimeoutError path is not exercised here (it would make the suite slow);
it is covered by real engine integration in test_server_protocol.py.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from engine_bridge import BattleEnv  # noqa: E402


def write_script(tmp_path, body):
    path = tmp_path / "fake_engine.sh"
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return str(path)


def test_read_assembles_line_without_trailing_newline(tmp_path):
    # The engine died mid-line: _read must still return the assembled JSON, then None at EOF.
    binary = write_script(tmp_path, 'printf \'{"ev":"state","turn":1}\'; exit 0')
    env = BattleEnv(binary=binary)

    try:
        assert env._read() == {"ev": "state", "turn": 1}
        assert env._read() is None  # clean EOF with an empty buffer
    finally:
        env.proc.kill()


def test_quit_tolerates_dead_engine(tmp_path):
    # The engine exits immediately: the BrokenPipeError on send must be swallowed by quit().
    binary = write_script(tmp_path, "exit 0")
    env = BattleEnv(binary=binary)
    time.sleep(0.3)  # let the fake engine exit so the pipe is definitely broken

    env.quit()  # must not raise
