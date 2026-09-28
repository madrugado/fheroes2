"""Unit tests for rl/engine_bridge.py robustness paths (fake binary instead of the engine).

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


def test_child_env_drops_parent_channel_flags(tmp_path, monkeypatch):
    # A replica spawned next to a real game must not inherit its AI log or channel flags.
    dump = tmp_path / "env.txt"
    binary = write_script(tmp_path, f'env > "{dump}"\n')
    monkeypatch.setenv("FHEROES2_AI_LOG", str(tmp_path / "ai.jsonl"))
    monkeypatch.setenv("FHEROES2_BATTLE_AGENT", "1")
    monkeypatch.setenv("FHEROES2_STRATEGY_SERVER", "1")
    monkeypatch.setenv("FHEROES2_DATA", "/keep/me")

    env = BattleEnv(binary=binary, map_name="Arena.mp2")
    try:
        env.proc.wait(timeout=5)
    finally:
        env.proc.kill()

    names = dict(line.split("=", 1) for line in dump.read_text().splitlines() if "=" in line)
    assert "FHEROES2_AI_LOG" not in names
    assert "FHEROES2_BATTLE_AGENT" not in names
    assert "FHEROES2_STRATEGY_SERVER" not in names
    assert names["FHEROES2_BATTLE_SERVER"] == "1"
    assert names["FHEROES2_AUTO_PLAYTEST_MAP"] == "Arena.mp2"
    assert names["FHEROES2_DATA"] == "/keep/me"
