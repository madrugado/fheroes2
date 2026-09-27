"""Integration tests for the external agent channels of a real game (requires the ./fheroes2
binary and game data; skipped automatically when the binary is missing).

- A dead agent: both channels must fall back to the built-in AI permanently after the FIRST
  unanswered query (regression: the "broken" flag used to live in two different statics, so
  every later query was still written to the dead agent).
- One agent serving both channels (az/game_agent.py) over the same pipe in a real game.
"""

import json
import os
import subprocess
import sys

import pytest

sys.path.insert( 0, os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), ".." ) )

from battle_agent import _LineReader  # noqa: E402
from game_agent import GameAgent  # noqa: E402
from strategy_policies import TempoPolicy  # noqa: E402

REPO_ROOT = os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )
BINARY = os.path.join( REPO_ROOT, "fheroes2" )
MAP_NAME = "Arena.mp2"
# The first battles on Arena.mp2 happen on day 4; the world seed is random per engine process,
# so one extra day of margin (a 4-day run once ended without any agent battle).
DAYS = 5

pytestmark = pytest.mark.skipif( not os.path.exists( BINARY ), reason="fheroes2 binary not built" )


def test_dead_agent_gets_one_query_per_channel_then_builtin_ai_plays_on():
    env = dict( os.environ )
    env.update( {
        "FHEROES2_STRATEGY_SERVER": "1",
        "FHEROES2_BATTLE_AGENT": "1",
        "FHEROES2_AUTO_PLAYTEST": "1",
        "FHEROES2_AUTO_PLAYTEST_DAYS": "1",  # the first strategic query comes on day 1
        "FHEROES2_AUTO_PLAYTEST_MAP": MAP_NAME,
    } )
    proc = subprocess.Popen( [BINARY], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             text=False, env=env, bufsize=0 )
    proc.stdin.close()  # the agent is gone before the first query

    events = []
    try:
        reader = _LineReader( proc.stdout.fileno() )
        while True:
            line = reader.read_line()
            if line is None:
                break
            try:
                events.append( json.loads( line ) )
            except json.JSONDecodeError:
                continue
        assert proc.wait( timeout=30 ) == 0, "the game must finish on the built-in AI"
    finally:
        if proc.poll() is None:
            proc.kill()

    kinds = [e.get( "ev" ) for e in events]
    assert kinds.count( "decision" ) == 1, "strategic queries after the agent died"
    assert kinds.count( "state" ) <= 1, "battle queries after the agent died"
    assert kinds.count( "battle_start" ) <= 1
    # game_end goes out while at least one channel is alive: with no battle within the day
    # limit the battle channel never saw the dead agent.
    if "state" in kinds:
        assert "game_end" not in kinds


def test_one_agent_serves_both_channels_in_a_real_game():
    agent = GameAgent( strategy_policy=TempoPolicy(), binary=BINARY, map_name=MAP_NAME, days=DAYS, playthroughs=1,
                       policy="random", seed=7 )
    records = []
    try:
        summaries = agent.run( on_record=records.append )
    finally:
        agent.close()

    strategy = [r for r in records if r.get( "kind" ) == "strategy"]
    battle = [r for r in records if r.get( "kind" ) != "strategy"]

    assert len( summaries ) == 1, "the game must report its result"
    counts = f"{len( strategy )} strategic / {len( battle )} battle records"
    assert strategy, f"no strategic decisions reached the agent ({counts})"
    assert battle, f"no battle decisions reached the agent ({counts})"
    assert any( r["chosen"] is not None for r in strategy )
    assert all( r["act"] is not None for r in battle ), "the random battle policy always decides"
    assert all( "game_end" in r for r in records ), "the outcome must reach every record"


def test_game_survives_the_agent_process_dying():
    """The agent process exits (both pipe ends closed) after the first event: the engine must
    finish the game on the built-in AI instead of being killed by SIGPIPE on the next write
    (regression: exit code -13)."""
    env = dict( os.environ )
    env.update( {
        "FHEROES2_STRATEGY_SERVER": "1",
        "FHEROES2_BATTLE_AGENT": "1",
        "FHEROES2_AUTO_PLAYTEST": "1",
        "FHEROES2_AUTO_PLAYTEST_DAYS": "1",
        "FHEROES2_AUTO_PLAYTEST_MAP": MAP_NAME,
    } )
    proc = subprocess.Popen( [BINARY], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             text=False, env=env, bufsize=0 )
    try:
        assert _LineReader( proc.stdout.fileno() ).read_line() is not None
        proc.stdout.close()
        proc.stdin.close()
        assert proc.wait( timeout=60 ) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
