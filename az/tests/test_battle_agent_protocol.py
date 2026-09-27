"""Integration test for the real-battle agent protocol (requires the ./fheroes2 binary and
game data; skipped automatically when the binary is missing).

One short autonomous playtest with FHEROES2_BATTLE_AGENT=1 drives every battle decision
from Python: agent-chosen actions (first battle) and built-in AI delegation (later
battles) must both be accepted by the engine — no battle_fallback events, every battle
completed, and the setup events must carry the replica-reconstruction data.
"""

import json
import os
import select
import subprocess
import time

import pytest

REPO_ROOT = os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )
BINARY = os.path.join( REPO_ROOT, "fheroes2" )
MAP_NAME = "Arena.mp2"
DAYS = 5

pytestmark = pytest.mark.skipif( not os.path.exists( BINARY ), reason="fheroes2 binary not built" )


def run_agent_session( decide ):
    """Runs one playtest; `decide(state_event)` must return the reply operation dict.

    Returns (events, replies_sent, fallbacks)."""
    env = dict( os.environ )
    env.update( {
        "FHEROES2_BATTLE_AGENT": "1",
        "FHEROES2_AUTO_PLAYTEST": "1",
        "FHEROES2_AUTO_PLAYTEST_DAYS": str( DAYS ),
        "FHEROES2_AUTO_PLAYTEST_MAP": MAP_NAME,
    } )

    proc = subprocess.Popen( [BINARY], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, text=False, env=env, bufsize=0 )

    events: list[dict] = []
    replies_sent = 0
    fallbacks = 0
    buffer = b""
    fd = proc.stdout.fileno()

    def read_line( timeout ):
        # Battle states exceed the pipe buffer, so the line is assembled from raw timed
        # reads instead of a buffered readline() (see engine_bridge._read).
        nonlocal buffer
        deadline = time.monotonic() + timeout
        while b"\n" not in buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError( "engine did not produce output within 60 seconds" )
            ready, _, _ = select.select( [fd], [], [], min( remaining, 1.0 ) )
            if not ready:
                continue
            chunk = os.read( fd, 65536 )
            if not chunk:
                line, buffer = buffer, b""
                return line.decode() if line.strip() else None
            buffer += chunk
        line, buffer = buffer.split( b"\n", 1 )
        return line.decode()

    try:
        while True:
            line = read_line( 60.0 )
            if line is None:
                break

            try:
                ev = json.loads( line )
            except json.JSONDecodeError:
                continue

            events.append( ev )

            if ev.get( "ev" ) == "state" and "bid" in ev:
                reply = decide( ev )
                proc.stdin.write( ( json.dumps( reply, separators=( ",", ":" ) ) + "\n" ).encode() )
                proc.stdin.flush()
                replies_sent += 1
            elif ev.get( "ev" ) == "battle_fallback":
                fallbacks += 1
    finally:
        if proc.poll() is None:
            try:
                proc.stdin.close()
            except Exception:
                pass
            try:
                proc.wait( timeout=10 )
            except subprocess.TimeoutExpired:
                proc.kill()

    return events, replies_sent, fallbacks


@pytest.fixture( scope="module" )
def session():
    """One shared engine session. During the first battle every second decision is made by
    the agent (first legal move) and the others are delegated to the built-in AI, so both
    reply modes are exercised."""
    events: list = []
    counts = {"action": 0, "planner": 0}
    first_bid: list = []

    def decide( state ):
        if not first_bid:
            first_bid.append( state["bid"] )
        if state["bid"] == first_bid[0] and counts["action"] <= counts["planner"]:
            counts["action"] += 1
            return {"op": "action", **state["legal"][0]}
        counts["planner"] += 1
        return {"op": "planner"}

    events, replies_sent, fallbacks = run_agent_session( decide )
    return {"events": events, "replies": replies_sent, "fallbacks": fallbacks, "counts": counts}


def battle_stats( events ):
    starts = [e for e in events if e.get( "ev" ) == "battle_start"]
    ends = [e for e in events if e.get( "ev" ) == "battle_end"]
    states = [e for e in events if e.get( "ev" ) == "state" and "bid" in e]
    return starts, states, ends


def test_agent_decisions_are_accepted( session ):
    events = session["events"]
    starts, states, ends = battle_stats( events )

    # The playtest map must produce at least one agent-driven battle within the day limit.
    assert len( starts ) >= 1, "no battles happened during the playtest"
    assert len( starts ) == len( ends ), "every battle must report its result"
    assert session["fallbacks"] == 0, "the engine rejected an agent decision"
    assert session["replies"] == len( states ), "every decision query must be answered"

    # Agent-chosen actions must have been exercised at least once.
    assert session["counts"]["action"] > 0, "no agent-chosen action was sent"


def test_battle_setup_carries_replication_data( session ):
    """The battle_start event must contain everything needed to reconstruct the battle in a
    headless replica (see az/battle_agent.py)."""
    starts, states, _ = battle_stats( session["events"] )
    assert starts, "no battles happened"

    for setup in starts:
        assert setup["seed"] > 0 and setup["tile"] >= 0 and setup["wseed"] > 0
        assert setup["searchable"] in ( 0, 1 )
        for side in ( "att", "def" ):
            assert setup[side]["spread"] in ( 0, 1 )
            for slot, mon, count in setup[side]["stacks"]:
                assert slot >= 0 and mon > 0 and count > 0

    # Decision queries must reference the battle id of the setup and carry legal moves.
    bids = {e["bid"] for e in starts}
    assert states and all( e["bid"] in bids for e in states )
    assert all( e.get( "legal" ) for e in states )
