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


def test_game_end_is_reported_to_a_battle_only_agent( session ):
    """game_end used to be emitted only with the strategic channel enabled."""
    kinds = [e.get( "ev" ) for e in session["events"]]
    assert kinds.count( "game_end" ) == 1
    assert kinds[-1] == "game_end"
    assert "decision" not in kinds, "the strategic channel must stay off"


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
            assert setup[side]["c"] in ( 0, 1, 2, 4, 8, 16, 32 )
            for slot, mon, count in setup[side]["stacks"]:
                assert slot >= 0 and mon > 0 and count > 0
            if "hid" in setup[side]:
                # The commander hero: its save-game serialization, hex-encoded.
                assert setup[side]["hid"] >= 0
                bytes.fromhex( setup[side]["hero"] )

    # AI heroes attack on this map: at least one setup carries a commander.
    assert any( "hid" in setup[side] for setup in starts for side in ( "att", "def" ) )

    # Decision queries must reference the battle id of the setup and carry legal moves.
    bids = {e["bid"] for e in starts}
    assert states and all( e["bid"] in bids for e in states )
    assert all( e.get( "legal" ) for e in states )


def test_mcts_replica_stays_synced_in_hero_battles():
    """The replica rebuilds every battle from battle_start — including the commander heroes,
    restored from their save-game serialization — and mirrors every move; in all searchable
    battles (no castle/town on the tile) it must stay in sync with the real battle. Before the
    commander replication, hero battles desynced after the first hit (skills, morale, luck)."""
    import sys

    sys.path.insert( 0, os.path.join( REPO_ROOT, "az" ) )
    import battle_agent

    runner = battle_agent.BattleAgentRunner( BINARY, MAP_NAME, 8, 1, "mcts", sims=2,
                                             extra_env={"FHEROES2_AUTO_PLAYTEST_SEED": "7"} )
    records: list[dict] = []
    heroes_seen: list[int] = []
    original_start = runner._replica_start

    def replica_start( state ):
        setup = runner._setup or {}
        heroes_seen.append( sum( 1 for side in ( "att", "def" ) if "hid" in setup.get( side, {} ) ) )
        return original_start( state )

    runner._replica_start = replica_start
    try:
        runner.run( on_record=records.append )
    finally:
        runner.close()

    searchable = [r for r in records if r["searchable"] == 1 and r["turn"] <= runner.max_battle_turns]
    assert searchable, "no searchable battle decisions"
    assert any( count > 0 for count in heroes_seen ), "no hero battle was replicated"
    desynced = [r for r in searchable if r["replica_synced"] is not True]
    assert not desynced, f"{len( desynced )}/{len( searchable )} searchable decisions lost the replica"
