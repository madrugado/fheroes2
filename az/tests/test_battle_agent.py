"""Unit tests for az/battle_agent.py with a scripted fake engine process (no real engine)."""

import io
import json
import os
import random
import sys

sys.path.insert( 0, os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), ".." ) )

import battle_agent  # noqa: E402
from battle_agent import BattleAgentRunner, format_stacks, states_equal  # noqa: E402


class FakeProc:
    """Stands in for subprocess.Popen: stdout replays a fixed script, stdin captures replies."""

    def __init__( self, script ):
        read_fd, write_fd = os.pipe()
        payload = "".join( json.dumps( line ) + "\n" for line in script ) + "not json at all\n"
        os.write( write_fd, payload.encode() )
        os.close( write_fd )  # EOF after the script
        self.stdout = os.fdopen( read_fd, "rb", buffering=0 )
        self.stdin = io.BytesIO()
        self.terminated = False

    def poll( self ):
        return 0

    def wait( self, timeout=None ):
        return 0

    def terminate( self ):
        self.terminated = True

    def kill( self ):
        pass


def make_runner( script, policy="planner", model=None ):
    runner = object.__new__( BattleAgentRunner )  # bypass __init__: no real engine process
    runner.proc = FakeProc( script )
    runner._reader = battle_agent._LineReader( runner.proc.stdout.fileno() )
    runner.policy_name = policy
    runner.model = model
    runner.sims = 4
    runner.map_name = "Arena.mp2"
    runner.rng = random.Random( 1 )
    runner._setup = None
    runner._pending_action = None
    runner._replica_env = None
    runner._replica_tried = False
    runner._replica_synced = False
    runner._replica_desynced = False
    runner._replica_state = None
    runner._records = []
    return runner


def replies( runner ):
    return [json.loads( line ) for line in runner.proc.stdin.getvalue().decode().splitlines()]


class FakeReplicaEnv:
    """Stands in for the headless battle-server replica: scripted state replies."""

    def __init__( self ):
        self.map_names: list[str] = []
        self.script: list[dict] = []
        self.new_battle_kwargs: list[dict] = []
        self.mirrored: list[tuple] = []
        self.closed = False

    def new_battle( self, seed, attacker, defender, tile=-1, world_seed=None, spread_att=None, spread_def=None ):
        self.new_battle_kwargs.append( dict( seed=seed, attacker=attacker, defender=defender, tile=tile,
                                             world_seed=world_seed, spread_att=spread_att, spread_def=spread_def ) )
        return self.script.pop( 0 ) if self.script else None

    def action( self, act, args ):
        self.mirrored.append( ( act, tuple( args ) ) )
        return self.script.pop( 0 ) if self.script else None

    def close( self ):
        self.closed = True


def replica_factory( replica ):
    """BattleEnv stand-in: records the map the runner asked the replica to load."""

    def factory( map_name=None ):
        replica.map_names.append( map_name )
        return replica

    return factory


class FakeMcts:
    """Stands in for mcts.Mcts: always proposes the first legal move of the searched state."""

    def __init__( self, env, policy_value=None, rng=None, root_noise=0.0 ):
        assert isinstance( env, FakeReplicaEnv )
        self.env = env

    def run( self, state, sims ):
        assert sims == 4
        legal = [( m["act"], tuple( m["args"] ) ) for m in state["legal"]]
        return legal, [10.0] + [0.0] * ( len( legal ) - 1 )


BATTLE_START = {
    "ev": "battle_start",
    "bid": 1,
    "seed": 42,
    "tile": 408,
    "wseed": 1000,
    "searchable": 1,
    "att": {"spread": 1, "stacks": [[0, 13, 30], [2, 21, 25]]},
    "def": {"spread": 0, "stacks": [[1, 22, 20]]},
}

STATE_A = {"ev": "state", "bid": 1, "turn": 1, "cur": 3,
           "units": [{"u": 3, "side": "att", "mon": 13, "q": 30, "hpl": 10, "i": 22, "ti": -1, "sp": 2, "shots": 12, "moved": 0}],
           "obstacles": [17],
           "legal": [{"act": 0, "args": [22, 23]}, {"act": 8, "args": [3]}]}

STATE_B = {"ev": "state", "bid": 1, "turn": 1, "cur": 4,
           "units": [{"u": 3, "side": "att", "mon": 13, "q": 30, "hpl": 10, "i": 23, "ti": -1, "sp": 2, "shots": 12, "moved": 1},
                     {"u": 4, "side": "def", "mon": 22, "q": 20, "hpl": 20, "i": 32, "ti": -1, "sp": 2, "shots": 0, "moved": 0}],
           "obstacles": [17],
           "legal": [{"act": 8, "args": [4]}]}

BATTLE_END = {"ev": "battle_end", "bid": 1, "result": "att"}
GAME_END = {"ev": "game_end", "playthrough": 0, "day": 5, "results": [{"c": 0, "s": 2, "d": 5}]}


def test_format_stacks():
    assert format_stacks( [[0, 13, 30], [2, 21, 25]] ) == "0:13x30,2:21x25"


def test_states_equal_ignores_agent_fields():
    assert states_equal( STATE_A, dict( STATE_A, bid=1, searchable=1, ev="state" ) )
    assert not states_equal( STATE_A, dict( STATE_A, cur=99 ) )
    assert not states_equal( STATE_A, dict( STATE_A, obstacles=[] ) )


def test_planner_mode_delegates_every_decision():
    script = [BATTLE_START, STATE_A, STATE_B, BATTLE_END, GAME_END]
    runner = make_runner( script, policy="planner" )
    records = []
    summaries = runner.run( on_record=records.append )

    assert len( summaries ) == 1 and summaries[0]["day"] == 5
    assert replies( runner ) == [{"op": "planner"}, {"op": "planner"}]
    assert len( records ) == 2
    assert records[0]["bid"] == 1 and records[0]["act"] is None
    assert all( r["game_end"]["day"] == 5 for r in records )


def test_random_mode_replies_legal_action():
    script = [BATTLE_START, STATE_A, BATTLE_END, GAME_END]
    runner = make_runner( script, policy="random" )
    runner.run()

    ( reply, ) = replies( runner )
    assert reply["op"] == "action"
    assert {"act": reply["act"], "args": reply["args"]} in [dict( m ) for m in STATE_A["legal"]]


def test_policy_mode_picks_argmax_of_priors():
    class FakeModel:
        def evaluate( self, state ):
            return {0: 0.2, 1: 0.8}, 0.3

    script = [BATTLE_START, STATE_A, BATTLE_END, GAME_END]
    runner = make_runner( script, policy="policy", model=FakeModel() )
    runner.run()

    ( reply, ) = replies( runner )
    assert reply == {"op": "action", "act": 8, "args": [3]}  # prior 0.8 -> second legal move


def test_policy_mode_without_model_delegates():
    script = [BATTLE_START, STATE_A, BATTLE_END, GAME_END]
    runner = make_runner( script, policy="policy", model=None )
    runner.run()

    assert replies( runner ) == [{"op": "planner"}]


def test_mcts_mode_sets_up_replica_and_mirrors_actions(monkeypatch):
    """The replica receives the battle setup, mirrors every real action and feeds the search;
    the search result is applied to the real battle."""
    replica = FakeReplicaEnv()
    replica.script = [STATE_A, STATE_B]  # root state and the state after mirroring the action

    monkeypatch.setattr( battle_agent, "BattleEnv", replica_factory( replica ) )
    monkeypatch.setattr( "mcts.Mcts", FakeMcts )

    script = [BATTLE_START, STATE_A, STATE_B, BATTLE_END, GAME_END]
    runner = make_runner( script, policy="mcts" )
    records = []
    runner.run( on_record=records.append )

    assert replica.map_names == ["Arena.mp2"]  # the replica loads the same map as the game
    ( kwargs, ) = replica.new_battle_kwargs
    assert kwargs["attacker"] == "0:13x30,2:21x25"
    assert kwargs["defender"] == "1:22x20"
    assert kwargs["world_seed"] == 1000 and kwargs["tile"] == 408
    assert kwargs["spread_att"] is True and kwargs["spread_def"] is False

    # The chosen action was mirrored into the replica exactly once.
    assert replica.mirrored == [( 0, ( 22, 23 ) )]

    ( first, second ) = replies( runner )
    assert first == {"op": "action", "act": 0, "args": [22, 23]}
    assert second == {"op": "action", "act": 8, "args": [4]}
    assert records[0]["replica_synced"] is True
    assert records[1]["replica_synced"] is True


def test_mcts_mode_replica_root_mismatch_degrades(monkeypatch):
    """The replica cannot reproduce the battle root state (e.g. a hero battle): the runner must
    not search and delegate to the built-in AI (no model configured)."""
    replica = FakeReplicaEnv()
    replica.script = [dict( STATE_A, cur=999 )]  # does not match the real state

    monkeypatch.setattr( battle_agent, "BattleEnv", replica_factory( replica ) )

    script = [BATTLE_START, STATE_A, BATTLE_END, GAME_END]
    runner = make_runner( script, policy="mcts" )
    runner.run()

    assert runner._replica_synced is False
    assert replies( runner ) == [{"op": "planner"}]


def test_fallback_event_marks_replica_desynced(monkeypatch):
    """The engine rejected our action and used the planner instead: the pending mirror action
    must be dropped and the rest of the battle must not rely on the replica."""
    replica = FakeReplicaEnv()
    replica.script = [STATE_A]

    monkeypatch.setattr( battle_agent, "BattleEnv", replica_factory( replica ) )
    monkeypatch.setattr( "mcts.Mcts", FakeMcts )

    script = [BATTLE_START, STATE_A, {"ev": "battle_fallback", "bid": 1, "what": "invalid action"}, STATE_B, BATTLE_END, GAME_END]
    runner = make_runner( script, policy="mcts" )
    records = []
    runner.run( on_record=records.append )

    assert replies( runner ) == [{"op": "action", "act": 0, "args": [22, 23]}, {"op": "planner"}]
    assert runner._pending_action is None
    assert runner._replica_desynced is True
    assert records[1]["replica_synced"] is False


def test_run_survives_eof_and_close_is_safe():
    runner = make_runner( [] )
    summaries = runner.run()
    assert summaries == []

    runner.close()
    assert runner.proc.terminated is False  # the process already exited: no terminate needed
