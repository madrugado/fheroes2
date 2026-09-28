"""Unit tests for rl/game_agent.py: one reader loop serving the strategic and the battle channel
over a scripted fake engine (no real engine)."""

import json
import os
import sys

sys.path.insert( 0, os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), ".." ) )
sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

from game_agent import GameAgent  # noqa: E402
from strategy_policies import TempoPolicy  # noqa: E402
from test_battle_agent import BATTLE_END, BATTLE_START, STATE_A, make_runner, replies  # noqa: E402

TURN_CONTEXT = {"ev": "turn_context", "t": 1, "heroes": [{"id": 5, "i": 0, "mp": 1000, "mmp": 1000, "str": 1.0}]}
DECISION = {"ev": "decision", "t": 1, "h": 5, "from": 0,
            "cands": [{"i": 10, "obj": 0, "v": 3.0, "d": 100}, {"i": 20, "obj": 0, "v": 1.0, "d": 100}]}
GAME_END = {"ev": "game_end", "playthrough": 0, "day": 2, "results": [{"c": "Blue", "s": "2", "d": 2}]}


def make_agent( script, strategy_policy, battle_policy="planner" ):
    agent = make_runner( script, policy=battle_policy )  # a BattleAgentRunner with a fake engine
    agent.__class__ = GameAgent  # bypass __init__ (no real engine) and add the strategic state
    agent.strategy_policy = strategy_policy
    agent._strategy_records = []
    agent._on_strategy_record = None
    return agent


def test_interleaved_channels_are_dispatched_in_engine_order():
    """Hero decision -> the battle it starts -> the next hero decision: every query gets exactly
    one reply of the right protocol, in order."""
    script = [TURN_CONTEXT, DECISION, BATTLE_START, STATE_A, BATTLE_END, DECISION, GAME_END]
    agent = make_agent( script, TempoPolicy() )
    records = []

    summaries = agent.run( on_record=records.append )

    assert replies( agent ) == [
        {"op": "pick", "h": 5, "i": 10},
        {"op": "planner"},
        {"op": "pick", "h": 5, "i": 10},
    ]
    assert [r.get( "kind", "battle" ) for r in records] == ["target", "battle", "target"]
    assert len( summaries ) == 1 and summaries[0]["day"] == 2

    # The outcome reaches the records of BOTH channels.
    outcome = {"day": 2, "results": GAME_END["results"]}
    assert all( r["game_end"] == outcome for r in records )


def test_declined_strategic_decision_is_skipped():
    agent = make_agent( [DECISION, GAME_END], lambda ev: None )
    records = []
    agent.run( on_record=records.append )

    assert replies( agent ) == [{"op": "skip"}]
    assert records[0]["chosen"] is None


def test_turn_context_reaches_context_aware_policy():
    seen = []

    class Policy:
        def observe_turn( self, ev ):
            seen.append( ev["t"] )

        def __call__( self, ev ):
            return ev["cands"][1]

    agent = make_agent( [TURN_CONTEXT, DECISION, GAME_END], Policy() )
    agent.run()

    assert seen == [1]
    assert replies( agent ) == [{"op": "pick", "h": 5, "i": 20}]


def test_battle_only_events_leave_strategy_records_empty():
    agent = make_agent( [BATTLE_START, STATE_A, BATTLE_END, GAME_END], TempoPolicy() )
    records = []
    agent.run( on_record=records.append )

    assert [json.dumps( r.get( "kind", "battle" ) ) for r in records] == ['"battle"']
    assert agent._strategy_records == []
