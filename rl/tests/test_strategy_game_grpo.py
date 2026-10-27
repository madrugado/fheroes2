"""Unit tests for rl/strategy_game_grpo.py (no engine)."""

import os
import random
import sys

sys.path.insert( 0, os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), ".." ) )

import pytest  # noqa: E402

from strategy_game_grpo import SampleBranch, group_records, per_kind  # noqa: E402


class FakePolicy:
    """A strategic policy with fixed probabilities, enough for SampleBranch's draws."""

    view = "full"

    def reset( self ) -> None:
        pass

    def observe_turn( self, turn_context: dict ) -> None:
        pass

    def probabilities( self, kind: str, event: dict ) -> list[float]:
        return [0.7, 0.2, 0.1]

    def record( self, kind: str, event: dict, index: int | None ) -> None:
        pass

    def history( self, color: str ) -> dict:
        return {"days": [], "decisions": []}


ARMY = {"ev": "army", "t": 3, "p": "Blue", "h": 1, "res": [0] * 7, "cands": []}
TARGET = {"ev": "decision", "t": 3, "p": "Blue", "h": 1, "cands": [{"i": 10}, {"i": 20}, {"i": 30}]}


def test_group_records_push_every_answer_by_its_game_advantage_with_the_rebuilt_history():
    days = [{"t": 1}, {"t": 2}]
    first = {"kind": "army", "event": {"p": "Blue", "n": 1}, "ctx": 0, "answer": 2, "p": 0.6}
    second = {"kind": "target", "event": {"p": "Blue", "n": 2}, "ctx": 1, "answer": 0, "p": 0.9}
    group = {"seed": 7, "map": "m", "games": [
        {"k": 0, "reward": 1.0, "days": days, "answers": [first, second]},
        {"k": 1, "reward": -1.0, "days": days, "answers": [dict( first, answer=1 )]},
    ]}
    records = group_records( group, eps=0.0 )
    assert len( records ) == 3
    a, b, c = records
    assert a["adv"] == {2: 1.0} and b["adv"] == {0: 1.0} and c["adv"] == {1: -1.0}
    assert a["context"] == {"t": 1} and a["history"] == {"days": [], "decisions": []}
    assert b["context"] == {"t": 2} and b["history"]["days"] == [{"t": 1}]
    assert b["history"]["decisions"] == [{"kind": "army", "event": first["event"], "context": {"t": 1}, "answer": 2}]
    assert group_records( {"seed": 1, "map": "m", "games": [dict( group["games"][0] ), dict( group["games"][0] )]} ) == []


def test_per_kind_parses_kind_value_pairs():
    assert per_kind( "target=1,build=3", "x" ) == {"target": 1.0, "build": 3.0}
    assert per_kind( "", "x" ) == {} and per_kind( " army =4 ", "x" ) == {"army": 4.0}
    with pytest.raises( SystemExit ):
        per_kind( "target", "x" )


def test_sample_branch_stores_the_temperature_of_every_answer():
    """Per-kind T (idea a): army drawn almost greedily at T=0.01, targets at the global T; every answer keeps its T."""
    branch = SampleBranch( FakePolicy(), "Blue", 5.0, random.Random( 1 ), {"army": 0.01}, {} )
    for _ in range( 40 ):
        branch._answer( "army", dict( ARMY ) )
        branch._answer( "target", dict( TARGET ) )
    army_answers = [a["answer"] for a in branch.answers if a["kind"] == "army"]
    assert all( a["T"] == 0.01 for a in branch.answers if a["kind"] == "army" )
    assert all( a["T"] == 5.0 for a in branch.answers if a["kind"] == "target" )
    assert army_answers.count( 0 ) / len( army_answers ) > 0.9  # T = 0.01 stays at the greedy option
    assert not any( a.get( "eps" ) for a in branch.answers )


def test_sample_branch_epsilon_answers_are_uniform_and_marked():
    """--kind-epsilon 1.0: every army answer is uniform-random among the options and marked 'eps'."""
    branch = SampleBranch( FakePolicy(), "Blue", 1.0, random.Random( 3 ), {}, {"army": 1.0} )
    for _ in range( 60 ):
        branch._answer( "army", dict( ARMY ) )
    answers = [a["answer"] for a in branch.answers]
    assert sorted( set( answers ) ) == [0, 1, 2]  # uniform, not the tempered draw
    assert all( a.get( "eps" ) == 1 and a["T"] == 1.0 for a in branch.answers )


def test_group_records_zero_the_epsilon_answers_and_keep_the_others():
    """An 'eps' answer is off-policy (advantage 0) and must not zero the following answers of its game."""
    days = [{"t": 1}]
    group = {"seed": 7, "map": "m", "games": [
        {"reward": 1.0, "days": days, "answers": [{"kind": "army", "event": ARMY, "ctx": 0, "answer": 2, "eps": 1, "T": 1.0}]},
        {"reward": -1.0, "days": days, "answers": [{"kind": "army", "event": ARMY, "ctx": 0, "answer": 1, "T": 4.0}]},
        {"reward": 1.0, "days": days, "answers": [
            {"kind": "army", "event": ARMY, "ctx": 0, "answer": 2, "eps": 1, "T": 1.0},
            {"kind": "target", "event": TARGET, "ctx": 0, "answer": 1}]},
        {"reward": -1.0, "days": days, "answers": [{"kind": "army", "event": ARMY, "ctx": 0, "answer": 0}]},
    ]}
    records = group_records( group, eps=0.0 )
    assert [r["adv"] for r in records] == [{2: 0.0}, {1: -1.0}, {2: 0.0}, {1: 1.0}, {0: -1.0}]
    assert records[0]["eps"] == 1 and records[0]["T"] == 1.0 and records[1]["T"] == 4.0
    assert "eps" not in records[1] and "T" not in records[3]
