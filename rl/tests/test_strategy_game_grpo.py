"""Unit tests for rl/strategy_game_grpo.py (no engine)."""

import os
import sys

sys.path.insert( 0, os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), ".." ) )

from strategy_game_grpo import group_records  # noqa: E402


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
