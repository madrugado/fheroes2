"""Unit tests for rl/strategy_game_grpo.py (no engine)."""

import os
import sys

sys.path.insert( 0, os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), ".." ) )

from strategy_game_grpo import group_records  # noqa: E402


def test_group_records_push_every_exploration_answer_by_its_game_advantage():
    answer = {"kind": "army", "event": {"p": "Blue"}, "context": {"t": 3}, "history": {"days": [], "decisions": []}, "own": 2}
    group = {"seed": 7, "map": "m", "games": [
        {"k": 0, "reward": 0.0, "explored": []},
        {"k": 1, "reward": 1.0, "explored": [dict( answer, answer=0 ), dict( answer, answer=1 )]},
        {"k": 2, "reward": -1.0, "explored": [dict( answer, answer=1 )]},
    ]}
    records = group_records( group, eps=0.0 )
    assert len( records ) == 3
    good, also_good, bad = records
    assert good["adv"] == {0: good["adv"][0]} and good["adv"][0] > 0 and also_good["adv"][1] > 0 and bad["adv"][1] < 0
    assert abs( good["adv"][0] + bad["adv"][1] ) < 1e-9 and good["own"] == 2 and good["seed"] == 7
    assert group_records( {"seed": 1, "map": "m", "games": [group["games"][0]]} ) == []
