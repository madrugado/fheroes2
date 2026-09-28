"""Unit tests for rl/strategy_bench.py (pure functions) and the per-color policy wrapper."""

import os
import sys

sys.path.insert( 0, os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), ".." ) )

from strategy_bench import bootstrap_ci, compare, player_stats, sign_test_p, summarize  # noqa: E402
from strategy_policies import ForColor, greedy_policy  # noqa: E402


def game_end( **players ):
    return {"ev": "game_end", "day": 14, "results": [{"c": color, **fields} for color, fields in players.items()]}


def test_player_stats_maps_states_to_outcomes():
    stats = player_stats( game_end( Blue={"s": "0", "k": 2, "h": 3, "str": 900, "g": 5000},
                                    Red={"s": "1", "d": 9},
                                    Green={"s": "2", "k": 1, "h": 1, "str": 100, "g": 10} ) )
    assert stats["Blue"] == {"outcome": 1, "k": 2, "h": 3, "str": 900, "g": 5000}
    assert stats["Red"]["outcome"] == -1 and stats["Red"]["k"] == 0  # eliminated: no kingdom stats
    assert stats["Green"]["outcome"] == 0


def test_compare_is_lexicographic_outcome_castles_strength():
    base = {"outcome": 0, "k": 1, "h": 1, "str": 500, "g": 100}
    assert compare( base, dict( base, str=600 ) )["verdict"] == "better"
    assert compare( base, dict( base, k=2, str=10 ) )["verdict"] == "better"  # castles outrank strength
    assert compare( base, dict( base, outcome=-1, k=5 ) )["verdict"] == "worse"  # outcome outranks all
    diff = compare( base, dict( base, g=50 ) )
    assert diff["verdict"] == "equal" and diff["g"] == -50  # gold is reported, not ranked


def test_summarize_counts_and_means():
    pairs = [
        {"verdict": "better", "outcome": 0, "k": 1, "h": 0, "str": 100, "g": 0, "changed": True},
        {"verdict": "equal", "outcome": 0, "k": 0, "h": 0, "str": 0, "g": 0, "changed": False},
    ]
    summary = summarize( pairs )
    assert ( summary["better"], summary["equal"], summary["worse"] ) == ( 1, 1, 0 )
    assert summary["mean_d_str"] == 50.0 and summary["mean_d_k"] == 0.5
    assert summary["changed_games"] == 1
    assert summarize( [] )["pairs"] == 0


def test_for_color_only_controls_one_player():
    seen = []

    class Policy:
        def observe_turn( self, ev ):
            seen.append( ev["p"] )

        def __call__( self, ev ):
            return greedy_policy( ev )

    policy = ForColor( Policy(), "Blue" )
    decision = {"cands": [{"i": 1, "v": 1.0, "d": 1}, {"i": 2, "v": 5.0, "d": 1}]}

    policy.observe_turn( {"p": "Blue"} )
    policy.observe_turn( {"p": "Red"} )
    assert seen == ["Blue"]
    assert policy( dict( decision, p="Blue" ) )["i"] == 2
    assert policy( dict( decision, p="Red" ) ) is None


def test_sign_test_p_values():
    assert sign_test_p( 0, 0 ) == 1.0
    assert sign_test_p( 5, 5 ) == 1.0
    assert abs( sign_test_p( 10, 0 ) - 2 / 1024 ) < 1e-12  # all 10 non-ties one way
    assert sign_test_p( 16, 15 ) > 0.8


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    values = [float( v ) for v in range( -50, 51 )]
    low, high = bootstrap_ci( values )
    assert low < 0.0 < high
    assert bootstrap_ci( values ) == ( low, high )
    assert bootstrap_ci( [3.0, 3.0] ) == ( 3.0, 3.0 )


def test_is_override_per_kind():
    from strategy_bench import is_override

    target = {"kind": "target", "cands": [{"i": 1}, {"i": 2}]}
    assert not is_override( dict( target, chosen=None ) )
    assert not is_override( dict( target, chosen=1 ) ) and is_override( dict( target, chosen=2 ) )
    assert not is_override( {"kind": "hire", "bi": 0, "chosen": 0} ) and is_override( {"kind": "hire", "bi": -1, "chosen": 0} )
    assert is_override( {"kind": "hire", "bi": 1, "chosen": -1} )
    assert is_override( {"kind": "build", "chosen": 0} ) and not is_override( {"kind": "build", "chosen": None} )
    assert is_override( {"kind": "army", "chosen": 50} ) and not is_override( {"kind": "army", "chosen": 100} )
