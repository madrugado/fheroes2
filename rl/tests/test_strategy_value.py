"""strategy_value.py: states of a trajectory, the value head, the exploring policy in a real game."""

import os
import random
import sys

import pytest

sys.path.insert( 0, os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), ".." ) )

torch = pytest.importorskip( "torch" )
pytest.importorskip( "transformers" )

import strategy_value  # noqa: E402
from strategy_net import value_tokens  # noqa: E402
from transformer_model import AzBattleTransformer, strategic_value  # noqa: E402

REPO_ROOT = os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )
BINARY = os.path.join( REPO_ROOT, "fheroes2" )

CONTEXT = {"ev": "turn_context", "p": "Blue", "res": [5, 0, 5, 0, 0, 0, 3000], "castles": [{"i": 100}],
           "heroes": [{"id": 7, "mp": 900, "mmp": 1200, "str": 400.0}]}
ARMY = {"ev": "army", "p": "Blue", "castle": 100, "reason": "visit", "garrison": 50.0, "hero": 400.0,
        "offer": [{"mon": 1, "avail": 10, "n": 10, "str": 11.0}]}


def trajectory():
    days = [dict( CONTEXT, t=day ) for day in ( 1, 2, 3 )]
    decisions = [{"kind": "army", "event": dict( ARMY, t=day ), "ctx": day - 1, "answer": 1} for day in ( 1, 1, 2, 3 )]
    return {"seed": 1, "map": "m", "color": "Blue", "final": 1.5, "days": days, "decisions": decisions}


def test_a_state_sees_only_the_previous_days_and_their_answers():
    context, history = strategy_value.state_history( trajectory(), 2 )
    assert context["t"] == 3
    assert [d["t"] for d in history["days"]] == [1, 2]
    assert [d["event"]["t"] for d in history["decisions"]] == [1, 1, 2]  # not today's
    assert history["decisions"][0]["context"]["t"] == 1
    assert len( strategy_value.states_of( trajectory() ) ) == 3


def test_value_head_is_bounded_and_independent_of_padding():
    model = AzBattleTransformer().eval()
    short = value_tokens( dict( CONTEXT, t=1 ), [] )
    context, history = strategy_value.state_history( trajectory(), 2 )
    long = value_tokens( context, [], history )
    with torch.no_grad():
        alone = strategic_value( model, [short] )
        batch = strategic_value( model, [long, short] )
    assert batch.shape == ( 2, ) and bool( ( batch.abs() <= 2.0 ).all() )
    assert torch.allclose( batch[1], alone[0], atol=1e-5 )


@pytest.mark.skipif( not os.path.exists( BINARY ), reason="fheroes2 binary not built" )
def test_exploring_policy_records_both_players_and_replays_the_builtin_game_at_epsilon_zero():
    from strategy_env import StrategyEnv

    def run( epsilon ):
        policy = strategy_value.ExploringPolicy( epsilon, random.Random( 3 ) )
        env = StrategyEnv( binary=BINARY, map_name="2kings.mp2", days=6, playthroughs=1, seed=11 )
        try:
            end = env.run( policy )[-1]
        finally:
            env.close()
        return policy, end

    class Silent:
        def __call__( self, ev ):
            return None

    env = StrategyEnv( binary=BINARY, map_name="2kings.mp2", days=6, playthroughs=1, seed=11 )
    try:
        builtin_end = env.run( Silent() )[-1]
    finally:
        env.close()
    greedy, greedy_end = run( 0.0 )
    assert greedy_end["results"] == builtin_end["results"]
    assert set( greedy.days ) == {"Blue", "Green"} and all( len( d ) == 6 for d in greedy.days.values() )
    assert all( 0 <= d["ctx"] < 6 for decisions in greedy.decisions.values() for d in decisions )
    explored, _ = run( 1.0 )
    assert sum( len( d ) for d in explored.decisions.values() ) > 0
