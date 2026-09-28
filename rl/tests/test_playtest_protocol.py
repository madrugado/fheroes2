"""Integration tests for the seeded autonomous playtest and the strategic event fields
(requires the ./fheroes2 binary and game data; skipped automatically when it is missing)."""

import os
import sys

import pytest

sys.path.insert( 0, os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), ".." ) )

from strategy_env import StrategyEnv  # noqa: E402

REPO_ROOT = os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )
BINARY = os.path.join( REPO_ROOT, "fheroes2" )
MAP_NAME = "Arena.mp2"
DAYS = 2

pytestmark = pytest.mark.skipif( not os.path.exists( BINARY ), reason="fheroes2 binary not built" )


def record_session( seed ):
    """One seeded game with the built-in choices; returns every event the agent saw."""
    env = StrategyEnv( binary=BINARY, map_name=MAP_NAME, days=DAYS, playthroughs=1, seed=seed )
    events = []

    class Recorder:
        def observe_turn( self, ev ):
            events.append( ev )

        def __call__( self, ev ):
            events.append( ev )
            return None

    try:
        summaries = env.run( Recorder() )
    finally:
        env.close()
    return events, summaries


@pytest.fixture( scope="module" )
def sessions():
    return record_session( 5 ), record_session( 5 ), record_session( 6 )


def test_equal_seeds_replay_equal_games( sessions ):
    ( events_a, end_a ), ( events_b, end_b ), ( events_c, _ ) = sessions
    assert events_a and events_a == events_b
    assert end_a == end_b
    assert events_a != events_c, "different seeds should produce different games"


def test_events_carry_the_player_color_of_game_end( sessions ):
    ( events, summaries ), _, _ = sessions
    colors = {result["c"] for result in summaries[-1]["results"]}
    assert colors
    assert {ev["p"] for ev in events} <= colors
    assert all( "p" in ev for ev in events )


def test_game_end_reports_kingdom_stats( sessions ):
    ( _, summaries ), _, _ = sessions
    for result in summaries[-1]["results"]:
        for key in ( "k", "h", "str", "g" ):
            assert isinstance( result[key], int ) and result[key] >= 0
    # Day 2 of Arena.mp2: every player still has a castle.
    assert all( result["k"] >= 1 for result in summaries[-1]["results"] )
