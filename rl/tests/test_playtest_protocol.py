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


def test_turn_context_shows_what_the_player_sees():
    """Own heroes/castles in full (hero and castle screens), the day of the week, and castles of
    other owners outside the fog (quick info); never the player's own castles among them."""
    env = StrategyEnv( binary=BINARY, map_name="2kings.mp2", days=12, playthroughs=1, seed=3 )
    contexts = []

    class Recorder:
        def observe_turn( self, ev ):
            contexts.append( ev )

        def __call__( self, ev ):
            return None

    try:
        env.run( Recorder() )
    finally:
        env.close()
    assert contexts
    for ctx in contexts:
        assert 1 <= ctx["wd"] <= 7 and ctx["wk"] >= 1 and len( ctx["res"] ) == 7
        for hero in ctx["heroes"]:
            assert len( hero["sk"] ) == 14 and hero["lvl"] >= 1 and isinstance( hero["art"], list )
            assert hero["army"] and all( len( stack ) == 7 and stack[1] > 0 and stack[2] > 0 for stack in hero["army"] )
        for castle in ctx["castles"]:
            assert castle["b"] > 0 and len( castle["dw"] ) == 6 and all( len( d ) == 7 for d in castle["dw"] )
            assert castle["dw"][0][0] > 0  # every castle has its first dwelling
        own = {castle["i"] for castle in ctx["castles"]}
        for castle in ctx["rcastles"]:
            assert castle["c"] != ctx["p"] and castle["i"] not in own and castle["vis"] in ( 0, 1, 2, 3 )
            assert castle["vis"] > 0 or castle["army"] == []  # no Thieves' Guild: the defenders are unknown
    assert any( ctx["rcastles"] for ctx in contexts )
    assert any( ctx["wd"] == 1 for ctx in contexts ) and any( ctx["wd"] == 7 for ctx in contexts )
