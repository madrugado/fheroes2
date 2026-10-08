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


def record_session( seed, rng_streams=None, map_name=MAP_NAME, days=DAYS, reseed=None ):
    """One seeded game with the built-in choices; returns every event the agent saw."""
    env = StrategyEnv( binary=BINARY, map_name=map_name, days=days, playthroughs=1, seed=seed, rng_streams=rng_streams, reseed=reseed )
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


def test_separate_random_streams_are_reproducible_and_switchable():
    """FHEROES2_RNG_STREAMS: equal seeds still replay equal games; the switch changes the game (the
    dice are re-keyed per turn); off by default (the plain session equals an explicit off)."""
    plain, _ = record_session( 5, rng_streams=False, map_name="2kings.mp2", days=8 )
    inherited, _ = record_session( 5, map_name="2kings.mp2", days=8 )
    streams_a, end_a = record_session( 5, rng_streams=True, map_name="2kings.mp2", days=8 )
    streams_b, end_b = record_session( 5, rng_streams=True, map_name="2kings.mp2", days=8 )
    assert plain == inherited
    assert streams_a and streams_a == streams_b and end_a == end_b
    assert streams_a != plain


def test_reseed_with_streams_keeps_the_game_before_its_day():
    """FHEROES2_RESEED still works with separate streams: the game is identical before the salted day
    (turn contexts of days < 5) and differs afterwards."""
    base, _ = record_session( 5, rng_streams=True, map_name="2kings.mp2", days=10 )
    salted, _ = record_session( 5, rng_streams=True, map_name="2kings.mp2", days=10, reseed=( 5, 3 ) )
    before = [ev for ev in base if ev.get( "t", 0 ) < 5]
    assert before and before == [ev for ev in salted if ev.get( "t", 0 ) < 5]
    assert base != salted


def planned_roles( plan ):
    """(color, day, hero id, role, creatures) of every own hero in the turn contexts of a seeded game."""
    env = StrategyEnv( binary=BINARY, map_name="2kings.mp2", days=12, playthroughs=1, seed=7, plan=plan )
    rows = []

    class Recorder:
        def observe_turn( self, ev ):
            for hero in ev.get( "heroes" ) or []:
                rows.append( ( ev["p"], ev["t"], hero["id"], hero.get( "role" ), sum( stack[1] for stack in hero.get( "army" ) or [] ) ) )

        def __call__( self, ev ):
            return None

    try:
        env.run( Recorder() )
    finally:
        env.close()
    return rows


def test_plan_makes_one_champion_and_minimal_secondaries_for_its_color_only():
    """FHEROES2_PLAN (ai_plan.h): champion=1 gives the planned color one main hero even with two
    heroes (upstream: only with more than three), secondary_min=1 leaves the others with tiny armies;
    the other color and a game without the plan keep the built-in roles. Equal seeds replay equally."""
    plain = planned_roles( None )
    planned = planned_roles( "color=Blue,champion=1,secondary_min=1" )
    assert not any( role == 4 for _, _, _, role, _ in plain )
    blue_days = {day for color, day, _, _, _ in planned if color == "Blue"}
    champion_days = {day for color, day, _, role, _ in planned if color == "Blue" and role == 4}
    assert champion_days and len( champion_days ) >= len( blue_days ) - 2  # the role is visible from the next turn on
    assert not any( role == 4 for color, _, _, role, _ in planned if color != "Blue" )
    champions = {hero for color, _, hero, role, _ in planned if color == "Blue" and role == 4}
    assert len( champions ) == 1  # kept while the hero lives
    assert planned == planned_roles( "color=Blue,champion=1,secondary_min=1" )


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
