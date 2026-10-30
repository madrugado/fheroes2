"""Integration tests for the full strategic layer (hero targets + building + hiring + army budget) against the
real engine (requires ./fheroes2 and game data; skipped otherwise)."""

import os
import random
import sys

import pytest

sys.path.insert( 0, os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), ".." ) )

from strategy_env import StrategyEnv  # noqa: E402
from strategy_policies import NOTHING, RandomPolicy, builtin_policy  # noqa: E402

REPO_ROOT = os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )
BINARY = os.path.join( REPO_ROOT, "fheroes2" )
MAP_NAME = "Battlefi.mp2"  # 6 players: plenty of building and hiring within a week
DAYS = 7
SEED = 11

pytestmark = pytest.mark.skipif( not os.path.exists( BINARY ), reason="fheroes2 binary not built" )


def play( policy ):
    env = StrategyEnv( binary=BINARY, map_name=MAP_NAME, days=DAYS, playthroughs=1, seed=SEED )
    records = []
    try:
        summaries = env.run( policy, on_decision=records.append )
    finally:
        env.close()
    return summaries[-1], records


def signature( records ):
    """What the agent was asked, in order (the choices themselves excluded)."""
    return [( r["kind"], r["t"], r["p"], r.get( "castle" ), repr( r["cands"] ) ) for r in records]


@pytest.fixture( scope="module" )
def builtin_game():
    return play( builtin_policy )


def test_every_kind_of_query_arrives( builtin_game ):
    _, records = builtin_game
    kinds = {r["kind"] for r in records}
    assert kinds == {"target", "build", "hire", "army"}
    builds = [r for r in records if r["kind"] == "build"]
    # The built-in AI decided (skip) and the engine reported what it built.
    assert all( r["chosen"] is None and r["src"] == "builtin" for r in builds )
    assert any( r["result"] != 0 for r in builds )


def test_explicit_builtin_choices_replay_the_builtin_game( builtin_game ):
    """Answering every build/hire query with the built-in AI's own choice must give exactly the
    game where the agent always skips (the choice path of the engine applies it identically)."""
    game_end, records = builtin_game
    built = {( r["t"], r["castle"] ): r["result"] for r in records if r["kind"] == "build"}

    class Echo:
        def __call__( self, decision ):
            return None

        def build( self, ev ):
            building = built[( ev["t"], ev["castle"] )]
            if building == 0:
                return NOTHING
            return next( c for c in ev["cands"] if c["b"] == building )

        def hire( self, ev ):
            return ev["cands"][ev["bi"]] if ev["bi"] >= 0 else NOTHING

        def army( self, ev ):
            return 100  # the full budget is exactly the built-in army purchase

    echo_end, echo_records = play( Echo() )
    assert signature( echo_records ) == signature( records )
    assert echo_end == game_end
    assert any( r["chosen"] not in ( None, 0 ) for r in echo_records if r["kind"] == "build" )


def test_agent_choices_are_applied():
    _, records = play( RandomPolicy( random.Random( 1 ) ) )

    builds = [r for r in records if r["kind"] == "build" and r["chosen"] not in ( None, 0 )]
    assert builds
    assert all( r["result"] == r["chosen"] and r["src"] == "agent" for r in builds )

    # Army budgets below 100% leave resources unspent.
    armies = [r for r in records if r["kind"] == "army"]
    assert {0, 50} & {r["chosen"] for r in armies}

    # The random policy hires where the built-in AI would not (bi == -1): hiring is the agent's call.
    assert any( r["kind"] == "hire" and r["bi"] == -1 and r["chosen"] is not None and r["chosen"] >= 0 for r in records )


def test_learned_policy_drives_every_kind_in_a_real_game( tmp_path ):
    """The learned policy end to end: features are built from real engine events, the model
    answers all four query kinds, and the engine applies the answers."""
    import json

    sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )
    import strategy_model as sm
    from strategy_policies import LearnedPolicy
    from test_strategy_model import synthetic_records

    model = sm.select_and_train( synthetic_records(), margins=[-1e9], archs=["ridge"], rule="all", log=lambda *_: None )
    path = tmp_path / "model.json"
    path.write_text( json.dumps( model ) )

    _, records = play( LearnedPolicy( str( path ) ) )

    answered = {r["kind"] for r in records if r["chosen"] is not None}
    assert answered == {"target", "build", "hire", "army"}
    builds = [r for r in records if r["kind"] == "build" and r["chosen"] not in ( None, 0 )]
    assert builds and all( r["result"] == r["chosen"] and r["src"] == "agent" for r in builds )


def test_explicit_answers_equal_to_the_build_hint_replay_the_builtin_game_on_a_water_map():
    """2kings has water: the build probe cannot see the boat purchase, so its hint `bi` can name a building the real
    development no longer affords after a boat (or nothing where it buys one). An explicit answer equal to the hint
    runs the real development (2026-10-30: before, explicit built-in answers won 21 of 80 2kings games instead of 40)."""
    from strategy_net import answer_of, builtin_option, query_options

    class Explicit:
        def __call__( self, ev ):
            return None

        def build( self, ev ):
            options = query_options( "build", ev, "student" )
            index = builtin_option( "build", options, ev )
            return None if index is None else answer_of( options[index] )

    def game( policy, seed ):
        env = StrategyEnv( binary=BINARY, map_name="2kings.mp2", days=150, playthroughs=1, seed=seed, rng_streams=True )
        builds = []
        try:
            summary = env.run( policy, on_decision=lambda r: builds.append( ( r["t"], r["p"], r.get( "castle" ), r.get( "result" ) ) )
                               if r["kind"] == "build" else None )[-1]
        finally:
            env.close()
        return summary, builds

    for seed in ( 2, 3 ):  # both diverged before the fix (a boat bought instead of a hinted building, a moat not hinted)
        explicit, skip = game( Explicit(), seed ), game( builtin_policy, seed )
        assert explicit[1] == skip[1]
        assert explicit[0].get( "results" ) == skip[0].get( "results" )
