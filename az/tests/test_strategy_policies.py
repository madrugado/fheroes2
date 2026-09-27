"""Unit tests for az/strategy_policies.py (pure functions, no engine)."""

import os
import random
import sys

import pytest

sys.path.insert( 0, os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), ".." ) )

from strategy_policies import (  # noqa: E402
    STRATEGY_POLICIES,
    TempoPolicy,
    builtin_policy,
    greedy_policy,
    make_strategy_policy,
)


def context( *heroes ):
    return {"ev": "turn_context", "t": 1, "heroes": [{"id": h, "i": 0, "mp": mp, "mmp": mmp, "str": 1.0} for h, mp, mmp in heroes]}


def decision( hero, *cands ):
    return {"ev": "decision", "t": 1, "h": hero, "from": 0, "cands": [{"i": i, "obj": 0, "v": v, "d": d} for i, v, d in cands]}


def test_baselines():
    ev = decision( 1, ( 10, 5.0, 100 ), ( 20, 9.0, 100 ) )
    assert greedy_policy( ev )["i"] == 20
    assert builtin_policy( ev ) is None
    assert greedy_policy( decision( 1 ) ) is None


def test_make_strategy_policy_knows_every_name():
    for name in STRATEGY_POLICIES:
        if name == "learned":
            continue  # needs a model file, see test_strategy_model.py
        make_strategy_policy( name, random.Random( 0 ) )
    with pytest.raises( ValueError ):
        make_strategy_policy( "nope", random.Random( 0 ) )


def test_tempo_prefers_reachable_target_over_slightly_better_far_one():
    policy = TempoPolicy( gamma=0.8 )
    policy.observe_turn( context( ( 1, 500, 1500 ) ) )

    # 1000 needs one extra turn (ceil((900 - 500) / 1500) = 1) -> 800 < 900 reachable now.
    ev = decision( 1, ( 10, 1000.0, 900 ), ( 20, 900.0, 400 ) )
    assert policy.score( ev, ev["cands"][0] ) == pytest.approx( 800.0 )
    assert policy.score( ev, ev["cands"][1] ) == pytest.approx( 900.0 )
    assert policy( ev )["i"] == 20


def test_tempo_discount_grows_with_travel_turns():
    policy = TempoPolicy( gamma=0.5 )
    policy.observe_turn( context( ( 1, 0, 100 ) ) )

    ev = decision( 1, ( 10, 100.0, 250 ) )  # ceil(250 / 100) = 3 turns
    assert policy.score( ev, ev["cands"][0] ) == pytest.approx( 12.5 )


def test_tempo_without_context_falls_back_to_value():
    policy = TempoPolicy()
    ev = decision( 7, ( 10, 5.0, 10000 ), ( 20, 3.0, 1 ) )
    assert policy( ev )["i"] == 10


def test_tempo_penalizes_tiles_claimed_by_other_heroes_this_turn():
    policy = TempoPolicy( claim_penalty=0.5 )
    policy.observe_turn( context( ( 1, 1000, 1000 ), ( 2, 1000, 1000 ) ) )

    assert policy( decision( 1, ( 10, 100.0, 10 ), ( 20, 60.0, 10 ) ) )["i"] == 10
    # Hero 2 sees the same targets: tile 10 is claimed by hero 1 -> 50 < 60.
    assert policy( decision( 2, ( 10, 100.0, 10 ), ( 20, 60.0, 10 ) ) )["i"] == 20
    # The claiming hero itself is not penalized (re-activation in the same turn).
    assert policy( decision( 1, ( 10, 100.0, 10 ), ( 20, 60.0, 10 ) ) )["i"] == 10

    # A new kingdom turn clears the claims.
    policy.observe_turn( context( ( 1, 1000, 1000 ), ( 2, 1000, 1000 ) ) )
    assert policy( decision( 2, ( 10, 100.0, 10 ), ( 20, 60.0, 10 ) ) )["i"] == 10


def test_tempo_keeps_engine_order_on_ties_and_handles_empty():
    policy = TempoPolicy()
    assert policy( decision( 1, ( 10, 5.0, 1 ), ( 20, 5.0, 1 ) ) )["i"] == 10
    assert policy( decision( 1 ) ) is None
