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


# --- strategic layer: build / hire queries ---

from strategy_policies import NOTHING, ForColor, RandomPolicy, attach_build_result, strategic_reply  # noqa: E402

BUILD = {"ev": "build", "t": 2, "p": "Blue", "castle": 77, "race": 1, "defensive": 0, "res": [0] * 7,
         "cands": [{"b": 16, "name": "Statue", "trade": 0, "cost": [0] * 7}, {"b": 2, "name": "Tavern", "trade": 1, "cost": [0] * 7}]}
HIRE = {"ev": "hire", "t": 2, "p": "Blue", "heroes": 1, "res": [0] * 7, "bi": 1,
        "cands": [{"castle": 77, "slot": 1, "hero": 4, "race": 1, "lvl": 1, "val": 1.0, "army": 0},
                  {"castle": 77, "slot": 2, "hero": 9, "race": 2, "lvl": 1, "val": 2.0, "army": 0}]}


class Scripted:
    def __init__( self, build=None, hire=None ):
        self._build, self._hire = build, hire

    def __call__( self, decision ):
        return None

    def build( self, ev ):
        return self._build

    def hire( self, ev ):
        return self._hire


def test_strategic_reply_build():
    assert strategic_reply( Scripted( build=BUILD["cands"][1] ), BUILD )[0] == {"op": "build", "castle": 77, "b": 2}
    reply, record = strategic_reply( Scripted( build=NOTHING ), BUILD )
    assert reply == {"op": "build", "castle": 77, "b": 0} and record["chosen"] == 0
    reply, record = strategic_reply( Scripted(), BUILD )
    assert reply == {"op": "skip"} and record["chosen"] is None and record["kind"] == "build"
    # A policy without build(): built-in choice.
    assert strategic_reply( lambda ev: None, BUILD )[0] == {"op": "skip"}


def test_strategic_reply_hire():
    reply, record = strategic_reply( Scripted( hire=HIRE["cands"][0] ), HIRE )
    assert reply == {"op": "hire", "castle": 77, "slot": 1} and record["chosen"] == 0 and record["bi"] == 1
    reply, record = strategic_reply( Scripted( hire=NOTHING ), HIRE )
    assert reply == {"op": "hire", "castle": -1} and record["chosen"] == -1
    assert strategic_reply( Scripted(), HIRE )[0] == {"op": "skip"}


def test_strategic_reply_target_and_unknown():
    decision = {"ev": "decision", "t": 1, "p": "Blue", "h": 3, "from": 0, "cands": [{"i": 5, "v": 1.0, "d": 1}]}
    reply, record = strategic_reply( greedy_policy, decision )
    assert reply == {"op": "pick", "h": 3, "i": 5} and record["kind"] == "target" and record["chosen"] == 5
    with pytest.raises( ValueError ):
        strategic_reply( greedy_policy, {"ev": "turn_context"} )


def test_attach_build_result_matches_castle_and_day():
    records = [strategic_reply( Scripted(), BUILD )[1], strategic_reply( Scripted(), dict( BUILD, castle=88 ) )[1]]
    attach_build_result( records, {"ev": "build_result", "t": 2, "castle": 77, "b": 16, "src": "builtin"} )
    attach_build_result( records, {"ev": "build_result", "t": 2, "castle": 99, "b": 2, "src": "builtin"} )  # no query: ignored
    assert records[0]["result"] == 16 and records[0]["src"] == "builtin"
    assert "result" not in records[1]


def test_random_policy_and_for_color_cover_build_and_hire():
    policy = RandomPolicy( random.Random( 0 ) )
    assert policy.build( BUILD ) in BUILD["cands"]
    picks = {repr( policy.hire( HIRE ) ) for _ in range( 50 )}
    assert repr( NOTHING ) in picks and len( picks ) == 3

    blue = ForColor( Scripted( build=NOTHING, hire=NOTHING ), "Blue" )
    assert blue.build( BUILD ) == NOTHING and blue.hire( HIRE ) == NOTHING
    assert blue.build( dict( BUILD, p="Red" ) ) is None and blue.hire( dict( HIRE, p="Red" ) ) is None
    assert ForColor( lambda ev: None, "Blue" ).build( BUILD ) is None  # wrapped policy without build()


ARMY = {"ev": "army", "t": 3, "p": "Blue", "castle": 77, "reason": "visit", "guest": 5, "garrison": 10.0, "hero": 50.0,
        "res": [0] * 7, "offer": [{"mon": 1, "avail": 12, "n": 12, "str": 11.0}]}


def test_strategic_reply_army():
    class Budget:
        def __init__( self, value ):
            self.value = value

        def __call__( self, ev ):
            return None

        def army( self, ev ):
            return self.value

    reply, record = strategic_reply( Budget( 50 ), ARMY )
    assert reply == {"op": "army", "castle": 77, "pct": 50} and record["chosen"] == 50 and record["reason"] == "visit"
    assert strategic_reply( Budget( NOTHING ), ARMY )[0]["pct"] == 0
    assert strategic_reply( Budget( None ), ARMY )[0] == {"op": "skip"}
    assert RandomPolicy( random.Random( 0 ) ).army( ARMY ) in ( 0, 50, 100 )
    assert ForColor( Budget( 0 ), "Red" ).army( ARMY ) is None and ForColor( Budget( 0 ), "Blue" ).army( ARMY ) == 0
