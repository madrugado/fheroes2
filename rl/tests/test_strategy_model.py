"""Unit tests for rl/strategy_model.py, the rl/strategy_rollout.py helpers and LearnedPolicy
(synthetic data, no engine)."""

import json
import os
import random
import sys

import numpy as np

sys.path.insert( 0, os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), ".." ) )

import strategy_model as sm  # noqa: E402
from strategy_policies import NOTHING, LearnedPolicy, make_strategy_policy  # noqa: E402
from strategy_rollout import Branch, builtin_option, convert_legacy, parse_seeds, same_query, select_queries, stat_delta  # noqa: E402

CONTEXT = {"ev": "turn_context", "t": 3, "p": "Blue", "res": [0, 0, 0, 0, 0, 0, 5000], "castles": [{"n": "A", "i": 1}],
           "heroes": [{"id": 7, "i": 0, "mp": 1000, "mmp": 1500, "str": 300.0}]}


def target_event( cands, t=3 ):
    return {"ev": "decision", "t": t, "p": "Blue", "h": 7, "from": 0, "cands": [{"i": i, "obj": obj, "v": v, "d": d} for i, obj, v, d in cands]}


def build_event( bits, t=3 ):
    return {"ev": "build", "t": t, "p": "Blue", "castle": 5, "race": 1, "defensive": 0, "res": [5] * 6 + [5000],
            "cands": [{"b": b, "name": str( b ), "trade": 0, "cost": [0] * 6 + [1000]} for b in bits]}


def hire_event( bi=0 ):
    return {"ev": "hire", "t": 3, "p": "Blue", "heroes": 1, "res": [0] * 6 + [5000], "bi": bi,
            "cands": [{"castle": 5, "slot": 1, "hero": 1, "race": 1, "lvl": 1, "val": 800.0, "army": 100.0},
                      {"castle": 5, "slot": 2, "hero": 2, "race": 2, "lvl": 2, "val": 900.0, "army": 100.0}]}


def army_event( reason="visit" ):
    return {"ev": "army", "t": 3, "p": "Blue", "castle": 5, "reason": reason, "guest": 7, "garrison": 10.0, "hero": 50.0,
            "res": [0] * 6 + [5000], "offer": [{"mon": 1, "avail": 10, "n": 10, "str": 20.0}]}


def record( seed, n, kind, event, index, option, builtin, label ):
    return {"seed": seed, "n": n, "kind": kind, "event": event, "context": CONTEXT, "option_index": index, "option": option,
            "builtin_index": builtin, "delta": {"outcome": 0, "k": 0, "h": 0, "str": label, "g": 0}}


def synthetic_records( seeds=range( 8 ) ):
    """Hidden truths: target object type 5 is worth +600; the Statue (bit 16) +500 over anything;
    hiring slot 2 +300; army budget 50% +200. Built-in choices: top candidate / first building /
    slot 1 / 100%."""
    rng = random.Random( 0 )
    records = []
    for seed in seeds:
        for n in range( 10 ):
            objs = [rng.choice( [1, 5, 9] ) for _ in range( 3 )]
            ev = target_event( [( 100 + k, objs[k], 1000.0 - 200 * k, 100 * ( k + 1 ) ) for k in range( 3 )] )
            value = {1: 0.0, 5: 600.0, 9: -400.0}
            for j in range( 3 ):
                records.append( record( seed, n, "target", ev, j, ev["cands"][j], 0, 0.0 if j == 0 else value[objs[j]] - value[objs[0]] ) )

            bits = rng.sample( [2, 16, 256, 1024], 3 )
            ev = build_event( bits )
            options = sm.options_of( "build", ev )
            worth = [500.0 if o != NOTHING and o["b"] == 16 else 0.0 for o in options]
            for i, option in enumerate( options ):
                records.append( record( seed, 100 + n, "build", ev, i, option, 0, worth[i] - worth[0] ) )

            ev = hire_event()
            for i, option in enumerate( sm.options_of( "hire", ev ) ):
                records.append( record( seed, 200 + n, "hire", ev, i, option, 0, 300.0 if i == 1 else 0.0 ) )

            ev = army_event( rng.choice( ["visit", "hire"] ) )
            for i, option in enumerate( sm.options_of( "army", ev ) ):
                records.append( record( seed, 300 + n, "army", ev, i, option, 2, 200.0 if option == 50 else 0.0 ) )
    return records


def test_label_score_weights_castles_and_outcome():
    assert sm.label_score( {"str": 100, "k": 1, "outcome": 0} ) == 100 + sm.CASTLE_WEIGHT
    assert sm.label_score( {"str": 0, "k": 0, "outcome": -1} ) < -1000


def test_options_of_every_kind():
    assert len( sm.options_of( "target", target_event( [( i, 1, 1.0, 1 ) for i in range( 6 )] ), top=4 ) ) == 4
    assert sm.options_of( "build", build_event( [2] ) )[-1] == NOTHING
    assert sm.options_of( "hire", hire_event() )[-1] == NOTHING and len( sm.options_of( "hire", hire_event() ) ) == 3
    assert sm.options_of( "army", army_event() ) == [0, 50, 100]


def test_feature_vectors_have_a_fixed_width_per_kind():
    for kind, ev in ( ( "target", target_event( [( 1, 5, 1000.0, 500 ), ( 2, 9, 500.0, 2000 )] ) ), ( "build", build_event( [2, 16] ) ),
                      ( "hire", hire_event() ), ( "army", army_event() ) ):
        widths = {len( sm.option_features( kind, ev, ctx, option, i, [5, 9] ) ) for i, option in enumerate( sm.options_of( kind, ev ) )
                  for ctx in ( CONTEXT, None )}
        assert len( widths ) == 1, kind


def test_target_features_are_relative_to_the_top_candidate():
    ev = target_event( [( 1, 5, 1000.0, 500 ), ( 2, 9, 500.0, 2000 )] )
    top, second = ( sm.target_features( ev, CONTEXT, j, [5, 9] ) for j in range( 2 ) )
    assert top[0] == 1.0 and second[0] == 0.0  # is-top (built-in) flag
    assert second[3] == 0.5  # value ratio to the top candidate
    assert top[6] == 1.0 and second[6] == 0.0  # reachable with the hero's move points (1000)
    assert second[5] == ( 2000 - 500 ) / 1500  # extra distance in hero turns (mmp 1500)


def test_build_and_army_features_encode_the_option():
    ev = build_event( [2, 16] )
    statue = sm.build_features( ev, CONTEXT, ev["cands"][1] )
    nothing = sm.build_features( ev, CONTEXT, NOTHING )
    assert nothing[0] == 1.0 and statue[0] == 0.0
    assert statue[7 + 4] == 1.0 and sum( statue[7:39] ) == 1.0  # bit 16 -> index 4 of the one-hot
    assert sum( nothing[7:39] ) == 0.0
    assert sm.army_features( army_event(), CONTEXT, 50 )[0] == 0.5


def test_every_kind_is_learned_and_enabled_on_a_clear_signal():
    records = synthetic_records()
    model = sm.select_and_train( records, margins=[0.0, 100.0], archs=["ridge"], rule="ci", log=lambda *_: None )
    assert set( model["kinds"] ) == set( sm.KINDS )
    for kind, sub in model["kinds"].items():
        assert sub["enabled"], kind
        assert sub["cv"]["mean_gain"] > 100, kind


def test_noise_labels_keep_the_kind_disabled():
    rng = random.Random( 3 )
    records = [dict( r, delta=dict( r["delta"], str=0.0 if r["option_index"] == r["builtin_index"] else rng.gauss( 0, 1000 ) ) )
               for r in synthetic_records() if r["kind"] == "army"]
    model = sm.select_and_train( records, margins=[0.0], archs=["ridge"], rule="ci", log=lambda *_: None )
    assert not model["kinds"]["army"]["enabled"]


def test_mlp_fits_too():
    records = [r for r in synthetic_records() if r["kind"] == "hire"]
    x, y = sm.design( records, [] )
    model = json.loads( json.dumps( sm.fit_mlp( x, y, epochs=300 ) ) )
    gains = sm.query_gains( model, records, [], margin=0.0 )
    assert np.mean( gains ) > 200


def test_learned_policy_answers_every_kind(tmp_path):
    model = sm.select_and_train( synthetic_records(), margins=[0.0], archs=["ridge"], rule="mean", log=lambda *_: None )
    path = tmp_path / "model.json"
    path.write_text( json.dumps( model ) )

    policy = make_strategy_policy( "learned", random.Random( 0 ), str( path ) )
    assert isinstance( policy, LearnedPolicy )
    policy.observe_turn( CONTEXT )

    # Target: candidate 2 is object type 5 (+600), the top one is type 9 (-400).
    assert policy( target_event( [( 1, 9, 1000.0, 100 ), ( 2, 5, 800.0, 200 ), ( 3, 1, 600.0, 300 )] ) )["i"] == 2
    assert policy( target_event( [( 1, 5, 1000.0, 100 )] ) ) is None
    assert policy.build( build_event( [2, 16, 256] ) )["b"] == 16
    assert policy.hire( hire_event() )["slot"] == 2
    assert policy.army( army_event() ) == 50

    # A disabled kind keeps the built-in choice.
    model["kinds"]["army"]["enabled"] = False
    path.write_text( json.dumps( model ) )
    assert LearnedPolicy( str( path ) ).army( army_event() ) is None


def test_learned_policy_reads_the_first_model_format(tmp_path):
    records = [r for r in synthetic_records() if r["kind"] == "target"]
    vocab = sm.build_obj_vocab( records )
    x, y = sm.design( records, vocab )
    v1 = dict( sm.fit_ridge( x, y ), kind="ridge", obj_vocab=vocab, margin=0.0, top=3 )
    del v1["arch"]
    path = tmp_path / "v1.json"
    path.write_text( json.dumps( v1 ) )
    policy = LearnedPolicy( str( path ) )
    policy.observe_turn( CONTEXT )
    assert policy( target_event( [( 1, 9, 1000.0, 100 ), ( 2, 5, 800.0, 200 )] ) )["i"] == 2


def test_branch_answers_only_the_expected_query():
    ev = hire_event()
    branch = Branch( pick_at=1, answer=NOTHING, expected=ev )
    branch.observe_turn( CONTEXT )
    assert branch.army( army_event() ) is None  # query 0: built-in
    assert branch.hire( ev ) == NOTHING and branch.picked and not branch.diverged
    assert [q["kind"] for q in branch.queries] == ["army", "hire"] and branch.queries[1]["context"] is CONTEXT

    diverging = Branch( pick_at=0, answer=50, expected=army_event( "defense" ) )
    assert diverging.army( army_event( "visit" ) ) is None and diverging.diverged


def test_builtin_option_per_kind():
    ev = build_event( [2, 16] )
    options = sm.options_of( "build", ev )
    assert builtin_option( "build", ev, options, 16 ) == 1
    assert builtin_option( "build", ev, options, 0 ) == 2  # nothing built -> NOTHING
    assert builtin_option( "build", ev, options, 4096 ) is None and builtin_option( "build", ev, options, None ) is None
    assert builtin_option( "hire", hire_event( bi=1 ), sm.options_of( "hire", hire_event() ) ) == 1
    assert builtin_option( "hire", hire_event( bi=-1 ), sm.options_of( "hire", hire_event() ) ) == 2
    assert builtin_option( "army", army_event(), [0, 50, 100] ) == 2
    assert builtin_option( "target", target_event( [( 1, 1, 1.0, 1 )] ), [] ) == 0


def test_rollout_helpers():
    assert parse_seeds( "3-5" ) == [3, 4, 5] and parse_seeds( "7,9" ) == [7, 9]
    assert stat_delta( {"outcome": 0, "k": 2, "h": 1, "str": 50, "g": 10}, {"outcome": 0, "k": 1, "h": 1, "str": 80, "g": 0} ) == \
        {"outcome": 0, "k": 1, "h": 0, "str": -30, "g": 10}
    assert same_query( army_event(), json.loads( json.dumps( army_event() ) ) )
    assert not same_query( army_event(), army_event( "hire" ) )

    queries = [{"n": n, "kind": kind, "event": dict( ev, t=n % 12 )} for n, ( kind, ev ) in
               enumerate( [( "army", army_event() ), ( "hire", hire_event() ), ( "build", build_event( [2] ) )] * 8 )]
    chosen = select_queries( queries, {"army": 2, "hire": 5, "build": 3}, max_day=10, top=4, rng=random.Random( 0 ) )
    assert [q["n"] for q in chosen] == sorted( q["n"] for q in chosen )
    assert sum( q["kind"] == "army" for q in chosen ) == 2 and sum( q["kind"] == "hire" for q in chosen ) == 5
    assert all( 1 <= q["event"]["t"] <= 10 for q in chosen )


def test_convert_legacy_target_records():
    legacy = {"seed": 1, "n": 4, "j": 1, "decision": target_event( [( 1, 1, 2.0, 1 ), ( 2, 1, 1.0, 1 )] ), "context": CONTEXT,
              "cand": {"i": 2}, "delta": {"outcome": 0, "k": 0, "h": 0, "str": 5, "g": 0}, "map": "m", "horizon": 7}
    converted = convert_legacy( legacy )
    assert converted["kind"] == "target" and converted["option_index"] == 1 and converted["builtin_index"] == 0
    assert converted["event"] is legacy["decision"] and convert_legacy( converted ) is converted
