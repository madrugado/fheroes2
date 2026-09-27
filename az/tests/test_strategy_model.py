"""Unit tests for az/strategy_model.py, az/strategy_rollout.py helpers and LearnedPolicy
(synthetic data, no engine)."""

import json
import os
import random
import sys

import numpy as np

sys.path.insert( 0, os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), ".." ) )

import strategy_model as sm  # noqa: E402
from strategy_policies import LearnedPolicy, make_strategy_policy  # noqa: E402
from strategy_rollout import Branch, parse_seeds, same_decision, select_decisions, stat_delta  # noqa: E402

CONTEXT = {"ev": "turn_context", "t": 3, "p": "Blue", "res": [0, 0, 0, 0, 0, 0, 5000], "castles": [{"n": "A", "i": 1}],
           "heroes": [{"id": 7, "i": 0, "mp": 1000, "mmp": 1500, "str": 300.0}]}


def decision( cands, t=3, n_hero=7 ):
    return {"ev": "decision", "t": t, "p": "Blue", "h": n_hero, "from": 0,
            "cands": [{"i": i, "obj": obj, "v": v, "d": d} for i, obj, v, d in cands]}


def synthetic_records( seeds=range( 8 ), rng_seed=0 ):
    """A world where object type 5 is secretly worth +600 strength and type 9 costs -400,
    whatever the built-in value says."""
    rng = random.Random( rng_seed )
    records = []
    for seed in seeds:
        for n in range( 12 ):
            objs = [rng.choice( [1, 5, 9] ) for _ in range( 3 )]
            dec = decision( [( 100 + k, objs[k], 1000.0 - 200 * k, 100 * ( k + 1 ) ) for k in range( 3 )] )
            effect = {1: 0.0, 5: 600.0, 9: -400.0}
            for j in range( 3 ):
                label = 0.0 if j == 0 else effect[objs[j]] - effect[objs[0]]
                records.append( {"seed": seed, "n": n, "j": j, "decision": dec, "context": CONTEXT, "cand": dec["cands"][j],
                                 "delta": {"outcome": 0, "k": 0, "h": 0, "str": label, "g": 0}} )
    return records


def test_label_score_weights_castles_and_outcome():
    assert sm.label_score( {"str": 100, "k": 1, "outcome": 0} ) == 100 + sm.CASTLE_WEIGHT
    assert sm.label_score( {"str": 0, "k": 0, "outcome": -1} ) < -1000


def test_candidate_features_are_relative_to_the_top_candidate():
    dec = decision( [( 1, 5, 1000.0, 500 ), ( 2, 9, 500.0, 2000 )] )
    vocab = [5, 9]
    top, second = ( sm.candidate_features( dec, CONTEXT, j, vocab ) for j in range( 2 ) )
    assert len( top ) == len( second )
    assert top[0] == 1.0 and second[0] == 0.0  # is-top flag
    assert second[3] == 0.5  # value ratio to the top candidate
    assert top[6] == 1.0 and second[6] == 0.0  # reachable with the hero's move points (1000)
    assert second[5] == ( 2000 - 500 ) / 1500  # extra distance in hero turns (mmp 1500)
    assert top[-3:] == [1.0, 0.0, 0.0] and second[-3:] == [0.0, 1.0, 0.0]  # object one-hot + other


def test_features_survive_a_missing_context():
    dec = decision( [( 1, 5, 1000.0, 500 ), ( 2, 9, 0.0, 10 )] )
    assert len( sm.candidate_features( dec, None, 1, [] ) ) == len( sm.candidate_features( dec, CONTEXT, 1, [] ) )


def test_ridge_and_mlp_learn_the_synthetic_effect():
    records = synthetic_records()
    vocab = sm.build_obj_vocab( records )
    x, y = sm.design( records, vocab )
    for model in ( sm.fit_ridge( x, y, alpha=1.0 ), sm.fit_mlp( x, y, epochs=400 ) ):
        gain = sm.policy_gain( model, records, vocab, margin=0.0 )
        assert gain["mean_gain"] > 0.5 * gain["oracle_gain"] > 0
        assert gain["negative"] <= gain["positive"] // 4


def test_cross_validation_keeps_seeds_apart_and_reports_every_margin():
    records = synthetic_records()
    cv = sm.cross_validate( records, "ridge", margins=[0.0, 1e9], folds=4 )
    assert set( cv ) == {0.0, 1e9}
    assert cv[1e9]["switched"] == 0 and cv[1e9]["mean_gain"] == 0.0  # huge margin == built-in
    assert cv[0.0]["decisions"] == len( sm.group_decisions( records ) )
    assert cv[0.0]["mean_gain"] > 0


def test_learned_policy_round_trips_through_json(tmp_path):
    records = synthetic_records()
    model = sm.train_model( records, "ridge", margin=0.0, top=3 )
    path = tmp_path / "model.json"
    path.write_text( json.dumps( model ) )

    policy = make_strategy_policy( "learned", random.Random( 0 ), str( path ) )
    assert isinstance( policy, LearnedPolicy )
    policy.observe_turn( CONTEXT )
    # Candidate 2 is object type 5 (+600), the top one is type 9 (-400).
    assert policy( decision( [( 1, 9, 1000.0, 100 ), ( 2, 5, 800.0, 200 ), ( 3, 1, 600.0, 300 )] ) )["i"] == 2
    # Nothing better than the top candidate: keep the built-in choice.
    assert policy( decision( [( 1, 5, 1000.0, 100 ), ( 2, 9, 800.0, 200 )] ) ) is None
    assert policy( decision( [( 1, 5, 1000.0, 100 )] ) ) is None


def test_predict_matches_between_ridge_json_and_numpy():
    records = synthetic_records( seeds=range( 2 ) )
    vocab = sm.build_obj_vocab( records )
    x, y = sm.design( records, vocab )
    model = json.loads( json.dumps( sm.fit_ridge( x, y ) ) )
    assert np.allclose( sm.predict( model, x ), sm.predict( sm.fit_ridge( x, y ), x ) )


def test_branch_picks_only_at_the_expected_decision():
    dec = decision( [( 1, 5, 1000.0, 100 ), ( 2, 9, 800.0, 200 )] )
    branch = Branch( pick_at=1, pick_tile=2, expected=dec )
    branch.observe_turn( CONTEXT )
    assert branch( dec ) is None  # decision 0: built-in
    assert branch( dec )["i"] == 2 and branch.picked and not branch.diverged
    assert branch.decisions[1]["context"] is CONTEXT

    diverging = Branch( pick_at=0, pick_tile=2, expected=dict( dec, h=99 ) )
    assert diverging( dec ) is None and diverging.diverged

    missing = Branch( pick_at=0, pick_tile=42, expected=dec )
    assert missing( dec ) is None and missing.diverged


def test_rollout_helpers():
    assert parse_seeds( "3-5" ) == [3, 4, 5] and parse_seeds( "7,9" ) == [7, 9]
    assert stat_delta( {"outcome": 0, "k": 2, "h": 1, "str": 50, "g": 10}, {"outcome": 0, "k": 1, "h": 1, "str": 80, "g": 0} ) == \
        {"outcome": 0, "k": 1, "h": 0, "str": -30, "g": 10}
    dec = decision( [( 1, 5, 1.0, 1 ), ( 2, 5, 1.0, 1 )], t=4 )
    assert same_decision( dec, json.loads( json.dumps( dec ) ) )
    items = [{"n": n, "decision": decision( [( 1, 5, 1.0, 1 )] * ( 1 + n % 2 ), t=n )} for n in range( 20 )]
    chosen = select_decisions( items, per_seed=3, max_day=10, rng=random.Random( 0 ) )
    assert len( chosen ) == 3 and [c["n"] for c in chosen] == sorted( c["n"] for c in chosen )
    assert all( len( c["decision"]["cands"] ) >= 2 and 1 <= c["decision"]["t"] <= 10 for c in chosen )
