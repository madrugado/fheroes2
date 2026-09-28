"""Unit tests for the strategic output of the unified transformer (no engine)."""

import json
import os
import sys

sys.path.insert( 0, os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), ".." ) )

import pytest

torch = pytest.importorskip( "torch" )
pytest.importorskip( "transformers" )

import strategy_net  # noqa: E402
from transformer_model import STRAT_TOKEN_W, AzBattleTransformer, load_checkpoint, save_checkpoint, strategic_logits  # noqa: E402

CONTEXT = {"ev": "turn_context", "t": 3, "p": "Blue", "res": [5, 0, 5, 0, 0, 0, 3000],
           "castles": [{"i": 100}], "heroes": [{"id": 7, "mp": 900, "mmp": 1200, "str": 400.0}]}
TARGET = {"ev": "decision", "t": 3, "p": "Blue", "h": 7,
          "cands": [{"i": 10, "obj": 138, "v": 300.0, "d": 200}, {"i": 11, "obj": 134, "v": 90.0, "d": 100},
                    {"i": 12, "obj": 138, "v": 20.0, "d": 900}]}
ARMY = {"ev": "army", "t": 3, "p": "Blue", "castle": 100, "reason": "visit", "garrison": 50.0, "hero": 400.0,
        "offer": [{"mon": 1, "avail": 10, "n": 10, "str": 11.0}]}


def test_query_tokens_have_the_model_width():
    context, options = strategy_net.query_tokens( "target", TARGET, CONTEXT, [138] )
    assert len( context ) == STRAT_TOKEN_W and context[-1] == 1.0
    assert len( options ) == 3 and all( len( o ) == STRAT_TOKEN_W and o[-1] == 0.0 for o in options )


def test_strategic_logits_are_masked_and_independent_of_padding():
    model = AzBattleTransformer().eval()
    target = strategy_net.query_tokens( "target", TARGET, CONTEXT, [] )
    army = strategy_net.query_tokens( "army", ARMY, CONTEXT, [] )
    with torch.no_grad():
        logits, mask = strategic_logits( model, [target, army] )
        alone, _ = strategic_logits( model, [army] )
    assert mask.tolist() == [[True, True, True], [True, True, True]]
    assert torch.allclose( logits[1], alone[0], atol=1e-5 )


def test_option_scores_see_every_option():
    """The second copy of the options is scored: changing the LAST option changes the first score."""
    model = AzBattleTransformer().eval()
    context, options = strategy_net.query_tokens( "target", TARGET, CONTEXT, [] )
    changed = [list( o ) for o in options]
    changed[-1][0] += 5.0
    with torch.no_grad():
        a, _ = strategic_logits( model, [( context, options )] )
        b, _ = strategic_logits( model, [( context, changed )] )
    assert abs( float( a[0, 0] - b[0, 0] ) ) > 1e-6


def test_checkpoints_without_the_strategic_head_still_load( tmp_path ):
    model = AzBattleTransformer()
    state = {k: v for k, v in model.state_dict().items() if not k.startswith( ( "strat_proj.", "strat_head." ) )}
    path = tmp_path / "old.pt"
    torch.save( {"arch": "transformer", "config": model.config, "state_dict": state}, path )
    loaded = load_checkpoint( str( path ) )
    assert hasattr( loaded, "strat_head" )
    save_checkpoint( loaded, str( tmp_path / "new.pt" ) )
    load_checkpoint( str( tmp_path / "new.pt" ) )


def test_preference_pairs_from_rollout_labels():
    base = {"seed": 1, "map": "m", "horizon": 7, "n": 5, "kind": "army", "event": ARMY, "context": CONTEXT,
            "builtin_index": 2, "base": {}}
    records = [dict( base, option_index=0, option=0, delta={"outcome": 0, "k": 0, "h": 0, "str": -900, "g": 0} ),
               dict( base, option_index=1, option=50, delta={"outcome": 0, "k": 1, "h": 0, "str": 0, "g": 0} )]
    ( pair, ) = strategy_net.preference_pairs( records, margin=100 )
    assert pair["chosen"] == 1 and pair["rejected"] == 0  # 50% (+1 castle) > 100% (0) > 0% (-900)
    assert pair["scores"] == {"2": 0.0, "0": -900.0, "1": 2000.0}


def test_net_policy_answers_every_kind( tmp_path ):
    path = tmp_path / "m.pt"
    save_checkpoint( AzBattleTransformer(), str( path ) )
    policy = strategy_net.NetStrategyPolicy( str( path ) )
    policy.observe_turn( CONTEXT )
    probs = policy.probabilities( "target", TARGET )
    assert len( probs ) == 3 and abs( sum( probs ) - 1.0 ) < 1e-5
    assert policy.army( ARMY ) in ( 0, 50, 100 )
    choice = policy( TARGET )
    assert choice is None or choice in TARGET["cands"]
