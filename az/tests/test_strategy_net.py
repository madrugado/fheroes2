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


def test_policy_branch_answers_only_its_color_and_the_picked_query():
    import strategy_games

    class Fixed:
        def observe_turn( self, ctx ):
            pass

        def __call__( self, ev ):
            return ev["cands"][1]

        def build( self, ev ):
            return None

        def hire( self, ev ):
            return None

        def army( self, ev ):
            return 50

    other = dict( ARMY, p="Red" )
    branch = strategy_games.PolicyBranch( Fixed(), "Blue", pick_at=2, answer=0, expected=ARMY )
    assert branch( TARGET ) == TARGET["cands"][1]       # query 0: our policy
    assert branch.army( other ) is None                  # query 1: another color -> built-in
    assert branch.army( ARMY ) == 0                      # query 2: the branch's answer
    assert branch.army( ARMY ) == 50                     # query 3: our policy again
    assert [q["n"] for q in branch.queries] == [0, 2, 3] and not branch.diverged


def test_option_index_of_builtin_answers():
    import strategy_games

    options = strategy_net.query_options( "army", ARMY )
    assert strategy_games.option_index( "army", options, None, ARMY ) == options.index( 100 )
    assert strategy_games.option_index( "target", strategy_net.query_options( "target", TARGET ), None, TARGET ) == 0
    build = {"ev": "build", "t": 3, "p": "Blue", "castle": 100, "race": 1, "res": [5] * 7,
             "cands": [{"b": 16, "name": "Statue", "trade": 0, "cost": [0, 0, 0, 0, 0, 0, 1250]}]}
    options = strategy_net.query_options( "build", build )
    assert options[0] == strategy_net.BUILTIN and options[-1] == "nothing"
    assert strategy_games.builtin_index( "build", options, build ) == 0
    assert strategy_net.answer_of( options[0] ) is None
    context, tokens = strategy_net.query_tokens( "build", build, CONTEXT, [] )
    assert len( tokens ) == 3 and tokens[0][strategy_net.STRAT_FEATURES - 1] == 1.0


def test_real_hero_battles_from_the_ai_log():
    import strategy_games

    events = [
        {"ev": "battle_start", "bid": 1, "t": 3, "att": {"hero": {}, "c": "B"}, "def": {"hero": {}, "c": "R"}},
        {"ev": "battle_end", "bid": 1, "t": 3, "winner": "B", "att0": 1000, "att1": 600, "def0": 800, "def1": 0},
        {"ev": "battle_start", "bid": 2, "t": 4, "att": {"hero": {}, "c": "B"}, "def": {"c": "N"}},  # neutrals
        {"ev": "battle_end", "bid": 2, "t": 4, "winner": "B", "att0": 1, "att1": 1, "def0": 1, "def1": 0},
        {"ev": "battle_start", "bid": 3, "t": 20, "att": {"hero": {}, "c": "R"}, "def": {"hero": {}, "c": "B"}},  # too late
        {"ev": "battle_end", "bid": 3, "t": 20, "winner": "R", "att0": 1, "att1": 1, "def0": 1, "def1": 0},
    ]
    ( score, ) = strategy_games.real_hero_battles( events, "Blue", 1, 10 )
    assert abs( score - ( 1.0 + 0.6 - 0.0 ) ) < 1e-9


def test_duel_score_plays_both_sides_and_handles_missing_heroes():
    import strategy_games

    class FakeDuelEnv:
        def __init__( self ):
            self.calls = []

        def new_battle( self, **kwargs ):
            self.calls.append( ( kwargs["hero_att"][0], kwargs["hero_def"][0] ) )
            return {"ev": "state", "result": "att",
                    "units": [{"u": 1, "side": "att", "str": 100, "q": 1}, {"u": 2, "side": "def", "str": 0, "q": 0}]}

    env = FakeDuelEnv()
    results = {"Blue": {"top": {"hid": 5, "hero": "aa", "str": 900}},
               "Red": {"top": {"hid": 7, "hero": "bb", "str": 800}}, "Green": {"top": {"hid": 9, "hero": "cc", "str": 50}}}
    score = strategy_games.duel_score( env, results, "Blue", 1 )
    assert env.calls == [( 5, 7 ), ( 7, 5 )]  # the strongest rival (Red), both orientations
    assert score == 0.0  # the attacker wins either way: +win once, -loss once
    assert strategy_games.duel_score( env, {"Blue": {}, "Red": results["Red"]}, "Blue", 1 ) == -2.0
    assert strategy_games.duel_score( env, {"Blue": results["Blue"]}, "Blue", 1 ) == 2.0


def test_smoothed_nll_ignores_padding_options():
    import torch.nn.functional as F

    import train_strategy_net

    logits = torch.tensor( [[2.0, 0.0, -1e9], [1.0, 1.0, 1.0]] )  # the first query has 2 real options
    log_probs = F.log_softmax( logits, dim=1 )
    plain = train_strategy_net.smoothed_nll( log_probs, [0, 1], 0.0 )
    assert abs( float( plain ) - float( F.nll_loss( log_probs, torch.tensor( [0, 1] ) ) ) ) < 1e-6
    smooth = train_strategy_net.smoothed_nll( log_probs, [0, 1], 0.5 )
    assert float( smooth ) < 1e6  # padding (-1e9) is not part of the uniform term
