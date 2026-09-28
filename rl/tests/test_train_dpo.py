"""Unit tests for rl/train_dpo.py (torch required, no engine)."""

import math
import os
import sys

sys.path.insert( 0, os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), ".." ) )

import pytest

torch = pytest.importorskip( "torch" )

import train_dpo  # noqa: E402
from model import AzBattleNet  # noqa: E402
from policy_value import ResNetPolicyValue  # noqa: E402


def make_pair( chosen: int, rejected: int ) -> dict:
    units = [
        {"u": 1, "side": "att", "mon": 13, "q": 30, "hpl": 10, "i": 0, "ti": -1, "sp": 2, "shots": 8, "moved": 0},
        {"u": 2, "side": "def", "mon": 57, "q": 5, "hpl": 20, "i": 6, "ti": 7, "sp": 2, "shots": 0, "moved": 0},
    ]
    legal = [
        {"act": 0, "args": [1, 1]},                # MOVE to cell 1
        {"act": 1, "args": [0, -1, -1, 2, 1]},     # shot at unit 2
        {"act": 1, "args": [4, 6, 5, 2, 1]},       # melee on cell 6 from cell 5
        {"act": 8, "args": [1]},                   # SKIP
        {"act": 2, "args": [6, 1]},                # Fireball on cell 6 ...
        {"act": 2, "args": [7, 1]},                # ... and on cell 7: one shared slot / token
    ]
    state = {"turn": 2, "cur": 1, "units": units, "obstacles": [], "heroes": [{"side": "att", "sp": 20, "cast": 0}]}
    return {"state": state, "legal": legal, "chosen": chosen, "rejected": rejected, "battle": "b"}


def test_dpo_loss_rewards_moving_towards_the_chosen_move():
    zero = torch.zeros( 1 )
    loss, margin = train_dpo.dpo_loss( zero, zero, zero, zero, beta=0.1 )
    assert abs( float( loss ) - math.log( 2 ) ) < 1e-6 and float( margin ) == 0.0
    better, _ = train_dpo.dpo_loss( torch.tensor( [1.0] ), torch.tensor( [-1.0] ), zero, zero, beta=0.1 )
    worse, _ = train_dpo.dpo_loss( torch.tensor( [-1.0] ), torch.tensor( [1.0] ), zero, zero, beta=0.1 )
    assert float( better ) < math.log( 2 ) < float( worse )


def test_resnet_move_log_probs_match_the_mcts_priors():
    """DPO trains exactly the move probabilities evaluate() hands to MCTS."""
    model = AzBattleNet().eval()
    pairs = [make_pair( i, 0 ) for i in range( 6 )]
    with torch.no_grad():
        logps = train_dpo.resnet_move_logps( model, pairs, "chosen", "cpu" )
    state = dict( pairs[0]["state"], legal=pairs[0]["legal"] )
    priors, _ = ResNetPolicyValue( model ).evaluate( state )
    for i in range( 6 ):
        assert abs( math.exp( float( logps[i] ) ) - priors[i] ) < 1e-5


def test_transformer_move_log_probs_match_the_mcts_priors():
    pytest.importorskip( "transformers" )
    from transformer_model import AzBattleTransformer

    model = AzBattleTransformer().eval()
    pairs = [make_pair( i, 0 ) for i in range( 6 )]
    with torch.no_grad():
        logps = train_dpo.transformer_move_logps( model, pairs, "chosen", "cpu" )
    state = dict( pairs[0]["state"], legal=pairs[0]["legal"] )
    priors, _ = model.evaluate( state )
    for i in range( 6 ):
        assert abs( math.exp( float( logps[i] ) ) - priors[i] ) < 1e-4


def test_a_few_dpo_steps_prefer_the_chosen_move():
    torch.manual_seed( 0 )
    model = AzBattleNet()
    ref = AzBattleNet()
    ref.load_state_dict( model.state_dict() )
    pairs = [make_pair( 2, 0 )] * 8
    optimizer = torch.optim.SGD( model.parameters(), lr=0.5 )
    for _ in range( 20 ):
        pc = train_dpo.resnet_move_logps( model, pairs, "chosen", "cpu" )
        pr = train_dpo.resnet_move_logps( model, pairs, "rejected", "cpu" )
        with torch.no_grad():
            rc = train_dpo.resnet_move_logps( ref, pairs, "chosen", "cpu" )
            rr = train_dpo.resnet_move_logps( ref, pairs, "rejected", "cpu" )
        loss, _ = train_dpo.dpo_loss( pc, pr, rc, rr, beta=1.0 )
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
    with torch.no_grad():
        assert float( train_dpo.resnet_move_logps( model, pairs[:1], "chosen", "cpu" ) ) > \
            float( train_dpo.resnet_move_logps( model, pairs[:1], "rejected", "cpu" ) )
