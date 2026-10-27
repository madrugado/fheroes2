"""Unit tests for JOLT (rl/jolt.py): the strategic views, the battle hint inputs and the two-role losses (no engine)."""

import copy
import json
import os
import sys

sys.path.insert( 0, os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), ".." ) )

import pytest

torch = pytest.importorskip( "torch" )
pytest.importorskip( "transformers" )

import jolt  # noqa: E402
import strategy_net  # noqa: E402
from .test_strategy_net import CONTEXT, TARGET  # noqa: E402
from .test_train_dpo import make_pair  # noqa: E402
from transformer_model import STRAT_FEATURES, AzBattleTransformer, load_checkpoint, save_checkpoint  # noqa: E402

HIRE = {"ev": "hire", "t": 3, "p": "Blue", "heroes": 1, "res": [0, 0, 0, 0, 0, 0, 5000],
        "cands": [{"castle": 100, "slot": 1, "hero": 4, "race": 1, "lvl": 1, "val": 900, "army": 300.0},
                  {"castle": 100, "slot": 2, "hero": 9, "race": 2, "lvl": 2, "val": 1200, "army": 300.0}], "bi": 1}
BUILD = {"ev": "build", "t": 3, "p": "Blue", "castle": 100, "race": 1, "defensive": 0, "res": [5, 0, 5, 0, 0, 0, 3000],
         "cands": [{"b": 8, "trade": 0, "cost": [0, 0, 0, 0, 0, 0, 1000]}, {"b": 64, "trade": 0, "cost": [5, 0, 0, 0, 0, 0, 2000]}],
         "bi": 64}


def option_tokens( kind, event, view ):
    return strategy_net.query_tokens( kind, event, CONTEXT, [138], view=view )[2]


def test_student_view_hides_the_builtin_ai_from_hero_targets():
    """The planner's values, ranks and value order are invisible to the student, visible in the full view."""
    other = copy.deepcopy( TARGET )
    for cand, value in zip( other["cands"], ( 5.0, 700.0, 40.0 ) ):
        cand["v"] = value
    assert option_tokens( "target", TARGET, "student" ) == option_tokens( "target", other, "student" )
    assert option_tokens( "target", TARGET, "full" ) != option_tokens( "target", other, "full" )
    # Shown by distance: candidate 1 (d 100), 0 (d 200), 2 (d 900); the logits come back in the canonical order.
    assert strategy_net.option_order( "target", TARGET, "student" ) == [1, 0, 2]
    shown = torch.tensor( [[1.0, 2.0, 3.0]] )
    assert strategy_net.canonical_logits( shown, [[1, 0, 2]] ).tolist() == [[2.0, 1.0, 3.0]]


def test_student_view_hides_the_hire_choice_and_the_teacher_sees_it():
    hidden = option_tokens( "hire", HIRE, "student" )
    assert option_tokens( "hire", dict( HIRE, bi=0 ), "student" ) == hidden
    teacher = option_tokens( "hire", HIRE, "teacher" )
    assert [t[STRAT_FEATURES - 1] for t in teacher] == [0.0, 1.0, 0.0]  # candidate 1 = the built-in AI's hire
    assert all( t[STRAT_FEATURES - 1] == 0.0 for t in hidden )


def test_build_without_delegation_and_the_probe_as_the_teacher_hint():
    assert strategy_net.query_options( "build", BUILD, "full" )[0] == strategy_net.BUILTIN
    options = strategy_net.query_options( "build", BUILD, "student" )
    assert strategy_net.BUILTIN not in options and len( options ) == 3  # two buildings + nothing
    assert strategy_net.builtin_option( "build", options, BUILD ) == 1
    assert strategy_net.builtin_option( "build", options, dict( BUILD, bi=0 ) ) == 2  # nothing
    teacher = option_tokens( "build", BUILD, "teacher" )
    assert [t[STRAT_FEATURES - 1] for t in teacher] == [0.0, 1.0, 0.0]
    assert option_tokens( "build", BUILD, "student" ) == option_tokens( "build", dict( BUILD, bi=8 ), "student" )
    assert strategy_net.full_index( "build", 1, "student" ) == 2 and strategy_net.full_index( "hire", 1, "student" ) == 1


def test_net_policy_views_agree_with_the_training_log_probs( tmp_path ):
    torch.manual_seed( 0 )
    model = AzBattleTransformer()
    path = tmp_path / "m.pt"
    save_checkpoint( model, str( path ) )
    record = {"kind": "target", "event": TARGET, "context": CONTEXT, "history": {"days": [], "decisions": []}}
    for view in ( "student", "teacher" ):
        policy = strategy_net.NetStrategyPolicy( str( path ), view=view )
        policy.observe_turn( CONTEXT )
        probs = policy.probabilities( "target", TARGET )
        with torch.no_grad():
            trained = jolt.strategic_log_probs( model.eval(), [record], [], view )[0].exp().tolist()
        assert probs == pytest.approx( trained[:3], abs=1e-5 )


def battle_record():
    pair = make_pair( 0, 1 )
    return {"state": pair["state"], "legal": pair["legal"], "hint": [0.05, 0.6, 0.2, 0.05, 0.05, 0.05],
            "teacher": [1, 2, 1, 0], "student": [0, 3, 2, 2],
            "scores": {"0": [0.1, 0.0], "1": [0.9, 0.7], "2": [0.2, 0.3], "3": [-0.5, -0.4]}}


def test_an_untrained_teacher_equals_the_student_and_the_hint_moves_it():
    torch.manual_seed( 0 )
    model = AzBattleTransformer().eval()
    record = battle_record()
    with torch.no_grad():
        student, _, moves = jolt.battle_move_logps( model, [record], "cpu", hinted=False )
        teacher, _, _ = jolt.battle_move_logps( model, [record], "cpu", hinted=True )
        assert torch.allclose( student, teacher )
        model.hint_scale.fill_( 1.0 )
        hinted, _, _ = jolt.battle_move_logps( model, [record], "cpu", hinted=True )
        plain, _, _ = jolt.battle_move_logps( model, [record], "cpu", hinted=False )
    assert torch.allclose( plain, student )  # the hint layers never touch the student
    shot = moves.index( 1 )  # the hint's favourite
    assert float( hinted[shot] ) > float( student[shot] )


def test_battle_jolt_steps_teach_the_teacher_and_pull_the_student():
    torch.manual_seed( 0 )
    model = AzBattleTransformer()
    record = battle_record()
    loss = jolt.battle_loss( model, [record], "cpu", beta_t=0.01, beta_s=0.1 )
    loss.backward()
    assert torch.isfinite( loss ) and model.hint_scale.grad is not None and float( model.hint_scale.grad ) != 0.0
    optimizer = torch.optim.Adam( model.parameters(), lr=3e-3 )
    with torch.no_grad():
        before, _, moves = jolt.battle_move_logps( model.eval(), [record], "cpu", hinted=True )
    model.train()
    for _ in range( 10 ):
        optimizer.zero_grad()
        jolt.battle_loss( model, [record], "cpu", beta_t=0.01, beta_s=0.1 ).backward()
        optimizer.step()
    with torch.no_grad():
        after, _, _ = jolt.battle_move_logps( model.eval(), [record], "cpu", hinted=True )
    best = moves.index( 1 )  # the sampled move with the best rollouts
    assert float( after[best] ) > float( before[best] )
    metrics = jolt.battle_metrics( model, [record], "cpu" )
    assert set( metrics ) == {"kl", "student_score", "teacher_score", "hint_score", "hint_scale"}


def test_game_records_centre_the_reward_within_each_role():
    answer = {"kind": "target", "event": TARGET, "ctx": 0, "answer": 1}
    games = [{"role": "teacher", "reward": 1.0, "days": [CONTEXT], "answers": [answer]},
             {"role": "teacher", "reward": 3.0, "days": [CONTEXT], "answers": [answer]},
             {"role": "student", "reward": -5.0, "days": [CONTEXT], "answers": [answer, dict( answer, answer=0 )]},
             {"role": "student", "reward": -5.0, "days": [CONTEXT], "answers": [answer]}]
    records = jolt.game_records( {"seed": 1, "map": "m", "games": games}, eps=0.0 )
    assert [r["role"] for r in records] == ["teacher", "teacher", "student", "student", "student"]
    assert [r["adv"] for r in records[:2]] == pytest.approx( [-1.0, 1.0] )
    assert records[3]["history"]["decisions"][0]["answer"] == 1  # the student's first answer of that game


def test_strategic_jolt_steps_follow_the_advantage_and_the_teacher():
    torch.manual_seed( 0 )
    model = AzBattleTransformer()
    base = {"kind": "target", "event": TARGET, "context": CONTEXT, "history": {"days": [], "decisions": []}, "seed": 1, "map": "m"}
    teacher_good = dict( base, role="teacher", answer=2, adv=1.0 )
    teacher_bad = dict( base, role="teacher", answer=0, adv=-1.0 )
    student = dict( base, role="student", answer=0, adv=0.0 )
    optimizer = torch.optim.Adam( model.parameters(), lr=1e-3 )
    with torch.no_grad():
        start = jolt.strategic_log_probs( model.eval(), [base], [], "teacher" )[0]
    model.train()
    for _ in range( 10 ):
        optimizer.zero_grad()
        jolt.strategic_loss( model, [teacher_good, teacher_bad, student], [], beta_t=0.01, beta_s=0.1 ).backward()
        optimizer.step()
    with torch.no_grad():
        end = jolt.strategic_log_probs( model.eval(), [base], [], "teacher" )[0]
    assert float( end[2] ) > float( start[2] ) and float( end[0] ) < float( start[0] )
    metrics = jolt.strategic_metrics( model, [student], [] )
    assert 0.0 <= metrics["student_builtin"] <= 1.0 and metrics["kl"] >= 0.0


def test_checkpoints_without_the_hint_layers_load_with_a_silent_teacher( tmp_path ):
    model = AzBattleTransformer()
    state = {k: v for k, v in model.state_dict().items() if not k.startswith( "hint_" )}
    path = tmp_path / "old.pt"
    torch.save( {"arch": "transformer", "config": model.config, "state_dict": state}, str( path ) )
    loaded = load_checkpoint( str( path ) )
    assert float( loaded.hint_scale.detach() ) == 0.0 and float( loaded.hint_proj.weight.detach().abs().sum() ) == 0.0


def test_the_teacher_runs_on_bf16_weights():
    """Collectors run the joint net with bf16 weights (--bf16): the hint inputs must follow the weights' dtype."""
    model = AzBattleTransformer().eval().to( torch.bfloat16 )  # what for_inference does on a bf16 CPU
    probs = jolt._distribution( model, battle_record(), "cpu", hinted=True )
    assert abs( sum( probs ) - 1.0 ) < 1e-2


def test_win_counts_per_role_and_model( tmp_path ):
    import jolt_status

    games = tmp_path / "g.jsonl"
    group = {"model_time": 5.0, "hero_range": 6.0, "games": [
        {"role": "teacher", "reward": 6.0}, {"role": "teacher", "reward": 0.5, "outcome": 0},
        {"role": "student", "reward": -6.0}, {"role": "student", "reward": -1.0, "outcome": 0}]}
    games.write_text( json.dumps( group ) + "\n" )
    table = jolt_status.game_table( [str( games )] )
    assert table[5.0]["teacher"] == [( 1, 6.0 ), ( 0, 0.5 )]  # legacy line: won at the end of the scale
    assert table[5.0]["student"] == [( -1, -6.0 ), ( 0, -1.0 )]
    assert "won 1 (50.0%), lost 0, final duels won 2" in jolt_status.game_line( table[5.0]["teacher"] )

    battles = tmp_path / "b.jsonl"
    battles.write_text( json.dumps( {"model_time": 5.0, "teacher": [1, 1], "student": [0], "wins": {"0": [-1, 1], "1": [1, 1]}} ) + "\n" )
    ( tmp_path / "b.battles" ).write_text( json.dumps( {"model_time": 5.0, "result": 1} ) + "\n" )
    table = jolt_status.battle_table( [str( battles )] )
    assert table[5.0]["walked"] == [1] and table[5.0]["teacher"] == [1.0, 1.0] and table[5.0]["student"] == [0.5]


def test_the_battle_anchor_is_the_teachers_cross_entropy_to_its_hint():
    torch.manual_seed( 0 )
    model = AzBattleTransformer().eval()
    record = battle_record()
    with torch.no_grad():
        plain = jolt.battle_loss( model, [record], "cpu", beta_t=0.01, beta_s=0.1 )
        anchored = jolt.battle_loss( model, [record], "cpu", beta_t=0.01, beta_s=0.1, anchor=0.5 )
        teacher, _, moves = jolt.battle_move_logps( model, [record], "cpu", hinted=True )
    hint = [record["hint"][m] for m in moves]
    expected = -sum( h / sum( hint ) * float( lp ) for h, lp in zip( hint, teacher ) )
    assert float( anchored - plain ) == pytest.approx( 0.5 * expected, rel=1e-4 )
    # Only the anchor: the teacher moves towards the hint's favourite.
    unsampled = {k: v for k, v in record.items() if k not in ( "teacher", "student" )}
    optimizer = torch.optim.Adam( model.parameters(), lr=3e-3 )
    favourite = moves.index( max( range( len( record["hint"] ) ), key=record["hint"].__getitem__ ) )
    model.train()
    for _ in range( 10 ):
        optimizer.zero_grad()
        jolt.battle_loss( model, [unsampled], "cpu", beta_t=0.0, beta_s=0.0, plus=False, anchor=1.0 ).backward()
        optimizer.step()
    with torch.no_grad():
        after, _, _ = jolt.battle_move_logps( model.eval(), [record], "cpu", hinted=True )
    assert float( after[favourite] ) > float( teacher[favourite] )


def test_the_strategic_anchor_is_the_teachers_log_probability_of_the_builtin_answer():
    torch.manual_seed( 0 )
    model = AzBattleTransformer().eval()
    base = {"kind": "target", "event": TARGET, "context": CONTEXT, "history": {"days": [], "decisions": []}, "seed": 1, "map": "m"}
    records = [dict( base, role="teacher", answer=2, adv=1.0 ), dict( base, role="student", answer=0, adv=-1.0 )]
    builtin = strategy_net.builtin_option( "target", strategy_net.query_options( "target", TARGET, "student" ), TARGET )
    assert builtin is not None
    with torch.no_grad():
        plain = jolt.strategic_loss( model, records, [], beta_t=0.01, beta_s=0.1 )
        anchored = jolt.strategic_loss( model, records, [], beta_t=0.01, beta_s=0.1, anchor=0.3 )
        teacher = jolt.strategic_log_probs( model, records, [], "teacher" )
    assert float( anchored - plain ) == pytest.approx( -0.3 * float( teacher[:, builtin].mean() ), rel=1e-4 )
