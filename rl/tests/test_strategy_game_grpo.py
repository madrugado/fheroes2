"""Unit tests for rl/strategy_game_grpo.py (no engine)."""

import os
import random
import sys

sys.path.insert( 0, os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), ".." ) )

import pytest  # noqa: E402

from strategy_game_grpo import ( SampleBranch, aux_advantages, aux_score, aux_trajectory, deviation_candidates,  # noqa: E402
                                group_records, per_kind )
from strategy_net import builtin_option, query_options  # noqa: E402


class FakePolicy:
    """A strategic policy with fixed probabilities, enough for SampleBranch's draws."""

    view = "full"

    def reset( self ) -> None:
        pass

    def observe_turn( self, turn_context: dict ) -> None:
        pass

    def probabilities( self, kind: str, event: dict ) -> list[float]:
        return [0.7, 0.2, 0.1]

    def record( self, kind: str, event: dict, index: int | None ) -> None:
        pass

    def history( self, color: str ) -> dict:
        return {"days": [], "decisions": []}


ARMY = {"ev": "army", "t": 3, "p": "Blue", "h": 1, "res": [0] * 7, "cands": []}
TARGET = {"ev": "decision", "t": 3, "p": "Blue", "h": 1, "cands": [{"i": 10}, {"i": 20}, {"i": 30}]}


def test_group_records_push_every_answer_by_its_game_advantage_with_the_rebuilt_history():
    days = [{"t": 1}, {"t": 2}]
    first = {"kind": "army", "event": {"p": "Blue", "n": 1}, "ctx": 0, "answer": 2, "p": 0.6}
    second = {"kind": "target", "event": {"p": "Blue", "n": 2}, "ctx": 1, "answer": 0, "p": 0.9}
    group = {"seed": 7, "map": "m", "games": [
        {"k": 0, "reward": 1.0, "days": days, "answers": [first, second]},
        {"k": 1, "reward": -1.0, "days": days, "answers": [dict( first, answer=1 )]},
    ]}
    records = group_records( group, eps=0.0 )
    assert len( records ) == 3
    a, b, c = records
    assert a["adv"] == {2: 1.0} and b["adv"] == {0: 1.0} and c["adv"] == {1: -1.0}
    assert a["context"] == {"t": 1} and a["history"] == {"days": [], "decisions": []}
    assert b["context"] == {"t": 2} and b["history"]["days"] == [{"t": 1}]
    assert b["history"]["decisions"] == [{"kind": "army", "event": first["event"], "context": {"t": 1}, "answer": 2}]
    assert group_records( {"seed": 1, "map": "m", "games": [dict( group["games"][0] ), dict( group["games"][0] )]} ) == []


def test_per_kind_parses_kind_value_pairs():
    assert per_kind( "target=1,build=3", "x" ) == {"target": 1.0, "build": 3.0}
    assert per_kind( "", "x" ) == {} and per_kind( " army =4 ", "x" ) == {"army": 4.0}
    with pytest.raises( SystemExit ):
        per_kind( "target", "x" )


def test_sample_branch_stores_the_temperature_of_every_answer():
    """Per-kind T (idea a): army drawn almost greedily at T=0.01, targets at the global T; every answer keeps its T."""
    branch = SampleBranch( FakePolicy(), "Blue", 5.0, random.Random( 1 ), {"army": 0.01}, {} )
    for _ in range( 40 ):
        branch._answer( "army", dict( ARMY ) )
        branch._answer( "target", dict( TARGET ) )
    army_answers = [a["answer"] for a in branch.answers if a["kind"] == "army"]
    assert all( a["T"] == 0.01 for a in branch.answers if a["kind"] == "army" )
    assert all( a["T"] == 5.0 for a in branch.answers if a["kind"] == "target" )
    assert army_answers.count( 0 ) / len( army_answers ) > 0.9  # T = 0.01 stays at the greedy option
    assert not any( a.get( "eps" ) for a in branch.answers )


def test_sample_branch_epsilon_answers_are_uniform_and_marked():
    """--kind-epsilon 1.0: every army answer is uniform-random among the options and marked 'eps'."""
    branch = SampleBranch( FakePolicy(), "Blue", 1.0, random.Random( 3 ), {}, {"army": 1.0} )
    for _ in range( 60 ):
        branch._answer( "army", dict( ARMY ) )
    answers = [a["answer"] for a in branch.answers]
    assert sorted( set( answers ) ) == [0, 1, 2]  # uniform, not the tempered draw
    assert all( a.get( "eps" ) == 1 and a["T"] == 1.0 for a in branch.answers )


def test_group_records_zero_the_epsilon_answers_and_keep_the_others():
    """An 'eps' answer is off-policy (advantage 0) and must not zero the following answers of its game."""
    days = [{"t": 1}]
    group = {"seed": 7, "map": "m", "games": [
        {"reward": 1.0, "days": days, "answers": [{"kind": "army", "event": ARMY, "ctx": 0, "answer": 2, "eps": 1, "T": 1.0}]},
        {"reward": -1.0, "days": days, "answers": [{"kind": "army", "event": ARMY, "ctx": 0, "answer": 1, "T": 4.0}]},
        {"reward": 1.0, "days": days, "answers": [
            {"kind": "army", "event": ARMY, "ctx": 0, "answer": 2, "eps": 1, "T": 1.0},
            {"kind": "target", "event": TARGET, "ctx": 0, "answer": 1}]},
        {"reward": -1.0, "days": days, "answers": [{"kind": "army", "event": ARMY, "ctx": 0, "answer": 0}]},
    ]}
    records = group_records( group, eps=0.0 )
    assert [r["adv"] for r in records] == [{2: 0.0}, {1: -1.0}, {2: 0.0}, {1: 1.0}, {0: -1.0}]
    assert records[0]["eps"] == 1 and records[0]["T"] == 1.0 and records[1]["T"] == 4.0
    assert "eps" not in records[1] and "T" not in records[3]


def test_sample_branch_stores_the_global_query_index():
    branch = SampleBranch( FakePolicy(), "Blue", 1.0, random.Random( 1 ) )
    for _ in range( 3 ):
        branch._answer( "army", dict( ARMY ) )
    assert [a["n"] for a in branch.answers] == [0, 1, 2]


def test_sample_branch_forced_answer_preserves_the_rng_stream():
    """The deviations replay branches (idea 3): the draws happen exactly as in the base game, only the forced query's
    answer is overridden — every later sampled answer stays the same."""
    free = SampleBranch( FakePolicy(), "Blue", 1.0, random.Random( 11 ) )
    for _ in range( 6 ):
        free._answer( "army", dict( ARMY ) )
    overridden = ( free.answers[1]["answer"] + 1 ) % 3
    forced_branch = SampleBranch( FakePolicy(), "Blue", 1.0, random.Random( 11 ), forced={1: overridden} )
    for _ in range( 6 ):
        forced_branch._answer( "army", dict( ARMY ) )
    assert not forced_branch.diverged
    assert forced_branch.answers[1]["answer"] == overridden and free.answers[1]["answer"] != overridden
    assert [a["answer"] for i, a in enumerate( forced_branch.answers ) if i != 1] == \
           [a["answer"] for i, a in enumerate( free.answers ) if i != 1]


def test_sample_branch_forced_answer_marks_divergence():
    branch = SampleBranch( FakePolicy(), "Blue", 1.0, random.Random( 11 ), forced={0: 0}, expected=dict( ARMY, t=ARMY["t"] + 1 ) )
    branch._answer( "army", dict( ARMY ) )
    assert branch.diverged


def test_deviation_candidates_picks_only_deviating_non_epsilon_answers():
    """The built-in army answer is 100% (option 2 of (0, 50, 100)); eps draws and built-in answers are not candidates."""
    branch = SampleBranch( FakePolicy(), "Blue", 2.0, random.Random( 1 ) )
    builtin = builtin_option( "army", query_options( "army", ARMY ), ARMY )
    other = ( builtin + 1 ) % 3
    branch.answers = [
        {"kind": "army", "event": dict( ARMY, n=0 ), "ctx": 0, "answer": builtin, "T": 2.0, "n": 0},
        {"kind": "army", "event": dict( ARMY, n=1 ), "ctx": 0, "answer": other, "T": 2.0, "n": 1},
        {"kind": "army", "event": dict( ARMY, n=2 ), "ctx": 0, "answer": ( builtin + 2 ) % 3, "eps": 1, "T": 2.0, "n": 2},
        {"kind": "army", "event": dict( ARMY, n=3 ), "ctx": 1, "answer": builtin, "T": 2.0, "n": 3},
        {"kind": "army", "event": dict( ARMY, n=4 ), "ctx": 1, "answer": other, "T": 2.0, "n": 4},
    ]
    assert deviation_candidates( branch, "full", 4, random.Random( 0 ) ) == [1, 4]
    assert deviation_candidates( branch, "full", 1, random.Random( 0 ) ) in ([1], [4])


def test_group_records_add_the_auxiliary_advantage():
    """Idea 4: the intermediate-events advantage is ADDED to the game advantage (eps answers stay 0; no separate key)."""
    days = [{"t": 1}, {"t": 2}]
    # scores (battles only): game 0 goes -2 -> 0 (won 2 after day 1), game 1 goes +2 -> 0 (lost 2 after day 1):
    # per-day centred [(-2, 0)] / [(2, 0)], terminal 0 centred -> the day-1 advantages are +2 / -2, day 2 is 0.
    aux0 = [{"d": 1, "w": 0, "l": 2, "k": 1, "v": 1}, {"d": 2, "w": 2, "l": 2, "k": 2, "v": 2}]
    aux1 = [{"d": 1, "w": 2, "l": 0, "k": 1, "v": 2}, {"d": 2, "w": 2, "l": 2, "k": 2, "v": 2}]
    group = {"seed": 7, "map": "m", "games": [
        {"k": 0, "reward": 1.0, "days": days, "answers": [
            {"kind": "army", "event": ARMY, "ctx": 0, "answer": 1},
            {"kind": "target", "event": TARGET, "ctx": 1, "answer": 0}], "aux": aux0},
        {"k": 1, "reward": -1.0, "days": days, "answers": [{"kind": "army", "event": ARMY, "ctx": 0, "answer": 0, "eps": 1}], "aux": aux1},
    ]}
    weights = {"battles": 1.0, "castles": 0.0, "level": 0.0}
    raw = 2.0 ** 0.5  # the raw day-1 advantages are +-2; the pooled std of [2, 0, -2, 0] normalizes them to +-sqrt(2)
    advantages = aux_advantages( group["games"], weights, 1.0, 0.0 )
    assert advantages[0][0] == pytest.approx( raw ) and advantages[0][1] == 0.0
    assert advantages[1][0] == pytest.approx( -raw ) and advantages[1][1] == 0.0
    plain = group_records( group, eps=0.0 )
    records = group_records( group, eps=0.0, aux_weight=0.5, aux=weights )
    assert records[0]["adv"][1] == pytest.approx( 1.0 + 0.5 * raw )  # game advantage 1.0 + the aux part of its day
    assert records[1]["adv"] == {0: 1.0}  # day 2 carried no aux advantage
    assert records[2]["adv"] == {0: 0.0}  # eps: no policy term, with or without aux
    assert plain[0]["adv"] == {1: 1.0} and plain[1]["adv"] == {0: 1.0} and plain[2]["adv"] == {0: 0.0}  # eps


def test_aux_advantages_and_score():
    """The aux score is the weighted sum of its parts; the advantages are centred per day (GAE, lam 1) and normalized;
    games without a trajectory get none."""
    weights = {"battles": 1.0, "castles": 0.5, "level": 0.25}
    assert aux_score( {"w": 3, "l": 1, "k": 2, "v": 4}, weights ) == pytest.approx( 2 + 1 + 1 )
    games = [
        {"aux": [{"d": 1, "w": 0, "l": 1, "k": 1, "v": 2}, {"d": 2, "w": 1, "l": 1, "k": 2, "v": 3}]},
        {"aux": [{"d": 1, "w": 2, "l": 0, "k": 1, "v": 3}, {"d": 2, "w": 2, "l": 0, "k": 1, "v": 4}]},
    ]
    advantages = aux_advantages( games, weights, 1.0, 0.1 )
    assert len( advantages ) == 2 and len( advantages[0] ) == 2
    # Day 2: both games gained the same over the day, centred -> 0; day 1 differs: the weaker game improved (+).
    assert advantages[0][1] == pytest.approx( 0.0, abs=1e-9 ) and advantages[1][1] == pytest.approx( 0.0, abs=1e-9 )
    assert advantages[0][0] > 0 > advantages[1][0]
    assert aux_advantages( [games[0], {"aux": []}], weights, 1.0, 0.1 ) == [[], []]


def test_aux_trajectory_counts_battles_castles_and_level():
    branch = SampleBranch( FakePolicy(), "Blue", 1.0, random.Random( 1 ) )
    branch.our_days = [
        {"t": 1, "castles": [{"n": "a"}], "heroes": [{"lvl": 1}]},
        {"t": 3, "castles": [{"n": "a"}, {"n": "b"}], "heroes": [{"lvl": 1}, {"lvl": 3}]},
    ]
    events = [
        {"ev": "battle_start", "bid": 1, "att": {"c": "B", "hero": "x"}, "def": {"c": "R", "hero": "y"}},
        {"ev": "battle_end", "bid": 1, "t": 1, "winner": "B"},
        {"ev": "battle_start", "bid": 2, "att": {"c": "R", "hero": "x"}, "def": {"c": "B", "hero": "y"}},
        {"ev": "battle_end", "bid": 2, "t": 2, "winner": "R"},  # our loss on day 2
        {"ev": "battle_start", "bid": 3, "att": {"c": "G"}, "def": {"c": "R"}},
        {"ev": "battle_end", "bid": 3, "t": 2, "winner": "G"},  # not ours
        {"ev": "battle_start", "bid": 4, "att": {"c": "B"}, "def": {"c": "R"}},
        {"ev": "battle_end", "bid": 4, "t": 3, "winner": "N"},  # a draw counts neither way
        {"ev": "battle_start", "bid": 5, "att": {"c": "B", "hero": "x"}, "def": {"c": "N"}},
        {"ev": "battle_end", "bid": 5, "t": 3, "winner": "B"},  # a won fight against neutral monsters: not a hero battle
    ]
    trajectory = aux_trajectory( branch, events, "Blue" )
    assert trajectory == [{"d": 1, "w": 1, "l": 0, "mw": 0, "ml": 0, "k": 1, "v": 1},
                          {"d": 3, "w": 1, "l": 1, "mw": 1, "ml": 0, "k": 2, "v": 3}]
    assert aux_score( trajectory[1], {"battles": 1.0} ) == 0.0
    assert aux_score( trajectory[1], {"battles": 1.0, "monsters": 0.5} ) == 0.5


def test_greedy_after_the_deviation_answers_the_most_probable_option():
    free = SampleBranch( FakePolicy(), "Blue", 5.0, random.Random( 3 ) )
    for _ in range( 12 ):
        free._answer( "army", dict( ARMY ) )
    greedy = SampleBranch( FakePolicy(), "Blue", 5.0, random.Random( 3 ), forced={2: 1}, greedy_after=2 )
    for _ in range( 12 ):
        greedy._answer( "army", dict( ARMY ) )
    assert [a["answer"] for a in greedy.answers[:2]] == [a["answer"] for a in free.answers[:2]]  # the base game's draws
    assert greedy.answers[2]["answer"] == 1
    assert all( a["answer"] == 0 and a.get( "greedy" ) for a in greedy.answers[3:] )  # FakePolicy's argmax
    assert any( a["answer"] != 0 for a in free.answers[3:] )  # T=5 draws would have deviated


def test_deviation_row_replays_with_the_games_color_and_greedy_rest( monkeypatch ):
    import types

    import strategy_game_grpo as sgg

    seen = []

    def fake_play( args, seed, days, branch, report_days=None, reseed=None ):
        seen.append( ( branch.color, branch.greedy_after, reseed ) )
        return {}, {"marker": branch.forced}, [], {}

    monkeypatch.setattr( sgg, "play", fake_play )
    monkeypatch.setattr( sgg, "hero_state", lambda env, results, color, seed: ( color, results["marker"] ) )
    monkeypatch.setattr( sgg, "hero_components", lambda env, ref, results, color, seed: ( 0.5 if color == "Green" else 9.0, 0.1 ) )
    args = types.SimpleNamespace( temperature=1.0, map="m", days=30 )
    branch = SampleBranch( FakePolicy(), "Green", 1.0, random.Random( 1 ) )
    branch.our_days = [{"t": 3}]
    event = dict( ARMY, p="Green" )
    answer = {"kind": "army", "event": event, "ctx": 0, "answer": 0, "n": 4}
    builtin = builtin_option( "army", query_options( "army", event, "full" ), event )
    if builtin == 0:
        answer["answer"] = 1
    row = sgg.deviation_row( args, FakePolicy(), "full", branch, 99, 7, "Green", answer, [], None, 3, "mean" )
    assert row is not None and row["color"] == "Green" and row["rest"] == "greedy"
    assert all( color == "Green" and after == 4 for color, after, _ in seen ) and len( seen ) == 6
    assert [reseed for _, _, reseed in seen[::2]] == [None, ( 4, 1 ), ( 4, 2 )]
    assert row["gap"][0] == pytest.approx( sgg.combine_hero_parts( 0.5, 0.1, "mean" ), abs=1e-4 )
