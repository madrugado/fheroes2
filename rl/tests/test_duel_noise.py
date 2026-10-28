"""Unit tests for rl/duel_noise.py (no engine): the duel-noise decomposition of the summary."""

import os
import sys

sys.path.insert( 0, os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), ".." ) )

from duel_noise import fixed_label, summarize  # noqa: E402


def rows_of( game: int, values: list[float], fixed: bool = False ) -> list[dict]:
    return [{"seed": game, "m": m, "reward": value, "fixed": fixed, "state": "0" if fixed else "",
             "label": "hero", "duel_seeds": 5} for m, value in enumerate( values )]


def test_summarize_splits_duel_noise_from_between_game_differences():
    rows = rows_of( 1, [6.0] * 4, fixed=True )  # a won game: fixed label, never duelled
    rows += rows_of( 2, [1.0, -1.0, 1.0, -1.0] )  # duelled, noisy label: variance 1
    rows += rows_of( 3, [0.0, 0.0, 0.0, 0.0] )  # duelled, stable label: variance 0
    summary = summarize( rows, 0.77 )
    assert summary["games"] == 3 and summary["fixed_games"] == 1 and summary["duelled_games"] == 2
    assert summary["duel_variance"] == 0.5
    assert summary["duel_sd_of_one_measurement"] == round( 0.5 ** 0.5, 4 )
    assert summary["per_game_duel_sd_median"] == 0.5
    assert summary["duel_share_of_group_variance"] == round( 0.5 / ( 0.77 * 0.77 ), 3 )
    # between-game means: 6, 0, 0 -> mean 2, population variance (16 + 4 + 4) / 3
    assert summary["between_game_sd_across_seeds"] == round( 8 ** 0.5, 4 )
    assert summary["duel_seeds"] == [5] and summary["label"] == ["hero"]
    assert summary["hero_resolution"] == round( 12 / 2 ** 7, 4 )


def test_summarize_without_duels():
    summary = summarize( rows_of( 1, [6.0, 6.0], fixed=True ), 0.77 )
    assert summary["duel_variance"] == 0.0 and summary["duel_share_of_group_variance"] == 0.0
    assert summary["duelled_games"] == 0 and summary["fixed_games"] == 1


def test_fixed_label_matches_the_engine_label_ends():
    def args( label ):
        import types

        return types.SimpleNamespace( color="Blue", label=label )

    results = {"Blue": {"s": "0", "top": {"hid": 1}}, "Red": {"s": "", "top": {"hid": 2}}}
    assert fixed_label( args( "hero" ), results, "0", {"hid": 2} ) == -6.0
    assert fixed_label( args( "hero" ), results, "1", {"hid": 2} ) == 6.0
    assert fixed_label( args( "hero" ), results, "", {"hid": 2} ) is None
    assert fixed_label( args( "hero" ), results, "", None ) == 6.0
    assert fixed_label( args( "final" ), results, "0", {"hid": 2} ) == 1.0
    assert fixed_label( args( "final" ), results, "", None ) == 1.0
