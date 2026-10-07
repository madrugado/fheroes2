"""Unit tests for the oracle headroom summary (no engine)."""

import os
import sys

sys.path.insert( 0, os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), ".." ) )

import oracle_headroom  # noqa: E402


def rows_of( seed, option, values, builtin=0 ):
    """One row per salt 1..len(values): (hero, army) per salt."""
    return [{"seed": seed, "n": 1, "option": option, "builtin": builtin, "salt": salt, "hero": hero, "army": army,
             "won": False, "lost": False} for salt, ( hero, army ) in enumerate( values, start=1 )]


def test_the_oracle_gain_is_measured_on_fresh_lucks_only():
    zero = [( 0.0, 0.0 )] * 8
    rows = rows_of( 1, 0, zero )
    # Option 1 is truly better by +0.5 in both parts; option 2 only looked better on the selection lucks.
    rows += rows_of( 1, 1, [( 0.5, 0.5 )] * 8 )
    rows += rows_of( 1, 2, [( 2.0, 2.0 )] * 4 + [( -1.0, -1.0 )] * 4 )
    rows += rows_of( 2, 0, zero ) + rows_of( 2, 2, [( 2.0, 2.0 )] * 4 + [( -1.0, -1.0 )] * 4 )
    summary = oracle_headroom.summarize( rows, 4 )
    mean = summary["mean"]
    assert mean["deviations"] == 2
    # Seed 1: the lucky-looking option 2 is picked (2.0 > 0.5) and scores -1.0 on the fresh lucks; seed 2 the same.
    assert mean["selection_lucks_mean_gain"] == 2.0 and mean["fresh_mean"]["mean"] == -1.0
    # army_hero needs a clear army gain: option 2's gap is 2.0 with zero spread, so it is picked too.
    assert summary["army_hero"]["deviations"] == 2


def test_no_better_option_keeps_the_builtin_answer():
    rows = rows_of( 1, 0, [( 0.0, 0.0 )] * 4 ) + rows_of( 1, 1, [( -0.5, -0.5 )] * 4 )
    summary = oracle_headroom.summarize( rows, 2 )
    for rule in oracle_headroom.RULES:
        assert summary[rule]["deviations"] == 0 and summary[rule]["fresh_mean"]["mean"] == 0.0
