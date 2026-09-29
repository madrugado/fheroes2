"""label_noise.py: the summary (pure) and the engine's FHEROES2_RESEED (needs ./fheroes2)."""

import os
import sys

import pytest

sys.path.insert( 0, os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), ".." ) )

import label_noise  # noqa: E402
from strategy_env import StrategyEnv  # noqa: E402

REPO_ROOT = os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )
BINARY = os.path.join( REPO_ROOT, "fheroes2" )


def rows_of( query: int, a: list[float], b: list[float] ) -> list[dict]:
    rows = []
    for salt, ( x, y ) in enumerate( zip( a, b ) ):
        rows.append( {"seed": 1, "n": query, "kind": "build", "answer": "a", "salt": salt, "score": x} )
        rows.append( {"seed": 1, "n": query, "kind": "build", "answer": "b", "salt": salt, "score": y} )
    return rows


def test_summary_separates_luck_from_a_real_difference():
    # Query 1: b is better by 1.0 under every salt (a clear difference, luck shared by both answers).
    # Query 2: pure luck, the plain label (salt 0) says b > a, the salts say nothing.
    rows = rows_of( 1, [0.0, 0.1, -0.2, 0.3, 0.0], [1.0, 1.1, 0.8, 1.3, 1.0] ) + rows_of( 2, [0.0, 1.0, -1.0, 0.5, -0.5], [0.4, -1.0, 1.0, -0.5, 0.5] )
    summary = label_noise.summarize( rows )
    assert summary["queries"] == 2
    assert summary["queries_with_a_clear_difference"] == 1
    assert summary["plain_label_sign_agrees"] == "1/1"  # query 2's salt mean is exactly 0: no sign
    assert summary["mean_corr_a_b_over_salts"] == pytest.approx( ( 1.0 - 1.0 ) / 2 )


def contexts_of( reseed ):
    env = StrategyEnv( binary=BINARY, map_name="2kings.mp2", days=14, playthroughs=1, seed=4, reseed=reseed )
    contexts = []

    class Recorder:
        def observe_turn( self, ev ):
            contexts.append( ev )

        def __call__( self, ev ):
            return None

    try:
        env.run( Recorder() )
    finally:
        env.close()
    return contexts


@pytest.mark.skipif( not os.path.exists( BINARY ), reason="fheroes2 binary not built" )
def test_reseed_keeps_the_game_before_the_day_and_changes_it_after():
    plain, salted, again = contexts_of( None ), contexts_of( ( 8, 1 ) ), contexts_of( ( 8, 1 ) )
    assert salted == again  # deterministic
    before = [c for c in plain if c["t"] < 8]
    assert before and before == [c for c in salted if c["t"] < 8]
    assert [c for c in plain if c["t"] >= 8] != [c for c in salted if c["t"] >= 8]
