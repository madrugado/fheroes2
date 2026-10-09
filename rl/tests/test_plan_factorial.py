"""Unit test of the factorial analysis (synthetic sweep files, no engine)."""

import itertools
import json
import os
import random
import sys

sys.path.insert( 0, os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), ".." ) )

import plan_factorial  # noqa: E402


def test_main_effects_and_interactions_are_recovered( tmp_path ):
    rng = random.Random( 1 )
    letters = list( plan_factorial.FACTORS )
    for bits in itertools.product( ( 0, 1 ), repeat=len( letters ) ):
        cell = "".join( letter for letter, bit in zip( letters, bits ) if bit )
        x = {letter: ( 1 if bit else -1 ) for letter, bit in zip( letters, bits )}
        pairs = []
        for seed in range( 30 ):
            # S on: +0.4 (coefficient 0.2); S and R together: an interaction of +0.2 (coefficient 0.1)
            duel = 0.1 + 0.2 * x["S"] + 0.1 * x["S"] * x["R"] + rng.gauss( 0, 0.05 )
            pairs.append( {"seed": seed, "duel": duel, "str": 0, "k": 0, "outcome": 0, "h": 0, "g": 0} )
        with open( tmp_path / f"vs_builtin_f{cell}_1-30.json", "w" ) as f:
            json.dump( {"pairs": pairs}, f )
    rows = plan_factorial.load( str( tmp_path ) )
    assert len( rows ) == 32 * 30
    effects = {name: ( effect, low, high ) for name, effect, low, high in plan_factorial.analyze( rows, "duel", draws=100 )}
    assert abs( effects["mean"][0] - 0.1 ) < 0.01
    assert abs( effects["S"][0] - 0.4 ) < 0.02 and effects["S"][1] > 0
    assert abs( effects["SR"][0] - 0.2 ) < 0.02 and effects["SR"][1] > 0
    assert effects["R"][1] < 0 < effects["R"][2]  # no effect of R alone
