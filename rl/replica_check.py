"""Replica fidelity of the built-in battle AI: at every decision of seeded real games the engine reports its own
built-in AI's choice (FHEROES2_AGENT_EXPERT=1, "expert" in the decision query) and the headless replica is asked for
its choice ("suggest"); the agent plays the real choice, so the replica mirrors the whole real game. Counts agreements
(retreats/surrenders separately) and writes every mismatch with the commander's kingdom lists.

  rl/.venv/bin/python rl/replica_check.py --out /tmp/replica_mismatches.jsonl Battlefi.mp2:30:1-6 2kings.mp2:30:1-10

Result 2026-10-21 (Battlefi x6, Thechaos x4, 2kings x10 30 days, Arena x6 20 days): 33 168 decisions, all 31 retreats
and 3 surrenders agree, 23 other mismatches (one per battle, mostly round 2; cause not found).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

from battle_agent import BattleAgentRunner  # noqa: E402

ESCAPES = {6: "retreat", 7: "surrender"}


class ReplicaCheck( BattleAgentRunner ):
    def __init__( self, *args, totals: collections.Counter, out, seed: int, **kwargs ):
        super().__init__( *args, **kwargs )
        self.totals, self.out, self.game_seed = totals, out, seed

    def decide( self, state: dict ):
        real = state.get( "expert" )
        if self._replica_state is None and not self._replica_tried:
            self._replica_start( state )
        if real is None or real not in state["legal"]:
            self.totals["no_real_expert"] += 1
            return None
        if self._replica_env is None or self._replica_desynced or self._replica_state is None:
            self.totals["not_replicated"] += 1
        else:
            replica = ( self._replica_env.suggest() or {} ).get( "expert" )
            self.totals["compared"] += 1
            if replica != real:
                escape = {ESCAPES.get( real["act"] ), ESCAPES.get( ( replica or {} ).get( "act" ) )} - {None}
                self.totals["mismatch_escape" if escape else "mismatch_other"] += 1
                side = next( ( u["side"] for u in state["units"] if u["u"] == state["cur"] ), None )
                army = ( self._setup or {} ).get( side, {} )
                self.out.write( json.dumps( {"map": self.map_name, "seed": self.game_seed, "bid": state.get( "bid" ), "turn": state["turn"],
                                             "real": real, "replica": replica, "side": side, "hid": army.get( "hid" ),
                                             "kh": army.get( "kh" ), "kc": army.get( "kc" )} ) + "\n" )
            elif real["act"] in ESCAPES:
                self.totals["agree_" + ESCAPES[real["act"]]] += 1
        return real["act"], tuple( real["args"] )


def main() -> None:
    parser = argparse.ArgumentParser( description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter )
    parser.add_argument( "games", nargs="+", help="MAP:DAYS:FIRST-LAST seeds" )
    parser.add_argument( "--binary", default="./fheroes2" )
    parser.add_argument( "--out", default="/tmp/replica_mismatches.jsonl" )
    args = parser.parse_args()

    totals: collections.Counter = collections.Counter()
    with open( args.out, "w" ) as out:
        for spec in args.games:
            map_name, days, seeds = spec.split( ":" )
            first, last = ( int( value ) for value in seeds.split( "-" ) )
            for seed in range( first, last + 1 ):
                runner = ReplicaCheck( args.binary, map_name, int( days ), 1, "mcts", totals=totals, out=out, seed=seed,
                                       extra_env={"FHEROES2_AUTO_PLAYTEST_SEED": str( seed ), "FHEROES2_AGENT_EXPERT": "1"} )
                try:
                    runner.run()
                finally:
                    runner.close()
                print( map_name, seed, dict( totals ), flush=True )


if __name__ == "__main__":
    main()
