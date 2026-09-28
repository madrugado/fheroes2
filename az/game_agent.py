"""One external agent for a whole fheroes2 game: strategic hero targets AND battle actions.

The engine is spawned with both FHEROES2_STRATEGY_SERVER=1 (AIDecision, see
src/fheroes2/ai/ai_decision.cpp) and FHEROES2_BATTLE_AGENT=1 (see
src/fheroes2/battle/battle_agent.cpp). Both channels share stdin/stdout, and the engine blocks
on exactly one protocol at a time (a hero decision, or a battle unit decision inside the battle
that a hero move started), so a single reader loop dispatches every event:

    turn_context / decision / build / hire -> strategic policy (strategy_policies.py)
    battle_start / state / battle_* ...  -> battle policy (BattleAgentRunner in battle_agent.py)
    game_end                             -> both (outcome attached to every record)

Records go to az/data/game_agent_<strategy>_<battle>.jsonl with a "kind" field
("target" | "build" | "hire" | "battle").

Usage:
    az/.venv/bin/python az/game_agent.py --strategy tempo --battle mcts --sims 16 --days 7
    az/.venv/bin/python az/game_agent.py --strategy learned --strategy-model az/models/strategy_model.json --battle mcts --sims 4
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

from battle_agent import DEFAULT_MAX_BATTLE_TURNS, BattleAgentRunner  # noqa: E402
from strategy_policies import DEFAULT_MODEL, STRATEGIC_QUERIES, STRATEGY_POLICIES, attach_build_result, make_strategy_policy, strategic_reply  # noqa: E402


class GameAgent( BattleAgentRunner ):
    """BattleAgentRunner that also serves the strategic decision channel."""

    def __init__( self, strategy_policy, extra_env: dict | None = None, **kwargs ):
        super().__init__( extra_env=dict( extra_env or {}, FHEROES2_STRATEGY_SERVER="1" ), **kwargs )
        self.strategy_policy = strategy_policy
        self._strategy_records: list[dict] = []
        self._on_strategy_record = None

    def run( self, on_record=None ) -> list[dict]:
        # Strategic records go to the same sink as the battle records.
        self._on_strategy_record = on_record
        return super().run( on_record=on_record )

    def _handle_event( self, ev: dict ) -> bool:
        kind = ev.get( "ev" )
        if kind == "turn_context":
            if hasattr( self.strategy_policy, "observe_turn" ):
                self.strategy_policy.observe_turn( ev )
            return True

        if kind in STRATEGIC_QUERIES:
            reply, record = strategic_reply( self.strategy_policy, ev )
            self._strategy_records.append( record )
            if self._on_strategy_record is not None:
                self._on_strategy_record( record )
            self._send( reply )
            return True

        if kind == "build_result":
            attach_build_result( self._strategy_records, ev )
            return True

        if kind == "game_end":
            for record in self._strategy_records:
                record["game_end"] = {"day": ev.get( "day" ), "results": ev.get( "results" )}
            self._strategy_records = []
            return False  # the battle side attaches the outcome to its records too

        return False


def main() -> None:
    parser = argparse.ArgumentParser( description="One external agent for strategic and battle decisions" )
    parser.add_argument( "--strategy", choices=list( STRATEGY_POLICIES ), default="tempo" )
    parser.add_argument( "--battle", choices=["random", "planner", "policy", "mcts"], default="planner" )
    parser.add_argument( "--strategy-model", type=str, default=DEFAULT_MODEL, help="model file for --strategy learned" )
    parser.add_argument( "--binary", type=str, default="./fheroes2" )
    parser.add_argument( "--map", type=str, default="Arena.mp2" )
    parser.add_argument( "--days", type=int, default=7 )
    parser.add_argument( "--playthroughs", type=int, default=1 )
    parser.add_argument( "--sims", type=int, default=32, help="MCTS simulations per battle decision" )
    parser.add_argument( "--max-battle-turns", type=int, default=DEFAULT_MAX_BATTLE_TURNS, help="rounds before the built-in AI takes over a battle" )
    parser.add_argument( "--model", type=str, default=None, help="trained battle network checkpoint (policy/mcts)" )
    parser.add_argument( "--arch", choices=["resnet", "transformer"], default="resnet" )
    parser.add_argument( "--device", type=str, default="cpu" )
    parser.add_argument( "--seed", type=int, default=2026 )
    parser.add_argument( "--out", type=str, default="az/data" )
    args = parser.parse_args()

    if args.battle == "policy" or ( args.battle == "mcts" and args.model is not None ):
        from selfplay import load_policy_value

        model = load_policy_value( args.model, args.arch, args.device )
        print( f"model loaded: {args.model} ({args.arch}) on {args.device}", flush=True )
    else:
        model = None

    agent = GameAgent(
        strategy_policy=make_strategy_policy( args.strategy, random.Random( args.seed ), args.strategy_model ),
        binary=args.binary,
        map_name=args.map,
        days=args.days,
        playthroughs=args.playthroughs,
        policy=args.battle,
        model=model,
        sims=args.sims,
        max_battle_turns=args.max_battle_turns,
        seed=args.seed,
    )

    os.makedirs( args.out, exist_ok=True )
    out_path = os.path.join( args.out, f"game_agent_{args.strategy}_{args.battle}.jsonl" )
    records: list[dict] = []

    def collect( record: dict ) -> None:
        record.setdefault( "kind", "battle" )
        records.append( record )

    t0 = time.time()
    try:
        for item in agent.run( on_record=collect ):
            print( f"game_end: day {item.get( 'day' )} results={item.get( 'results' )}", flush=True )
    finally:
        agent.close()

        with open( out_path, "w" ) as out:
            for record in records:
                out.write( json.dumps( record ) + "\n" )

    n_strategy = sum( 1 for r in records if r["kind"] != "battle" )
    print( f"done in {time.time() - t0:.1f}s: {n_strategy} strategic + {len( records ) - n_strategy} battle decisions -> {out_path}" )


if __name__ == "__main__":
    main()
