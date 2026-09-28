"""Preference pairs for DPO on the battle policy, from counterfactual rollouts of real battles.

Walks real battles (harvest_battles.py setups) with the built-in AI on both sides. At sampled
decision points it compares a few candidate moves — the built-in AI's move, the policy's top
moves (--model) and a random legal move — by playing each to the end of the battle with the
built-in AI on both sides (battle server "restore" with "rollout": one roundtrip, ~3 ms). The
score of a move for the side to move is

    outcome (+1 win / 0 draw or round cap / -1 loss)
    + share of own army strength left - share of enemy army strength left

(strength = monster strength x count per stack, relative to the decision point). The best and
the worst candidate form a (chosen, rejected) pair when their scores differ by more than
--margin. Battles are deterministic, so the label is the exact consequence of the move under
built-in continuation — but a different move also reshuffles the random stream, so single
labels are noisy (a lucky roll can flip a battle).

Usage:
    az/.venv/bin/python az/battle_prefs.py --setups az/data/battles_Battlefi.jsonl ... \\
        --model az/models/az_battle_expert_v3.pt --out az/data/battle_prefs.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

from engine_bridge import BattleEnv, new_battle_from_setup  # noqa: E402
from gen_expert import STATE_KEYS, load_setups  # noqa: E402

SNAP_ID = 1
MAX_STEPS = 400


def side_strength( state: dict, side: str ) -> int:
    return sum( u.get( "str", u["q"] ) for u in state.get( "units", [] ) if u["side"] == side )


def move_score( root: dict, final: dict | None, side: str ) -> float:
    """Value of a rollout's final state for `side` (see the module doc)."""
    if final is None or final.get( "ev" ) != "state":
        return float( "-inf" )
    enemy = "def" if side == "att" else "att"
    result = final.get( "result" )
    outcome = 1.0 if result == side else ( -1.0 if result == enemy else 0.0 )
    own0, enemy0 = side_strength( root, side ), side_strength( root, enemy )
    own = side_strength( final, side ) / own0 if own0 else 0.0
    killed = 1.0 - ( side_strength( final, enemy ) / enemy0 if enemy0 else 0.0 )
    return outcome + own + killed - 1.0


def mover_side( state: dict ) -> str:
    return next( ( u["side"] for u in state["units"] if u["u"] == state["cur"] ), "att" )


def candidates( state: dict, expert: dict | None, policy_value, top: int, rng: random.Random ) -> list[int]:
    """Indexes into state["legal"]: the built-in move, the policy's top moves, one random move."""
    legal = state["legal"]
    chosen: list[int] = []
    if expert is not None and expert in legal:
        chosen.append( legal.index( expert ) )
    if policy_value is not None:
        priors, _ = policy_value.evaluate( state )
        for i in sorted( priors, key=priors.get, reverse=True )[:top]:
            if i not in chosen:
                chosen.append( i )
    extra = rng.randrange( len( legal ) )
    if extra not in chosen:
        chosen.append( extra )
    return chosen


def label_battle( env: BattleEnv, setup: dict, policy_value, args, rng: random.Random, key: str ) -> list[dict]:
    """Walks one battle with the built-in AI; returns its preference records."""
    records: list[dict] = []
    state = new_battle_from_setup( env, setup )
    for _ in range( MAX_STEPS ):
        if state is None or state.get( "ev" ) != "state" or state.get( "result" ) or not state.get( "legal" ):
            break

        expert = ( env.suggest() or {} ).get( "expert" )
        if len( state["legal"] ) > 1 and rng.random() < args.rate:
            env.snapshot_save( SNAP_ID )
            side = mover_side( state )
            scores = {}
            for i in candidates( state, expert, policy_value, args.top, rng ):
                move = state["legal"][i]
                final = env.snapshot_restore( SNAP_ID, path=[( move["act"], move["args"] )], rollout=True )
                scores[i] = move_score( state, final, side )
            best = max( scores, key=scores.get )
            worst = min( scores, key=scores.get )
            if scores[best] - scores[worst] > args.margin:
                records.append( {
                    "state": {k: state[k] for k in STATE_KEYS if k in state},
                    "legal": state["legal"],
                    "chosen": best,
                    "rejected": worst,
                    "scores": {str( i ): round( s, 4 ) for i, s in scores.items()},
                    "expert": state["legal"].index( expert ) if expert in state["legal"] else None,
                    "side": side,
                    "battle": key,
                } )

        if expert is None:
            break
        # The main line follows the built-in AI (main-line ops restore the main line themselves,
        # so the rollouts above do not disturb it).
        state = env.action( expert["act"], expert["args"] )

    env.snapshots_free()
    return records


def main() -> None:
    parser = argparse.ArgumentParser( description="Battle preference pairs from counterfactual rollouts" )
    parser.add_argument( "--setups", type=str, nargs="+", required=True )
    parser.add_argument( "--model", type=str, default=None, help="policy whose top moves are candidates" )
    parser.add_argument( "--arch", choices=["resnet", "transformer"], default="resnet" )
    parser.add_argument( "--device", type=str, default="cpu" )
    parser.add_argument( "--top", type=int, default=2, help="policy top moves per decision" )
    parser.add_argument( "--rate", type=float, default=0.5, help="share of decisions labeled" )
    parser.add_argument( "--margin", type=float, default=0.05, help="minimal score gap of a pair" )
    parser.add_argument( "--limit", type=int, default=0, help="at most this many battles (0: all)" )
    parser.add_argument( "--seed", type=int, default=5 )
    parser.add_argument( "--out", type=str, default="az/data/battle_prefs.jsonl" )
    args = parser.parse_args()

    policy_value = None
    if args.model:
        import torch

        torch.set_num_threads( 2 )
        from selfplay import load_policy_value

        policy_value = load_policy_value( args.model, args.arch, args.device )

    rng = random.Random( args.seed )
    setups = load_setups( args.setups )
    if args.limit:
        setups = setups[:args.limit]

    envs: dict[str, BattleEnv] = {}
    total = 0
    t0 = time.time()
    os.makedirs( os.path.dirname( args.out ) or ".", exist_ok=True )
    with open( args.out, "w" ) as out:
        try:
            for n, setup in enumerate( setups ):
                map_name = setup.get( "map", "Arena.mp2" )
                if map_name not in envs:
                    envs[map_name] = BattleEnv( map_name=map_name )
                key = f"real:{map_name}:{setup.get( 'game_seed' )}:{setup['bid']}"
                try:
                    records = label_battle( envs[map_name], setup, policy_value, args, rng, key )
                except TimeoutError:
                    print( f"battle {key}: timeout, respawning the engine", flush=True )
                    envs.pop( map_name ).close()
                    continue
                for record in records:
                    out.write( json.dumps( record, separators=( ",", ":" ) ) + "\n" )
                total += len( records )
                if ( n + 1 ) % 50 == 0 or n == len( setups ) - 1:
                    print( f"battle {n + 1}/{len( setups )}: {total} pairs, {time.time() - t0:.0f}s", flush=True )
        finally:
            for env in envs.values():
                env.close()

    print( f"done: {total} pairs -> {args.out}" )


if __name__ == "__main__":
    main()
