"""Whole games: our agent playing one color against the built-in AI on the other colors.

For every seed the engine plays one CONTROL game (every player on the built-in AI, both agent
channels open and answered with the built-in choice) and, for every requested color, one
TREATMENT game where that color is played by our agent: strategic decisions by --strategy
(builtin = the built-in choice) and its battle units by --battle (mcts: search in the synced
replica, optionally with a trained network). The opponents' battle units move by the built-in
battle AI (taken from the replica's "suggest" so the replica stays in sync; see
battle_agent.BattleAgentRunner._builtin_move). FHEROES2_AUTO_PLAYTEST_SEED makes the pair
comparable; the verdict per (seed, color) is strategy_bench's (outcome, castles, army strength).

Games run one after another (a treatment game runs two engines: the game and the replica).

Usage:
    rl/.venv/bin/python rl/play_vs_builtin.py --map 2kings.mp2 --days 30 --seeds 1-3 \\
        --battle mcts --model rl/models/az_battle_expert_v3.pt --sims 32
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

from game_agent import GameAgent  # noqa: E402
from harvest_battles import parse_seeds  # noqa: E402
from strategy_bench import compare, player_stats, summarize  # noqa: E402
from strategy_policies import DEFAULT_MODEL, STRATEGY_POLICIES, ForColor, builtin_policy, make_strategy_policy  # noqa: E402


def play_game( args, seed: int, color: str | None, model ) -> tuple[dict, dict]:
    """One seeded game; color None = control. Returns (game_end, battle statistics of our color)."""
    if color is None:
        strategy, battle = builtin_policy, "planner"
    else:
        strategy = ForColor( make_strategy_policy( args.strategy, random.Random( seed ), args.strategy_model ), color )
        battle = args.battle

    agent = GameAgent(
        strategy_policy=strategy,
        extra_env={"FHEROES2_AUTO_PLAYTEST_SEED": str( seed )},
        binary=args.binary,
        map_name=args.map,
        days=args.days,
        playthroughs=1,
        policy=battle,
        model=model,
        sims=args.sims,
        seed=seed,
        battle_color=color,
    )
    records: list[dict] = []
    try:
        summaries = agent.run( on_record=records.append )
    finally:
        agent.close()
    if not summaries:
        raise RuntimeError( f"seed {seed}, color {color}: no game_end" )

    ours = [r for r in records if "cur" in r and r.get( "own" )]
    stats = {
        "battle_decisions": len( ours ),
        "desynced": sum( 1 for r in ours if r.get( "replica_synced" ) is False ),
    }
    return summaries[-1], stats


def main() -> None:
    parser = argparse.ArgumentParser( description="Our agent vs the built-in AI in whole games (paired by seed)" )
    parser.add_argument( "--binary", type=str, default="./fheroes2" )
    parser.add_argument( "--map", type=str, default="2kings.mp2" )
    parser.add_argument( "--days", type=int, default=30 )
    parser.add_argument( "--seeds", type=str, default="1-3" )
    parser.add_argument( "--colors", type=str, default="", help="comma-separated colors (default: every player)" )
    parser.add_argument( "--strategy", choices=list( STRATEGY_POLICIES ), default="builtin" )
    parser.add_argument( "--strategy-model", type=str, default=DEFAULT_MODEL )
    parser.add_argument( "--battle", choices=["planner", "policy", "mcts"], default="mcts" )
    parser.add_argument( "--sims", type=int, default=32 )
    parser.add_argument( "--model", type=str, default=None, help="battle network checkpoint" )
    parser.add_argument( "--arch", choices=["resnet", "transformer"], default="resnet" )
    parser.add_argument( "--device", type=str, default="cpu" )
    parser.add_argument( "--out", type=str, default="rl/data" )
    parser.add_argument( "--tag", type=str, default="" )
    parser.add_argument( "--duel", action="store_true",
                         help="also score the end of the game like the strategic label (strategy_games.war_score: "
                              "won/lost, else duels against every rival): treatment minus control" )
    args = parser.parse_args()

    model = None
    if args.model:
        from selfplay import load_policy_value

        model = load_policy_value( args.model, args.arch, args.device )

    duel_env = None
    if args.duel:
        from engine_bridge import BattleEnv

        duel_env = BattleEnv( binary=args.binary, map_name=args.map )

    def final_duel( game_end: dict, color: str, seed: int ) -> float:
        # The strategic label's rule at the end of the game: +2 won / -2 lost, else duels of our
        # strongest hero against every active rival's strongest (strategy_games.war_score).
        from strategy_games import war_score

        results = {r["c"]: r for r in game_end.get( "results" ) or []}
        return war_score( duel_env, results, [], color, 0, 0, seed )

    pairs = []
    t0 = time.time()
    for seed in parse_seeds( args.seeds ):
        control, _ = play_game( args, seed, None, model )
        control_stats = player_stats( control )
        colors = [c for c in args.colors.split( "," ) if c] or sorted( control_stats )
        print( f"seed {seed} control: day {control.get( 'day' )} "
               + ", ".join( f"{c} k{s['k']} str{s['str']}" for c, s in sorted( control_stats.items() ) ), flush=True )
        for color in colors:
            game_end, stats = play_game( args, seed, color, model )
            pair = compare( control_stats[color], player_stats( game_end )[color] )
            pair.update( seed=seed, color=color, changed=True, **stats )
            if duel_env is not None:
                pair["duel"] = final_duel( game_end, color, seed ) - final_duel( control, color, seed )
            pairs.append( pair )
            ours = player_stats( game_end )[color]
            print( f"seed {seed} {color}: {pair['verdict']} (outcome {pair['outcome']:+d}, castles {pair['k']:+d}, "
                   f"army {pair['str']:+.0f}, gold {pair['g']:+.0f}; ours k{ours['k']} str{ours['str']}; "
                   f"{stats['battle_decisions']} battle decisions, {stats['desynced']} desynced)", flush=True )

    if duel_env is not None:
        duel_env.close()

    summary = summarize( pairs )
    if duel_env is not None:
        from strategy_bench import bootstrap_ci

        duels = [p["duel"] for p in pairs]
        summary.update( mean_d_duel=round( sum( duels ) / len( duels ), 3 ), ci95_d_duel=bootstrap_ci( duels ),
                        duel_better=sum( 1 for d in duels if d > 0.05 ), duel_worse=sum( 1 for d in duels if d < -0.05 ) )
    summary.update( map=args.map, days=args.days, strategy=args.strategy, battle=args.battle, sims=args.sims,
                    model=args.model, seconds=round( time.time() - t0 ) )
    print( json.dumps( summary ) )

    os.makedirs( args.out, exist_ok=True )
    name = f"vs_builtin_{os.path.splitext( args.map )[0]}_{args.strategy}_{args.battle}{args.tag}.json"
    with open( os.path.join( args.out, name ), "w" ) as out:
        json.dump( {"summary": summary, "pairs": pairs}, out, indent=1 )


if __name__ == "__main__":
    main()
