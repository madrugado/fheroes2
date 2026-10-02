"""Continuous on-policy DPO of the unified net's strategic output (user request 2026-09-28).

Round after round:
  1. play seeded games with the CURRENT model (strategy_games.label_game, --label war on a 3-week
     horizon) until `--pairs` new preference pairs are collected;
  2. DPO from the current model (it is also the reference) on those pairs, with the SFT anchor
     (train_strategy_net.py dpo, a separate process); with --accumulate on ALL pairs so far (every
     earlier round of this --out directory + --extra-pairs), not only the new ones;
  3. paired games against the built-in AI on held-out seeds (play_vs_builtin.py --duel: the same
     end-of-game rule as the label);
  4. append the round's numbers to the progress log and continue from the new model.

Everything runs one step at a time (one game engine + one duel engine while collecting, nothing
while training) — the machine load rule of AGENTS.md; on a bigger machine --jobs N collects N games
at a time (one process each) and --threads sets the DPO step's torch threads. Stop it any time; `--resume` continues from
the progress log (the last model and the next seed) and keeps the pairs of an interrupted collection
(they are appended per whole game).

Usage:
    rl/.venv/bin/python rl/strategy_loop.py --model rl/models/unified_sft_rivals.pt --rounds 5 \\
        --out rl/data/strategy_loop
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import subprocess
import sys
import time
import types

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

from engine_bridge import BattleEnv  # noqa: E402
from strategy_net import NetStrategyPolicy  # noqa: E402

PYTHON = sys.executable
HERE = os.path.dirname( os.path.abspath( __file__ ) )


_worker: dict = {}


def _worker_init( args, model: str ) -> None:
    """A collector process of --jobs: its own net (2 torch threads) and its own duel engine."""
    _worker["game_args"] = types.SimpleNamespace( binary=args.binary, map=args.map, days=args.days, horizons=str( args.horizon ), color=args.color,
                                                  per_game=args.per_game, random=1, margin=args.margin, label=args.label )
    _worker["policy"] = NetStrategyPolicy( model )
    _worker["duel_env"] = BattleEnv( binary=args.binary, map_name=args.map )


def _worker_game( seed: int ) -> list[dict]:
    import strategy_games

    return list( strategy_games.label_game( _worker["game_args"], _worker["policy"], seed, random.Random( seed ), _worker["duel_env"] ) )


def collect_parallel( args, model: str, first_seed: int, out_path: str ) -> tuple[int, int]:
    """collect() with args.jobs games at a time (one process each: game engine + duel engine + net);
    seeds are handed out in order, pairs are appended per finished game (any order), the games
    still running when enough pairs are in are finished and kept. Resume: seeds already in the file
    are skipped."""
    import concurrent.futures
    import multiprocessing

    pairs, done = 0, set()
    if os.path.exists( out_path ):
        with open( out_path ) as f:
            kept = [json.loads( line ) for line in f if line.strip()]
        pairs, done = len( kept ), {pair["seed"] for pair in kept}
        if kept:
            first_seed = max( done ) + 1
            print( f"  kept {pairs} pairs of an interrupted collection, continuing from seed {first_seed}", flush=True )
    seed = first_seed
    t0 = time.time()
    context = multiprocessing.get_context( "spawn" )
    with open( out_path, "a" ) as out, concurrent.futures.ProcessPoolExecutor( args.jobs, mp_context=context, initializer=_worker_init,
                                                                               initargs=( args, model ) ) as pool:
        running = {}
        while pairs < args.pairs or running:
            while pairs < args.pairs and len( running ) < args.jobs:
                running[pool.submit( _worker_game, seed )] = seed
                seed += 1
            finished, _ = concurrent.futures.wait( running, return_when=concurrent.futures.FIRST_COMPLETED )
            for future in finished:
                game_seed = running.pop( future )
                try:
                    game_pairs = future.result()
                except Exception as error:  # a stuck/failed game: skip its seed
                    print( f"  seed {game_seed}: failed ({error})", flush=True )
                    continue
                for pair in game_pairs:
                    out.write( json.dumps( pair, separators=( ",", ":" ) ) + "\n" )
                out.flush()
                pairs += len( game_pairs )
                print( f"  seed {game_seed}: {pairs}/{args.pairs} pairs, {time.time() - t0:.0f}s", flush=True )
    return pairs, seed


def collect( args, model: str, first_seed: int, out_path: str ) -> tuple[int, int]:
    """Plays games with `model` from `first_seed` until args.pairs pairs are written to out_path.
    Returns (pairs, next seed)."""
    import strategy_games

    if args.jobs > 1:
        return collect_parallel( args, model, first_seed, out_path )
    game_args = types.SimpleNamespace( binary=args.binary, map=args.map, days=args.days, horizons=str( args.horizon ), color=args.color,
                                       per_game=args.per_game, random=1, margin=args.margin, label=args.label )
    policy = NetStrategyPolicy( model )
    pairs = 0
    seed = first_seed
    if os.path.exists( out_path ):
        # An interrupted collection of this round: keep its pairs (written per whole game) and
        # continue after the last game that produced any.
        with open( out_path ) as f:
            done = [json.loads( line ) for line in f if line.strip()]
        if done:
            pairs, seed = len( done ), max( pair["seed"] for pair in done ) + 1
            print( f"  kept {pairs} pairs of an interrupted collection, continuing from seed {seed}", flush=True )
    duel_env = BattleEnv( binary=args.binary, map_name=args.map )
    t0 = time.time()
    try:
        with open( out_path, "a" ) as out:
            while pairs < args.pairs:
                game_pairs = list( strategy_games.label_game( game_args, policy, seed, random.Random( seed ), duel_env ) )
                for pair in game_pairs:
                    out.write( json.dumps( pair, separators=( ",", ":" ) ) + "\n" )
                out.flush()
                pairs += len( game_pairs )
                print( f"  seed {seed}: {pairs}/{args.pairs} pairs, {time.time() - t0:.0f}s", flush=True )
                seed += 1
    finally:
        duel_env.close()
    return pairs, seed


def run( command: list[str] ) -> str:
    """Runs a step in its own process (under nice) and returns its output."""
    result = subprocess.run( ["nice", "-n", "10", PYTHON, *command], capture_output=True, text=True, cwd=os.path.dirname( HERE ) )
    if result.returncode != 0:
        raise RuntimeError( f"{command[0]} failed:\n{result.stderr[-2000:]}" )
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser( description="Continuous on-policy strategic DPO" )
    parser.add_argument( "--model", required=True, help="starting unified checkpoint (SFT)" )
    parser.add_argument( "--rounds", type=int, default=5 )
    parser.add_argument( "--pairs", type=int, default=100, help="new pairs per round" )
    parser.add_argument( "--binary", default="./fheroes2" )
    parser.add_argument( "--map", default="2kings.mp2" )
    parser.add_argument( "--days", type=int, default=45 )
    parser.add_argument( "--horizon", type=int, default=21 )
    parser.add_argument( "--color", default="Blue" )
    parser.add_argument( "--per-game", type=int, default=8 )
    parser.add_argument( "--margin", type=float, default=0.1 )
    parser.add_argument( "--first-seed", type=int, default=1000 )
    parser.add_argument( "--sft-data", default="rl/data/strategy_sft_2kings.jsonl" )
    parser.add_argument( "--dpo-epochs", type=int, default=4 )
    parser.add_argument( "--eval-seeds", default="101-110" )
    parser.add_argument( "--eval-days", type=int, default=30 )
    parser.add_argument( "--out", default="rl/data/strategy_loop" )
    parser.add_argument( "--resume", action="store_true" )
    parser.add_argument( "--label", choices=["final", "war"], default="final",
                         help="final: the final duel at the end of the game (strategy_games.final_label); war: the 21-day war label" )
    parser.add_argument( "--jobs", type=int, default=1, help="games collected in parallel (each: 2 engines + a net on 2 torch threads)" )
    parser.add_argument( "--threads", type=int, default=2, help="torch threads of the DPO training step" )
    parser.add_argument( "--accumulate", action="store_true", help="DPO on all pairs collected so far, not only the round's" )
    parser.add_argument( "--eval-opponent", default=None,
                         help="the paired games are played against this strategic net (play_vs_builtin --opponent-model) "
                              "instead of the built-in AI" )
    parser.add_argument( "--base-model", default=None,
                         help="every round collects with and starts DPO from THIS model (not the previous round's); "
                              "with --accumulate: a fixed base and ever more pairs instead of a drifting chain" )
    parser.add_argument( "--extra-pairs", nargs="*", default=[], help="with --accumulate: earlier pair files to include" )
    args = parser.parse_args()
    sys.stdout.reconfigure( line_buffering=True )

    os.makedirs( args.out, exist_ok=True )
    progress_path = os.path.join( args.out, "progress.jsonl" )
    model, seed, first_round = args.model, args.first_seed, 1
    if args.resume and os.path.exists( progress_path ):
        with open( progress_path ) as f:
            rounds = [json.loads( line ) for line in f if line.strip()]
        if rounds:
            model, seed, first_round = rounds[-1]["model"], rounds[-1]["next_seed"], rounds[-1]["round"] + 1

    for round_index in range( first_round, first_round + args.rounds ):
        t0 = time.time()
        if args.base_model:
            model = args.base_model
        print( f"round {round_index}: collecting {args.pairs} pairs with {model} from seed {seed}", flush=True )
        pairs_path = os.path.join( args.out, f"pairs_r{round_index}.jsonl" )
        pairs, next_seed = collect( args, model, seed, pairs_path )

        new_model = os.path.join( args.out, f"model_r{round_index}.pt" )
        data = [pairs_path]
        if args.accumulate:
            data = args.extra_pairs + sorted( glob.glob( os.path.join( args.out, "pairs_r*.jsonl" ) ) )
        total = sum( 1 for path in data for line in open( path ) if line.strip() )
        print( f"round {round_index}: DPO on {total} pairs ({pairs} new) -> {new_model}", flush=True )
        dpo_log = run( [os.path.join( HERE, "train_strategy_net.py" ), "dpo", "--model", model, "--data", *data, "--sft-data", args.sft_data,
                        "--sft-weight", "0.1", "--label-smoothing", "0.1", "--epochs", str( args.dpo_epochs ), "--batch", "16", "--lr", "1e-4",
                        "--beta", "0.1", "--threads", str( args.threads ), "--out", new_model] )
        dpo_lines = [line for line in dpo_log.splitlines() if "dpo loss" in line or "accuracy" in line]

        print( f"round {round_index}: paired games on seeds {args.eval_seeds}", flush=True )
        eval_log = run( [os.path.join( HERE, "play_vs_builtin.py" ), "--map", args.map, "--days", str( args.eval_days ), "--seeds", args.eval_seeds,
                         "--strategy", "net", "--strategy-model", new_model, "--battle", "planner", "--duel", "--tag", f"_loop_r{round_index}",
                         "--out", args.out, *( ["--opponent-model", args.eval_opponent] if args.eval_opponent else [] )] )
        summary = json.loads( [line for line in eval_log.splitlines() if line.startswith( "{" )][-1] )

        record = {"round": round_index, "model": new_model, "pairs": pairs, "dpo_pairs": total, "seeds": [seed, next_seed - 1], "next_seed": next_seed,
                  "dpo": dpo_lines[-2:], "minutes": round( ( time.time() - t0 ) / 60, 1 ),
                  **{k: summary.get( k ) for k in ( "better", "equal", "worse", "mean_d_str", "ci95_d_str", "mean_d_duel", "ci95_d_duel",
                                                   "duel_better", "duel_worse" )}}
        with open( progress_path, "a" ) as f:
            f.write( json.dumps( record ) + "\n" )
        print( f"round {round_index}: {json.dumps( record )}", flush=True )

        model, seed = new_model, next_seed


if __name__ == "__main__":
    main()
