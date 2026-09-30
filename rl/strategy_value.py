"""Strategic value network (user decision 2026-09-29, after rl/label_noise.py showed that one-replay
war labels are coin flips): the unified transformer estimates a player's end-of-game score from
what the player has seen so far — its whole game history and today's snapshot (the same tokens as
the strategic queries, strategy_net.value_tokens) — through a value head at the context token.

    gen   — seeded 2kings games of the built-in AI in which every strategic query is answered by a
            random option with probability epsilon (so the net sees other answers than the built-in
            ones); for each player: every turn_context, every answer, and the end-of-game score:
            +2 won / -2 lost, else the mean duel of its strongest hero against every active rival's
            strongest (strategy_games.duel_all, 3 seeds x both sides) — the rule of our paired
            benchmarks (play_vs_builtin --duel).
    train — value regression on every (player, day) state of the games (MSE), held-out games for
            validation; an optional SFT anchor keeps the strategic policy output intact.
    eval  — held-out MSE / correlation by game phase.

Usage:
    rl/.venv/bin/python rl/strategy_value.py gen --seeds 2001-2300 --out rl/data/strategy_value_2kings.jsonl
    rl/.venv/bin/python rl/strategy_value.py train --model rl/models/unified_sft_ctx.pt \\
        --data rl/data/strategy_value_2kings.jsonl --sft-data rl/data/strategy_sft_2kings_v2.jsonl --out rl/models/unified_value.pt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from collections import defaultdict

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

from strategy_net import answer_of, query_options  # noqa: E402

KIND_OF_EVENT = {"decision": "target", "build": "build", "hire": "hire", "army": "army"}


class ExploringPolicy:
    """Every player's strategic answers: the built-in one, or with probability `epsilon` a uniformly
    random option. Records per color every turn_context and every answer (option index; the
    decision's context as an index into the color's days)."""

    def __init__( self, epsilon: float, rng: random.Random ):
        self.epsilon = epsilon
        self.rng = rng
        self.days: dict[str, list[dict]] = defaultdict( list )
        self.decisions: dict[str, list[dict]] = defaultdict( list )

    def observe_turn( self, turn_context: dict ) -> None:
        self.days[turn_context.get( "p" )].append( turn_context )

    def _answer( self, kind: str, event: dict ):
        from strategy_games import builtin_index

        options = query_options( kind, event )
        if len( options ) < 2:
            return None
        builtin = builtin_index( kind, options, event )
        index = builtin
        if builtin is None or self.rng.random() < self.epsilon:
            index = self.rng.randrange( len( options ) )
        color = event.get( "p" )
        self.decisions[color].append( {"kind": kind, "event": event, "ctx": len( self.days[color] ) - 1, "answer": index} )
        if index == builtin:
            return None  # the built-in choice itself (replays the built-in game exactly)
        return answer_of( options[index] )

    def __call__( self, decision: dict ):
        return self._answer( "target", decision )

    def build( self, event: dict ):
        return self._answer( "build", event )

    def hire( self, event: dict ):
        return self._answer( "hire", event )

    def army( self, event: dict ):
        return self._answer( "army", event )


def end_score( duel_env, results: dict, color: str, seed: int ) -> float:
    """+2 won / -2 lost, else our strongest hero's duels against every active rival's strongest."""
    from strategy_games import duel_all

    state = str( ( results.get( color ) or {} ).get( "s", "" ) )
    if state == "0":
        return 2.0
    if state == "1":
        return -2.0
    return duel_all( duel_env, results, color, seed )


def play_game( binary: str, map_name: str, days: int, seed: int, epsilon: float, duel_env ) -> list[dict]:
    """One exploring game; one trajectory record per player."""
    from strategy_env import StrategyEnv

    policy = ExploringPolicy( epsilon, random.Random( seed * 7919 + 1 ) )
    env = StrategyEnv( binary=binary, map_name=map_name, days=days, playthroughs=1, seed=seed )
    try:
        summaries = env.run( policy )
    finally:
        env.close()
    if not summaries:
        raise RuntimeError( f"seed {seed}: no game_end" )
    results = {r["c"]: r for r in summaries[-1].get( "results" ) or []}
    trajectories = []
    for color, contexts in policy.days.items():
        if color not in results:
            continue
        trajectories.append( {"seed": seed, "map": map_name, "color": color, "epsilon": epsilon, "end_day": summaries[-1].get( "day" ),
                              "final": end_score( duel_env, results, color, seed ), "days": contexts,
                              "decisions": policy.decisions.get( color, [] )} )
    return trajectories


# --- states -----------------------------------------------------------------------------------


def states_of( trajectory: dict ) -> list[tuple[dict, int]]:
    """(trajectory, day index) for every day the player saw."""
    return [( trajectory, index ) for index in range( len( trajectory["days"] ) )]


def state_history( trajectory: dict, index: int ) -> tuple[dict, dict]:
    """(today's context, history) of the player at the start of day `index`: the previous days and
    the answers given on them."""
    days = trajectory["days"]
    decisions = [dict( d, context=days[d["ctx"]] ) for d in trajectory["decisions"] if 0 <= d["ctx"] < index]
    return days[index], {"days": days[:index], "decisions": decisions}


def values( model, batch: list[tuple[dict, int]], obj_vocab: list[int] ):
    from strategy_net import MAX_STRATEGIC_TOKENS, value_tokens
    from transformer_model import strategic_value

    window = model.config.get( "window", MAX_STRATEGIC_TOKENS )
    states = []
    for trajectory, index in batch:
        context, history = state_history( trajectory, index )
        states.append( value_tokens( context, obj_vocab, history, window ) )
    return strategic_value( model, states )


def evaluate( model, states: list[tuple[dict, int]], obj_vocab: list[int], batch: int, mean_target: float ) -> dict:
    import torch

    model.eval()
    predictions, targets, days = [], [], []
    with torch.no_grad():
        for start in range( 0, len( states ), batch ):
            chunk = states[start:start + batch]
            predictions += values( model, chunk, obj_vocab ).tolist()
            targets += [trajectory["final"] for trajectory, _ in chunk]
            days += [trajectory["days"][index].get( "t", 0 ) for trajectory, index in chunk]
    report = {"states": len( targets ),
              "mse": statistics.fmean( ( p - t ) ** 2 for p, t in zip( predictions, targets ) ),
              "mse_of_the_mean": statistics.fmean( ( mean_target - t ) ** 2 for t in targets ),
              "corr": statistics.correlation( predictions, targets ) if len( set( predictions ) ) > 1 else 0.0}
    for name, low, high in ( ( "days 1-10", 1, 10 ), ( "days 11-25", 11, 25 ), ( "days 26+", 26, 10**6 ) ):
        rows = [( p, t ) for p, t, d in zip( predictions, targets, days ) if low <= d <= high]
        if len( rows ) > 2 and len( {p for p, _ in rows} ) > 1:
            report[name] = {"n": len( rows ), "mse": round( statistics.fmean( ( p - t ) ** 2 for p, t in rows ), 4 ),
                            "corr": round( statistics.correlation( [p for p, _ in rows], [t for _, t in rows] ), 4 )}
    return {k: round( v, 4 ) if isinstance( v, float ) else v for k, v in report.items()}


def load_trajectories( paths: list[str] ) -> list[dict]:
    trajectories = []
    for path in paths:
        with open( path ) as f:
            trajectories += [json.loads( line ) for line in f if line.strip()]
    return trajectories


def split_games( trajectories: list[dict], val_fraction: float ) -> tuple[list[dict], list[dict]]:
    games = sorted( {( t["map"], t["seed"] ) for t in trajectories} )
    random.Random( 0 ).shuffle( games )
    held = set( games[:max( 1, int( len( games ) * val_fraction ) )] )
    return [t for t in trajectories if ( t["map"], t["seed"] ) not in held], [t for t in trajectories if ( t["map"], t["seed"] ) in held]


def train( args ) -> None:
    import torch
    import torch.nn.functional as F

    from strategy_net import attach_history, obj_vocab_of
    from train_strategy_net import load_jsonl, option_log_probs, smoothed_nll
    from transformer_model import load_checkpoint, save_checkpoint

    torch.set_num_threads( args.threads )
    device = torch.device( "mps" if torch.backends.mps.is_available() else "cpu" )
    model = load_checkpoint( args.model, str( device ) )
    obj_vocab = list( model.config.get( "obj_vocab", [] ) )
    train_traj, val_traj = split_games( load_trajectories( args.data ), args.val )
    train_states = [s for t in train_traj for s in states_of( t )]
    val_states = [s for t in val_traj for s in states_of( t )]
    if args.val_max and len( val_states ) > args.val_max:
        val_states = random.Random( 0 ).sample( val_states, args.val_max )
    mean_target = statistics.fmean( t["final"] for t, _ in train_states )
    print( f"value: {len( train_traj )} train / {len( val_traj )} validation trajectories, {len( train_states )} / {len( val_states )} states, "
           f"mean target {mean_target:.3f}" )

    sft_anchor = []
    if args.sft_data and args.sft_weight > 0:
        sft_anchor = attach_history( load_jsonl( args.sft_data ) )
        if not obj_vocab:
            obj_vocab = obj_vocab_of( sft_anchor )
        print( f"sft anchor: {len( sft_anchor )} queries (weight {args.sft_weight})" )
    model.config["obj_vocab"] = obj_vocab

    print( f"before: {json.dumps( evaluate( model, val_states, obj_vocab, args.batch, mean_target ) )}" )
    optimizer = torch.optim.AdamW( model.parameters(), lr=args.lr, weight_decay=0.01 )
    model.body.gradient_checkpointing_enable( gradient_checkpointing_kwargs={"use_reentrant": False} )
    rng = random.Random( 1 )
    for epoch in range( args.epochs ):
        model.train()
        # Batches of similar history length (a batch is padded to its longest sequence).
        ordered = sorted( train_states, key=lambda s: ( s[1], rng.random() ) )
        batches = [ordered[start:start + args.batch] for start in range( 0, len( ordered ), args.batch )]
        rng.shuffle( batches )
        total, steps, started = 0.0, 0, time.time()
        for chunk in batches:
            predicted = values( model, chunk, obj_vocab )
            target = torch.tensor( [t["final"] for t, _ in chunk], dtype=torch.float32, device=device )
            loss = F.mse_loss( predicted, target )
            total += loss.item()
            if sft_anchor:
                sft_batch = rng.sample( sft_anchor, min( args.batch, len( sft_anchor ) ) )
                loss = loss + args.sft_weight * smoothed_nll( option_log_probs( model, sft_batch, obj_vocab ), [r["target"] for r in sft_batch], 0.1 )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_( model.parameters(), 1.0 )
            optimizer.step()
            steps += 1
            if steps % 50 == 0:
                print( f"  step {steps}/{len( batches )}: value mse {total / steps:.4f}, {time.time() - started:.0f}s", flush=True )
        print( f"epoch {epoch + 1}: value mse {total / max( steps, 1 ):.4f}" )
        print( f"epoch {epoch + 1}: {json.dumps( evaluate( model, val_states, obj_vocab, args.batch, mean_target ) )}" )
        save_checkpoint( model, args.out )
    print( f"model saved -> {args.out}" )


def main() -> None:
    parser = argparse.ArgumentParser( description="Strategic value network of the unified transformer" )
    sub = parser.add_subparsers( dest="cmd", required=True )
    gen = sub.add_parser( "gen" )
    gen.add_argument( "--binary", default="./fheroes2" )
    gen.add_argument( "--map", default="2kings.mp2" )
    gen.add_argument( "--days", type=int, default=45 )
    gen.add_argument( "--seeds", default="2001-2300" )
    gen.add_argument( "--epsilons", default="0,0.05,0.15", help="exploration rates, cycled over the seeds" )
    gen.add_argument( "--out", required=True )
    tr = sub.add_parser( "train" )
    tr.add_argument( "--model", required=True )
    tr.add_argument( "--data", nargs="+", required=True )
    tr.add_argument( "--sft-data", nargs="*", default=[] )
    tr.add_argument( "--sft-weight", type=float, default=0.5 )
    tr.add_argument( "--epochs", type=int, default=3 )
    tr.add_argument( "--batch", type=int, default=32 )
    tr.add_argument( "--lr", type=float, default=1e-4 )
    tr.add_argument( "--val", type=float, default=0.15 )
    tr.add_argument( "--val-max", type=int, default=3000 )
    tr.add_argument( "--threads", type=int, default=2 )
    tr.add_argument( "--out", required=True )
    args = parser.parse_args()
    sys.stdout.reconfigure( line_buffering=True )

    if args.cmd == "gen":
        from engine_bridge import BattleEnv
        from harvest_battles import parse_seeds

        epsilons = [float( e ) for e in args.epsilons.split( "," )]
        duel_env = BattleEnv( binary=args.binary, map_name=args.map )
        started = time.time()
        done = set()
        if os.path.exists( args.out ):  # resume: skip the games already written
            with open( args.out ) as f:
                done = {json.loads( line )["seed"] for line in f if line.strip()}
        try:
            with open( args.out, "a" ) as out:
                for number, seed in enumerate( parse_seeds( args.seeds ) ):
                    if seed in done:
                        continue
                    epsilon = epsilons[number % len( epsilons )]
                    try:
                        trajectories = play_game( args.binary, args.map, args.days, seed, epsilon, duel_env )
                    except ( TimeoutError, RuntimeError ) as error:
                        print( f"seed {seed}: failed ({error})" )
                        continue
                    for trajectory in trajectories:
                        out.write( json.dumps( trajectory, separators=( ",", ":" ) ) + "\n" )
                    out.flush()
                    finals = {t["color"]: round( t["final"], 2 ) for t in trajectories}
                    print( f"seed {seed} (eps {epsilon}): end day {trajectories[0]['end_day'] if trajectories else '-'}, finals {finals}, "
                           f"{time.time() - started:.0f}s" )
        finally:
            duel_env.close()
    else:
        train( args )


if __name__ == "__main__":
    main()
