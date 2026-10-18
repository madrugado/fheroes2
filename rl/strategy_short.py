"""Short-horizon strategic GRPO with the value network (user design 2026-10-18).

A strategic answer barely moves the END of a game, while the luck of the rest of the game moves it a lot: labels
from whole-game rollouts were noise (AGENTS.md, label noise / oracle headroom). Here an answer is judged a few days
later by the strategic value network instead:

  collect — seeded games (every player on `--plan`, built-in answers = the plan's). For sampled queries of our color
            (day t, at least two options) the built-in answer and `--random` other options are each replayed to day
            t + `--horizon` under `--salts` different lucks (salt 0 the plain game, salt k re-seeded from day t + 1,
            FHEROES2_RESEED). The game before the query is identical in every branch, so a record keeps the base game
            once and per branch only the suffix: our player's days and answers from the query on. No network needed.
  score   — rebuilds every branch's state at its last day (base prefix + suffix) and values it with a strategic net
            (strategy_value.values, hero label); the query becomes a `train_strategy_net pg` record: "salt_scores"
            {option: [value per luck]} (centred per query there: group advantages), with the query's history.

  rl/.venv/bin/python rl/strategy_short.py collect --map Battlefi.mp2 --days 30 --seeds 1-50 \\
      --plan champion=1,split_singles=1,max_heroes=3 --out rl/data/short_battlefi.jsonl
  rl/.venv/bin/python rl/strategy_short.py score --data rl/data/short_battlefi.jsonl --model rl/models/unified_200m_joint.pt \\
      --out rl/data/short_battlefi_pg.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
import time

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

from harvest_battles import parse_seeds  # noqa: E402
from strategy_games import PolicyBranch, builtin_index, play  # noqa: E402
from strategy_net import query_options  # noqa: E402

COLORS = ( "Blue", "Green", "Red", "Yellow", "Orange", "Purple" )


class BuiltinAnswers:
    """The strategic policy interface of PolicyBranch answering every query as the built-in AI (under the game's plan)."""

    def reset( self ) -> None:
        pass

    def observe_turn( self, turn_context: dict ) -> None:
        pass

    def history( self, color: str ) -> dict:
        return {"days": [], "decisions": []}  # rebuilt from the base game by `score`

    def decide( self, kind: str, event: dict ):
        options = query_options( kind, event )
        index = builtin_index( kind, options, event )
        return index if index is not None and 0 <= index < len( options ) else None  # None: the engine decides

    def record( self, kind: str, event: dict, choice ) -> None:
        pass


def query_position( branch: PolicyBranch, color: str, query: dict ) -> tuple[int, int] | None:
    """(number of our days up to and including the query's day, index of the query among our recorded answers)."""
    days = branch.days.get( color, [] )
    day_count = next( ( i + 1 for i, day in enumerate( days ) if day is query["context"] ), None )
    decision = next( ( i for i, d in enumerate( branch.decisions.get( color, [] ) ) if d["event"] is query["event"] ), None )
    if day_count is None or decision is None:
        return None
    return day_count, decision


def collect_game( args, seed: int, color: str, rng: random.Random ) -> list[dict]:
    policy = BuiltinAnswers()
    base = PolicyBranch( policy, color )
    play( args, seed, args.days, base )
    eligible = []
    for query in base.queries:
        options = query_options( query["kind"], query["event"] )
        t = query["event"].get( "t", 0 )
        if len( options ) >= 2 and 1 <= t <= args.days - args.horizon and query["answer_index"] is not None:
            position = query_position( base, color, query )
            if position is not None:
                eligible.append( ( query, position ) )
    chosen = sorted( rng.sample( eligible, min( args.per_game, len( eligible ) ) ), key=lambda item: item[0]["n"] )
    records = [{"type": "game", "seed": seed, "map": args.map, "color": color, "plan": args.plan, "base": base.trajectory( color )}]
    for query, ( day_count, decision ) in chosen:
        kind, event = query["kind"], query["event"]
        options = query_options( kind, event )
        own = query["answer_index"]
        others = [i for i in range( len( options ) ) if i != own]
        candidates = [own] + rng.sample( others, min( args.random, len( others ) ) )
        first_day = event["t"]
        branches: dict[str, list] = {}
        for index in candidates:
            suffixes = []
            for salt in range( args.salts ):
                branch = PolicyBranch( policy, color, pick_at=query["n"], answer_index=index, expected=event )
                try:
                    play( args, seed, first_day + args.horizon, branch, None, ( first_day + 1, salt ) if salt else None )
                except ( TimeoutError, RuntimeError ) as error:
                    print( f"seed {seed} query {query['n']} option {index} salt {salt}: failed ({error})", flush=True )
                    suffixes.append( None )
                    continue
                if branch.diverged or len( branch.days.get( color, [] ) ) < day_count:
                    suffixes.append( None )
                    continue
                suffixes.append( {"days": branch.days[color][day_count:], "decisions": branch.decisions.get( color, [] )[decision:]} )
            branches[str( index )] = suffixes
        records.append( {"type": "query", "seed": seed, "map": args.map, "color": color, "n": query["n"], "kind": kind, "event": event,
                         "day_count": day_count, "decision": decision, "own": own, "horizon": args.horizon, "branches": branches} )
    return records


def collect( args ) -> None:
    rng = random.Random( args.rng_seed )
    colors = args.colors.split( "," )
    started = time.time()
    with open( args.out, "a" ) as out:
        for number, seed in enumerate( parse_seeds( args.seeds ) ):
            color = colors[number % len( colors )]
            try:
                records = collect_game( args, seed, color, rng )
            except ( TimeoutError, RuntimeError ) as error:
                print( f"seed {seed}: game failed ({error})", flush=True )
                continue
            for record in records:
                out.write( json.dumps( record, separators=( ",", ":" ) ) + "\n" )
            out.flush()
            print( f"seed {seed} {color}: {len( records ) - 1} queries, {time.time() - started:.0f}s", flush=True )


def branch_trajectory( base: dict, record: dict, suffix: dict ) -> dict:
    """Our player's game in a branch: the base game up to the query + the branch's days and answers from the query on."""
    return {"days": base["days"][:record["day_count"]] + suffix["days"],
            "decisions": base["decisions"][:record["decision"]] + suffix["decisions"]}


def query_history( base: dict, record: dict ) -> tuple[dict, dict]:
    """(the query's context = our last day before it, the history the strategic net sees for the query)."""
    days = base["days"][:record["day_count"]]
    decisions = [{"kind": d["kind"], "event": d["event"], "context": base["days"][d["ctx"]], "answer": d["answer"]}
                 for d in base["decisions"][:record["decision"]]]
    return days[-1], {"days": days[:-1], "decisions": decisions}


class ShortReader:
    """Reads collect files incrementally (they may still be growing): every call returns the query records written
    since the previous one, each with its game's base trajectory; a partial last line is left for the next call."""

    def __init__( self, paths: list[str] ):
        self.paths = list( paths )
        self.offsets = {path: 0 for path in self.paths}
        self.games: dict[tuple, dict] = {}

    def new_queries( self ) -> list[tuple[dict, dict]]:
        found = []
        for path in self.paths:
            try:
                with open( path, "rb" ) as f:
                    f.seek( self.offsets[path] )
                    chunk = f.read()
            except FileNotFoundError:
                continue
            end = chunk.rfind( b"\n" )
            if end < 0:
                continue
            self.offsets[path] += end + 1
            for line in chunk[:end].splitlines():
                if not line.strip():
                    continue
                record = json.loads( line )
                key = ( record["map"], record["seed"], record["color"] )
                if record["type"] == "game":
                    self.games[key] = record["base"]
                elif key in self.games:
                    found.append( ( self.games[key], record ) )
        return found


def score_queries( model, queries: list[tuple[dict, dict]], value_scale: float, precision ) -> list[dict]:
    """`train_strategy_net pg` records: every branch's end state valued by the model ("salt_scores" per option)."""
    import torch

    import strategy_value

    obj_vocab = model.config["obj_vocab"]
    scored = []
    for base, record in queries:
        states, owners = [], []
        for option, suffixes in record["branches"].items():
            for salt, suffix in enumerate( suffixes ):
                if suffix is None or not suffix["days"]:
                    continue
                trajectory = branch_trajectory( base, record, suffix )
                states.append( ( trajectory, len( trajectory["days"] ) - 1 ) )
                owners.append( ( option, salt ) )
        if not states:
            continue
        with torch.no_grad(), precision:
            values = strategy_value.values( model, states, obj_vocab ).float().tolist()
        salt_scores: dict[str, list[float]] = {}
        for ( option, _ ), value in zip( owners, values ):
            salt_scores.setdefault( option, [] ).append( round( value * value_scale, 4 ) )
        context, history = query_history( base, record )
        scored.append( {"kind": record["kind"], "event": record["event"], "context": context, "history": history,
                        "seed": record["seed"], "map": record["map"], "n": record["n"], "color": record["color"],
                        "own": record["own"], "salt_scores": salt_scores} )
    return scored


def score( args ) -> None:
    import torch

    from transformer_model import load_checkpoint

    torch.set_num_threads( args.threads )
    model = load_checkpoint( args.model, "cpu" ).eval()
    precision = torch.autocast( "cpu", dtype=torch.bfloat16 ) if args.bf16 else contextlib.nullcontext()
    scored = score_queries( model, ShortReader( args.data ).new_queries(), args.value_scale, precision )
    with open( args.out, "w" ) as out:
        for record in scored:
            out.write( json.dumps( record, separators=( ",", ":" ) ) + "\n" )
    print( f"{len( scored )} scored queries -> {args.out}" )


def main() -> None:
    parser = argparse.ArgumentParser( description="Short-horizon strategic GRPO data valued by the strategic value network" )
    sub = parser.add_subparsers( dest="cmd", required=True )
    c = sub.add_parser( "collect" )
    c.add_argument( "--binary", default="./fheroes2" )
    c.add_argument( "--map", default="Battlefi.mp2" )
    c.add_argument( "--days", type=int, default=30 )
    c.add_argument( "--seeds", required=True )
    c.add_argument( "--colors", default=",".join( COLORS ), help="our color rotates over these, one per game" )
    c.add_argument( "--plan", default="champion=1,split_singles=1,max_heroes=3", help="FHEROES2_PLAN for every player" )
    c.add_argument( "--horizon", type=int, default=3, help="days after the query" )
    c.add_argument( "--salts", type=int, default=4, help="lucks per option" )
    c.add_argument( "--per-game", type=int, default=8, help="queries per game" )
    c.add_argument( "--random", type=int, default=3, help="random other options per query" )
    c.add_argument( "--rng-seed", type=int, default=1 )
    c.add_argument( "--out", required=True )
    s = sub.add_parser( "score" )
    s.add_argument( "--data", nargs="+", required=True )
    s.add_argument( "--model", required=True )
    s.add_argument( "--value-scale", type=float, default=3.0, help="value head output x this = label units (hero label: 3)" )
    s.add_argument( "--bf16", action="store_true" )
    s.add_argument( "--threads", type=int, default=2 )
    s.add_argument( "--out", required=True )
    args = parser.parse_args()
    collect( args ) if args.cmd == "collect" else score( args )


if __name__ == "__main__":
    main()
