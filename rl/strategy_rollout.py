"""Counterfactual rollout labels for every strategic choice (hero target, build, hire, army budget).

The built-in AI makes one choice per query, so plain games never show what another choice would
have led to. Seeded playthroughs are deterministic (FHEROES2_AUTO_PLAYTEST_SEED), which makes
exact counterfactuals cheap:

1. a base game with the given seed where every query is answered by the built-in AI (`skip`)
   enumerates the queries (the n-th strategic query is the same in every replay of that seed as
   long as all earlier answers are the built-in ones);
2. for a sampled query n of color p on day t, the BASELINE branch replays the game to day t + H
   with the built-in choices, and each ALTERNATIVE branch replays it identically but answers
   query n with option j (built-in AI afterwards);
3. the label of option j is the difference of p's kingdom stats (outcome, castles, heroes, army
   strength, gold) at day t + H between the alternative and the baseline branch.

Options per kind (see `options_of`): target — the top candidates by built-in value; build — the
candidate buildings + "nothing"; hire — the candidates + "nothing"; army — budgets 0/50/100%.
The built-in option is known for every kind (build: from the base game's build_result), its
label is 0 and it needs no branch. The day limit only ends the game (the AI does not know it), so
a branch cut at t + H is an exact prefix of a longer game (verified: identical AI logs). Every
alternative branch re-checks that query n is the expected one (determinism guard); stuck or
diverged branches are skipped with a log line.

Usage (keep --jobs low on a laptop; engines run under nice):
    rl/.venv/bin/python rl/strategy_rollout.py --map Battlefi.mp2 --seeds 201-216 --jobs 2
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

from strategy_bench import player_stats  # noqa: E402
from strategy_env import StrategyEnv  # noqa: E402
from strategy_model import options_of  # noqa: E402
from strategy_policies import NOTHING  # noqa: E402

STAT_KEYS = ( "outcome", "k", "h", "str", "g" )
KINDS = ( "target", "build", "hire", "army" )


def builtin_option( kind: str, event: dict, options: list, build_result: int | None = None ) -> int | None:
    """Index of the built-in AI's answer among `options` (None if unknown)."""
    if kind == "target":
        return 0
    if kind == "hire":
        bi = event.get( "bi", -1 )
        return bi if bi >= 0 else len( options ) - 1  # NOTHING is the last option
    if kind == "army":
        return options.index( 100 )
    if kind == "build":
        if build_result is None:
            return None
        if build_result == 0:
            return len( options ) - 1
        for index, option in enumerate( options ):
            if option != NOTHING and option["b"] == build_result:
                return index
        return None  # the built-in AI built something outside the offered list
    raise ValueError( kind )


class Branch:
    """Policy for one seeded game: built-in choices everywhere, except the answer `answer` to the
    strategic query with global index `pick_at` (queries of all kinds and colors are counted in
    arrival order). Records every query with the last turn_context of the deciding color."""

    def __init__( self, pick_at: int | None = None, answer=None, expected: dict | None = None ):
        self.pick_at = pick_at
        self.answer = answer
        self.expected = expected
        self.queries: list[dict] = []
        self.contexts: dict[str, dict] = {}
        self.diverged = False
        self.picked = False

    def observe_turn( self, turn_context: dict ) -> None:
        self.contexts[turn_context.get( "p" )] = turn_context

    def _answer( self, kind: str, event: dict ):
        index = len( self.queries )
        self.queries.append( {"n": index, "kind": kind, "event": event, "context": self.contexts.get( event.get( "p" ) )} )

        if index != self.pick_at:
            return None
        if self.expected is not None and not same_query( event, self.expected ):
            self.diverged = True
            return None

        self.picked = True
        return self.answer

    def __call__( self, decision: dict ):
        return self._answer( "target", decision )

    def build( self, event: dict ):
        return self._answer( "build", event )

    def hire( self, event: dict ):
        return self._answer( "hire", event )

    def army( self, event: dict ):
        return self._answer( "army", event )


def same_query( a: dict, b: dict ) -> bool:
    return all( a.get( key ) == b.get( key ) for key in ( "ev", "t", "p", "h", "from", "castle", "cands", "bi", "offer", "reason" ) )


def play( binary: str, map_name: str, days: int, seed: int, branch: Branch ) -> tuple[dict, list[dict]]:
    """Plays one seeded game with the branch policy; returns (game_end, choice records)."""
    env = StrategyEnv( binary=binary, map_name=map_name, days=days, playthroughs=1, seed=seed )
    records: list[dict] = []
    try:
        summaries = env.run( branch, on_decision=records.append )
    finally:
        env.close()
    if not summaries:
        raise RuntimeError( f"seed {seed}, days {days}: no game_end" )
    return summaries[-1], records


def select_queries( queries: list[dict], per_kind: dict[str, int], max_day: int, top: int, rng: random.Random ) -> list[dict]:
    """Queries worth labeling: at least two options, early enough for the horizon; up to
    `per_kind[kind]` per kind."""
    chosen = []
    for kind, count in per_kind.items():
        eligible = [q for q in queries if q["kind"] == kind and 1 <= q["event"].get( "t", 0 ) <= max_day and len( options_of( kind, q["event"], top ) ) >= 2]
        chosen += rng.sample( eligible, min( count, len( eligible ) ) )
    return sorted( chosen, key=lambda q: q["n"] )


def stat_delta( alt: dict, base: dict ) -> dict:
    return {key: alt[key] - base[key] for key in STAT_KEYS}


def make_record( seed: int, args, query: dict, index: int, option, builtin: int, delta: dict, base_stats: dict ) -> dict:
    return {
        "seed": seed,
        "map": args.map,
        "horizon": args.horizon,
        "n": query["n"],
        "kind": query["kind"],
        "event": query["event"],
        "context": query["context"],
        "option_index": index,
        "option": option,
        "builtin_index": builtin,
        "delta": delta,
        "base": base_stats,
    }


def label_seed( pool: ThreadPoolExecutor, args, seed: int ) -> list[dict]:
    rng = random.Random( seed )
    max_day = args.base_days - args.horizon

    base = Branch()
    _, base_records = play( args.binary, args.map, args.base_days, seed, base )
    # build_result of the base game, in query order (records and queries are aligned 1:1).
    results = [r.get( "result" ) for r in base_records]

    per_kind = {"target": args.per_target, "build": args.per_build, "hire": args.per_hire, "army": args.per_army}
    chosen = select_queries( base.queries, per_kind, max_day, args.top, rng )

    days_needed = sorted( {q["event"]["t"] + args.horizon for q in chosen} )
    baselines = dict( zip( days_needed, pool.map( lambda days: play( args.binary, args.map, days, seed, Branch() )[0], days_needed ) ) )

    jobs = []
    plans = []
    for query in chosen:
        options = options_of( query["kind"], query["event"], args.top )
        builtin = builtin_option( query["kind"], query["event"], options, results[query["n"]] )
        if builtin is None:
            continue
        # Bound the cost of queries with many options (build): a sample of the non-built-in ones.
        others = [i for i in range( len( options ) ) if i != builtin]
        if len( others ) > args.max_alternatives:
            others = sorted( rng.sample( others, args.max_alternatives ) )
        plans.append( ( query, options, builtin ) )
        jobs += [( query, index, options[index] ) for index in others]

    def run_alternative( job ):
        query, index, option = job
        days = query["event"]["t"] + args.horizon
        branch = Branch( pick_at=query["n"], answer=option, expected=query["event"] )
        try:
            game_end, _ = play( args.binary, args.map, days, seed, branch )
        except ( TimeoutError, RuntimeError ) as error:
            print( f"seed {seed}: query {query['n']} ({query['kind']}) option {index} failed: {error}", flush=True )
            return job, branch, None
        return job, branch, game_end

    records: list[dict] = []
    for query, options, builtin in plans:
        base_stats = player_stats( baselines[query["event"]["t"] + args.horizon] )[query["event"]["p"]]
        records.append( make_record( seed, args, query, builtin, options[builtin], builtin, {key: 0 for key in STAT_KEYS}, base_stats ) )

    builtin_of = {query["n"]: builtin for query, _, builtin in plans}
    for ( query, index, option ), branch, game_end in pool.map( run_alternative, jobs ):
        if game_end is None:
            continue
        if branch.diverged or not branch.picked:
            print( f"seed {seed}: query {query['n']} diverged in replay; skipped", flush=True )
            continue
        event = query["event"]
        base_stats = player_stats( baselines[event["t"] + args.horizon] )[event["p"]]
        alt_stats = player_stats( game_end )[event["p"]]
        records.append( make_record( seed, args, query, index, option, builtin_of[query["n"]], stat_delta( alt_stats, base_stats ), base_stats ) )

    return records


def convert_legacy( record: dict ) -> dict:
    """Target-only records of the first rollout version -> the generic format."""
    if "kind" in record:
        return record
    return {
        "seed": record["seed"],
        "map": record.get( "map" ),
        "horizon": record.get( "horizon" ),
        "n": record["n"],
        "kind": "target",
        "event": record["decision"],
        "context": record["context"],
        "option_index": record["j"],
        "option": record["cand"],
        "builtin_index": 0,
        "delta": record["delta"],
        "base": record.get( "base" ),
    }


def parse_seeds( text: str ) -> list[int]:
    if "-" in text:
        first, last = text.split( "-" )
        return list( range( int( first ), int( last ) + 1 ) )
    return [int( s ) for s in text.split( "," )]


def main() -> None:
    parser = argparse.ArgumentParser( description="Counterfactual rollout labels for strategic choices" )
    parser.add_argument( "--binary", type=str, default="./fheroes2" )
    parser.add_argument( "--map", type=str, default="Battlefi.mp2" )
    parser.add_argument( "--seeds", type=str, default="201-216", help="range 'a-b' or list 'a,b,c'" )
    parser.add_argument( "--per-target", type=int, default=6 )
    parser.add_argument( "--per-build", type=int, default=10 )
    parser.add_argument( "--per-hire", type=int, default=6 )
    parser.add_argument( "--per-army", type=int, default=6 )
    parser.add_argument( "--top", type=int, default=4, help="target candidates considered (by built-in value)" )
    parser.add_argument( "--max-alternatives", type=int, default=3, help="alternative options labeled per query" )
    parser.add_argument( "--horizon", type=int, default=7, help="days simulated after the query" )
    parser.add_argument( "--base-days", type=int, default=21, help="queries are sampled up to base-days - horizon" )
    parser.add_argument( "--jobs", type=int, default=2, help="engine processes in parallel (keep low on a laptop)" )
    parser.add_argument( "--out", type=str, default="rl/data" )
    args = parser.parse_args()

    os.makedirs( args.out, exist_ok=True )
    out_path = os.path.join( args.out, f"strategy_rollouts_all_{os.path.splitext( args.map )[0]}_h{args.horizon}.jsonl" )

    t0 = time.time()
    total = 0
    with ThreadPoolExecutor( max_workers=args.jobs ) as pool, open( out_path, "a" ) as out:
        for seed in parse_seeds( args.seeds ):
            records = label_seed( pool, args, seed )
            for record in records:
                out.write( json.dumps( record ) + "\n" )
            out.flush()
            total += len( records )
            kinds = {k: sum( 1 for r in records if r["kind"] == k ) for k in KINDS}
            print( f"seed {seed}: {len( records )} records {kinds} ({time.time() - t0:.0f}s)", flush=True )

    print( f"done: {total} records -> {out_path} in {time.time() - t0:.0f}s" )


if __name__ == "__main__":
    main()
