"""Is there anything to learn per strategic decision? (user request 2026-10-07)

Every DPO loop so far made the strategic net worse than the built-in AI. Before more labels, this
measures the headroom: an ORACLE that, at a query of our color in a seeded built-in game, picks the
option that looked best over SELECTION lucks, and the gain of that pick over the built-in answer on
FRESH lucks it was not chosen on (choosing the best of noisy estimates and scoring it on the same
replays always looks good — the selection bias). No gain on fresh lucks = single decisions carry no
learnable advantage at this noise level.

Per sampled query: the built-in answer and up to --options - 1 random other options, each replayed
to the end of the game under salts 1 .. 2K (FHEROES2_RESEED from the day after the query; salt 0 is
not used). Scores per salt: the --label hero parts against the built-in answer's game of the same
salt (strategy_games.hero_state / hero_components; the built-in answer is 0 by definition). Salts
1..K select, K+1..2K evaluate. Rules: "mean" (mean of the two parts), "army" and "army_hero" (army
gap > 2 SE over the selection lucks and the hero part not against; else the built-in answer).

Only the built-in AI plays (fast: a 45-day 2kings game takes a few seconds). Usage:
    rl/.venv/bin/python rl/oracle_headroom.py --seeds 401-410 --per-game 3 --out rl/data/oracle/part.jsonl
    rl/.venv/bin/python rl/oracle_headroom.py --summary-only --out rl/data/oracle/all.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
import types
from collections import defaultdict

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

import strategy_games  # noqa: E402
from engine_bridge import BattleEnv  # noqa: E402
from harvest_battles import parse_seeds  # noqa: E402
from label_noise import BuiltinBranchPolicy  # noqa: E402
from strategy_net import query_options  # noqa: E402

RULES = ( "mean", "army", "army_hero" )


def measure_game( args, seed: int, duel_env: BattleEnv, out ) -> int:
    """Replays the sampled queries of one seeded built-in game; one row per (option, salt)."""
    policy = BuiltinBranchPolicy()
    base = strategy_games.PolicyBranch( policy, args.color )
    strategy_games.play( args, seed, args.days, base )
    eligible = [q for q in base.queries
                if 1 <= q["event"].get( "t", 0 ) <= args.last_day and len( query_options( q["kind"], q["event"] ) ) >= 2
                and q["answer_index"] is not None]
    rng = random.Random( seed )
    rows = 0
    for query in sorted( rng.sample( eligible, min( args.per_game, len( eligible ) ) ), key=lambda q: q["n"] ):
        kind, event = query["kind"], query["event"]
        builtin = query["answer_index"]
        others = [i for i in range( len( query_options( kind, event ) ) ) if i != builtin]
        options = [builtin] + sorted( rng.sample( others, min( args.options - 1, len( others ) ) ) )
        day = event["t"]
        for salt in range( 1, 2 * args.lucks + 1 ):
            reference = None
            for index in options:
                branch = strategy_games.PolicyBranch( policy, args.color, pick_at=query["n"], answer_index=index, expected=event )
                try:
                    played = strategy_games.play( args, seed, args.days, branch, None, ( day + 1, salt ) )
                except ( TimeoutError, RuntimeError ) as error:
                    print( f"  seed {seed} query {query['n']} salt {salt} option {index}: failed ({error})", flush=True )
                    if index == builtin:
                        break  # nothing to compare with under this salt
                    continue
                if branch.diverged:
                    print( f"  seed {seed} query {query['n']}: replay diverged, skipped", flush=True )
                    break
                if index == builtin:
                    reference = strategy_games.hero_state( duel_env, played[1], args.color, seed )
                    hero, army = 0.0, 0.0
                else:
                    hero, army = strategy_games.hero_components( duel_env, reference, played[1], args.color, seed )
                state = str( ( played[1].get( args.color ) or {} ).get( "s", "" ) )
                out.write( json.dumps( {"seed": seed, "n": query["n"], "kind": kind, "t": day, "option": index, "builtin": builtin,
                                        "salt": salt, "hero": hero, "army": army, "won": state == "0", "lost": state == "1"},
                                       separators=( ",", ":" ) ) + "\n" )
                rows += 1
            out.flush()
    return rows


def gap( values: list[float] ) -> tuple[float, float]:
    """(mean, standard error)."""
    if len( values ) < 2:
        return ( values[0] if values else 0.0 ), float( "inf" )
    return statistics.fmean( values ), statistics.stdev( values ) / math.sqrt( len( values ) )


def oracle_pick( options: dict, builtin: int, rule: str, lucks: list[int] ) -> int:
    """The option the oracle picks from the selection lucks (the built-in answer when nothing is better)."""
    best, best_value = builtin, 0.0
    for index, by_salt in options.items():
        if index == builtin:
            continue
        rows = [by_salt[s] for s in lucks if s in by_salt]
        if len( rows ) < 2:
            continue
        army = gap( [r["army"] for r in rows] )
        hero = gap( [r["hero"] for r in rows] )
        if rule == "mean":
            value = statistics.fmean( ( r["hero"] + r["army"] ) / 2 for r in rows )
        elif rule == "army":
            value = army[0]
        else:  # army_hero: a clear army gain with the hero part not against
            value = army[0] if army[0] > 2 * army[1] and hero[0] >= 0 else 0.0
        if value > best_value:
            best, best_value = index, value
    return best


def bootstrap_ci( values: list[float], draws: int = 2000 ) -> list[float]:
    rng = random.Random( 0 )
    means = sorted( statistics.fmean( rng.choices( values, k=len( values ) ) ) for _ in range( draws ) )
    return [round( means[int( 0.025 * draws )], 4 ), round( means[int( 0.975 * draws )], 4 )]


def summarize( rows: list[dict], lucks: int ) -> dict:
    """Per rule: how often the oracle deviates from the built-in answer and its gain on FRESH lucks."""
    queries: dict[tuple, dict] = defaultdict( lambda: defaultdict( dict ) )
    builtins: dict[tuple, int] = {}
    for row in rows:
        key = ( row["seed"], row["n"] )
        queries[key][row["option"]][row["salt"]] = row
        builtins[key] = row["builtin"]
    selection = list( range( 1, lucks + 1 ) )
    evaluation = list( range( lucks + 1, 2 * lucks + 1 ) )
    summary: dict = {"queries": len( queries )}
    for rule in RULES:
        gains: dict[str, list[float]] = {"hero": [], "army": [], "mean": [], "won": [], "lost": []}
        deviations = 0
        in_sample = []
        for key, options in queries.items():
            builtin = builtins[key]
            pick = oracle_pick( options, builtin, rule, selection )
            if pick == builtin:
                for part in gains:
                    gains[part].append( 0.0 )
                continue
            deviations += 1
            fresh = [options[pick][s] for s in evaluation if s in options[pick] and s in options[builtin]]
            base = {s: options[builtin][s] for s in evaluation if s in options[builtin]}
            if not fresh:
                continue
            gains["hero"].append( statistics.fmean( r["hero"] for r in fresh ) )
            gains["army"].append( statistics.fmean( r["army"] for r in fresh ) )
            gains["mean"].append( statistics.fmean( ( r["hero"] + r["army"] ) / 2 for r in fresh ) )
            gains["won"].append( statistics.fmean( float( r["won"] ) - float( base[r["salt"]]["won"] ) for r in fresh ) )
            gains["lost"].append( statistics.fmean( float( r["lost"] ) - float( base[r["salt"]]["lost"] ) for r in fresh ) )
            chosen = [options[pick][s] for s in selection if s in options[pick]]
            in_sample.append( statistics.fmean( ( r["hero"] + r["army"] ) / 2 for r in chosen ) )
        summary[rule] = {
            "deviations": deviations,
            "selection_lucks_mean_gain": round( statistics.fmean( in_sample ), 4 ) if in_sample else None,
            **{f"fresh_{part}": {"mean": round( statistics.fmean( values ), 4 ), "ci95": bootstrap_ci( values )}
               for part, values in gains.items() if values},
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser( description="Oracle headroom of single strategic decisions" )
    parser.add_argument( "--binary", default="./fheroes2" )
    parser.add_argument( "--map", default="2kings.mp2" )
    parser.add_argument( "--days", type=int, default=45 )
    parser.add_argument( "--last-day", type=int, default=24, help="queries are sampled from days 1 .. this" )
    parser.add_argument( "--color", default="Blue" )
    parser.add_argument( "--seeds", default="401-410" )
    parser.add_argument( "--per-game", type=int, default=3 )
    parser.add_argument( "--options", type=int, default=4, help="the built-in answer + random others" )
    parser.add_argument( "--lucks", type=int, default=10, help="selection lucks; as many fresh lucks evaluate" )
    parser.add_argument( "--out", required=True )
    parser.add_argument( "--summary-only", action="store_true" )
    args = parser.parse_args()
    sys.stdout.reconfigure( line_buffering=True )

    if not args.summary_only:
        game_args = types.SimpleNamespace( binary=args.binary, map=args.map, days=args.days, horizons="21", label="hero",
                                           color=args.color, last_day=args.last_day, per_game=args.per_game, options=args.options,
                                           lucks=args.lucks )
        duel_env = BattleEnv( binary=args.binary, map_name=args.map )
        started = time.time()
        try:
            with open( args.out, "a" ) as out:
                for seed in parse_seeds( args.seeds ):
                    rows = measure_game( game_args, seed, duel_env, out )
                    print( f"seed {seed}: {rows} replays, {time.time() - started:.0f}s" )
        finally:
            duel_env.close()

    with open( args.out ) as f:
        rows = [json.loads( line ) for line in f if line.strip()]
    summary = summarize( rows, args.lucks )
    print( json.dumps( summary ) )
    with open( os.path.splitext( args.out )[0] + "_summary.json", "w" ) as f:
        json.dump( summary, f, indent=1 )


if __name__ == "__main__":
    main()
