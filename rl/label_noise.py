"""How much of a strategic DPO label is chance (user request 2026-09-29).

The labels of strategy_games.py compare two answers to one query by ONE replay each: the war
score three weeks later (--label war) or the final duel at the end of the game (--label final). If most of that difference is luck, DPO cannot learn from
it. Here every sampled query of a seeded built-in game is answered two ways — the built-in answer
(a) and a random other option (b) — and each answer is replayed under several salts: the engine
re-seeds its random generator and shifts the world seed (battle luck) at the start of the day
after the query (FHEROES2_RESEED), so the game up to the answer is identical and only the luck
afterwards differs. Salt 0 is the plain replay the DPO labels use.

Per query we get score(a, salt) and score(b, salt); the summary compares
  - the luck spread of ONE answer (standard deviation over salts),
  - the difference between the answers (mean over salts of b - a) and its standard error,
  - whether the plain label (salt 0) has the sign of the salt-averaged difference,
  - the correlation of a and b over salts (how much common luck cancels in a paired label).

`--label hero` (2026-10-06): per salt the built-in answer's game (a) gives the reference hero (its
strongest rival's strongest hero, strategy_games.hero_state); both answers are scored against it:
hero = -hero_equivalent (higher = stronger), army = log2 of the strongest hero's army strength. The
summary is printed for the hero part, the army part and their mean.

Strictly one engine at a time (plus the duel server). Usage:
    rl/.venv/bin/python rl/label_noise.py --seeds 301-310 --per-game 3 --salts 4 --out rl/data/label_noise.jsonl
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
from strategy_net import query_options  # noqa: E402


class BuiltinBranchPolicy:
    """The built-in answer to every query, with the interface PolicyBranch expects."""

    def reset( self ) -> None:
        pass

    def observe_turn( self, _turn_context: dict ) -> None:
        pass

    def history( self, _color: str ) -> dict:
        return {"days": [], "decisions": []}

    def decide( self, kind: str, event: dict ) -> int | None:
        options = query_options( kind, event )
        return strategy_games.builtin_index( kind, options, event ) if len( options ) >= 2 else None

    def record( self, _kind: str, _event: dict, _index: int | None ) -> None:
        pass


def measure_game( args, seed: int, duel_env: BattleEnv, out ) -> int:
    """Replays the sampled queries of one seeded game under every salt; writes one row per replay."""
    policy = BuiltinBranchPolicy()
    base = strategy_games.PolicyBranch( policy, args.color )
    strategy_games.play( args, seed, args.days, base )
    horizon = strategy_games.horizons_of( args )[-1]
    eligible = [q for q in base.queries
                if 1 <= q["event"].get( "t", 0 ) <= args.days - horizon and len( query_options( q["kind"], q["event"] ) ) >= 2
                and q["answer_index"] is not None]
    rng = random.Random( seed )
    rows = 0
    for query in sorted( rng.sample( eligible, min( args.per_game, len( eligible ) ) ), key=lambda q: q["n"] ):
        kind, event = query["kind"], query["event"]
        builtin = query["answer_index"]
        other = rng.choice( [i for i in range( len( query_options( kind, event ) ) ) if i != builtin] )
        day = event["t"]
        for salt in range( args.salts + 1 ):
            reference = None  # --label hero: the hero_state of answer a under this salt
            for label, index in ( ( "a", builtin ), ( "b", other ) ):
                branch = strategy_games.PolicyBranch( policy, args.color, pick_at=query["n"], answer_index=index, expected=event )
                try:
                    # The final label is taken at the end of the game (as in the DPO branches).
                    until = args.days if args.label in ( "final", "hero" ) else day + horizon
                    played = strategy_games.play( args, seed, until, branch, None, ( day + 1, salt ) if salt else None )
                except ( TimeoutError, RuntimeError ) as error:
                    print( f"  seed {seed} query {query['n']} salt {salt} {label}: failed ({error})", flush=True )
                    continue
                if branch.diverged:
                    print( f"  seed {seed} query {query['n']}: replay diverged, skipped", flush=True )
                    continue
                stats = played[0]
                row = {"seed": seed, "n": query["n"], "kind": kind, "t": day, "answer": label, "option": index,
                       "salt": salt, "str": stats.get( "str" ), "k": stats.get( "k" )}
                if args.label == "hero":
                    if label == "a":
                        # Answer a is the reference game itself: 0 in both parts by definition.
                        reference = strategy_games.hero_state( duel_env, played[1], args.color, seed )
                        hero, army = ( None if reference["m"] is None else 0.0 ), 0.0
                    elif reference is None:
                        continue  # answer a failed under this salt: nothing to compare with
                    else:
                        hero, army = strategy_games.hero_components( duel_env, reference, played[1], args.color, seed )
                    row["hero"], row["army"] = hero, army
                    row["score"] = strategy_games.combine_hero_parts( hero, army, "mean" )
                else:
                    row["score"] = strategy_games.branch_scores( args, duel_env, played, day, seed )[0]
                out.write( json.dumps( row, separators=( ",", ":" ) ) + "\n" )
                out.flush()
                rows += 1
    return rows


def summarize( rows: list[dict], key: str = "score" ) -> dict:
    """Luck spread of one answer vs the difference between answers (see the module doc); `key` = the
    row field to look at (--label hero rows also carry "hero" and "army")."""
    queries: dict[tuple, dict] = defaultdict( lambda: {"a": {}, "b": {}} )
    for row in rows:
        if row.get( key ) is None:
            continue
        queries[( row["seed"], row["n"] )][row["answer"]][row["salt"]] = row[key]
        queries[( row["seed"], row["n"] )]["kind"] = row["kind"]

    spreads, diffs, standard_errors, agree, significant, correlations = [], [], [], [], 0, []
    usable = 0
    for query in queries.values():
        salts = sorted( s for s in query["a"] if s > 0 and s in query["b"] )
        if len( salts ) < 2:
            continue
        usable += 1
        a = [query["a"][s] for s in salts]
        b = [query["b"][s] for s in salts]
        spreads += [statistics.stdev( a ), statistics.stdev( b )]
        d = [y - x for x, y in zip( a, b )]
        mean_d = statistics.fmean( d )
        se = statistics.stdev( d ) / math.sqrt( len( d ) )
        diffs.append( mean_d )
        standard_errors.append( se )
        if abs( mean_d ) > 2 * se and mean_d != 0:
            significant += 1
        if 0 in query["a"] and 0 in query["b"]:
            plain = query["b"][0] - query["a"][0]
            if plain != 0 and mean_d != 0:
                agree.append( ( plain > 0 ) == ( mean_d > 0 ) )
        if statistics.pstdev( a ) > 0 and statistics.pstdev( b ) > 0:
            correlations.append( statistics.correlation( a, b ) )

    def mean( values ):
        return round( statistics.fmean( values ), 4 ) if values else None

    return {
        "queries": usable,
        "luck_sd_of_one_answer": mean( spreads ),
        "mean_abs_answer_difference": mean( [abs( d ) for d in diffs] ),
        "sd_of_answer_differences": round( statistics.pstdev( diffs ), 4 ) if len( diffs ) > 1 else None,
        "mean_se_of_a_difference": mean( standard_errors ),
        "queries_with_a_clear_difference": significant,
        "plain_label_sign_agrees": f"{sum( agree )}/{len( agree )}",
        "mean_corr_a_b_over_salts": mean( correlations ),
    }


def main() -> None:
    parser = argparse.ArgumentParser( description="Luck vs signal in the strategic war labels" )
    parser.add_argument( "--binary", default="./fheroes2" )
    parser.add_argument( "--map", default="2kings.mp2" )
    parser.add_argument( "--days", type=int, default=45 )
    parser.add_argument( "--horizons", default="21" )
    parser.add_argument( "--label", choices=["final", "hero", "war", "stats"], default="war",
                         help="final: the final duel at the end of the game (strategy_games.final_label)" )
    parser.add_argument( "--color", default="Blue" )
    parser.add_argument( "--seeds", default="301-310" )
    parser.add_argument( "--per-game", type=int, default=3 )
    parser.add_argument( "--salts", type=int, default=4, help="reseeded replays per answer (plus the plain one)" )
    parser.add_argument( "--rng-streams", action="store_true", help="separate random streams per turn (FHEROES2_RNG_STREAMS)" )
    parser.add_argument( "--out", required=True )
    parser.add_argument( "--summary-only", action="store_true", help="summarize an existing --out file" )
    args = parser.parse_args()
    sys.stdout.reconfigure( line_buffering=True )

    if not args.summary_only:
        game_args = types.SimpleNamespace( binary=args.binary, map=args.map, days=args.days, horizons=args.horizons, label=args.label,
                                           color=args.color, per_game=args.per_game, salts=args.salts, rng_streams=args.rng_streams or None )
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
    summary = summarize( rows )
    if any( "hero" in row for row in rows ):
        # The parts of --label hero; "score" is their mean.
        summary = {"mean": summary, "hero": summarize( rows, "hero" ), "army": summarize( rows, "army" )}
    print( json.dumps( summary ) )
    with open( os.path.splitext( args.out )[0] + "_summary.json", "w" ) as f:
        json.dump( summary, f, indent=1 )


if __name__ == "__main__":
    main()
