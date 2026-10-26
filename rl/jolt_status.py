"""Wins of the JOLT teacher and student (user: "the most important is the number of wins of both", 2026-10-26), from the
collectors' own on-policy data, per model version (the checkpoint the collector played; the training saves a new one at
every report):

  games   — strategy_game_grpo collect --jolt files: per role games, game wins / losses (`outcome`; older lines: the
            reward at the ends of the hero scale) and final duels won (reward > 0: our strongest hero needs less than its
            army to beat the reference hero half of the time), the mean reward;
  battles — jolt.py collect: the battles the STUDENT walked (<out>.battles: won / lost), and over the labeled decisions
            the share of won rollouts after the teacher's and after the student's sampled moves.

  rl/.venv/bin/python rl/jolt_status.py --games rl/data/jolt1/games_c*.jsonl --battles rl/data/jolt1/battles_b*.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

HERO_RANGE = 6.0  # strategy_games.HERO_LOG2_RANGE (equivalent_of puts won / lost games at the ends)


def _lines( paths: list[str] ):
    for path in paths:
        if not os.path.exists( path ):
            continue
        with open( path ) as f:
            for line in f:
                if line.strip():
                    try:
                        yield json.loads( line )
                    except json.JSONDecodeError:
                        continue  # a line still being written


def game_outcome( game: dict, scale: float ) -> int:
    if "outcome" in game:
        return int( game["outcome"] )
    reward = game["reward"]
    return 1 if reward >= scale - 1e-6 else ( -1 if reward <= -scale + 1e-6 else 0 )


def game_table( paths: list[str] ) -> dict:
    """{model_time: {role: [(outcome, reward), ...]}}"""
    table: dict = defaultdict( lambda: defaultdict( list ) )
    for group in _lines( paths ):
        scale = float( group.get( "hero_range", HERO_RANGE ) )
        for game in group["games"]:
            table[group.get( "model_time" )][game.get( "role", "-" )].append( ( game_outcome( game, scale ), game["reward"] ) )
    return table


def battle_table( paths: list[str] ) -> dict:
    """{model_time: {"walked": [result...], "teacher": [won share per sample...], "student": [...]}}"""
    table: dict = defaultdict( lambda: defaultdict( list ) )
    for path in paths:
        for summary in _lines( [os.path.splitext( path )[0] + ".battles"] ):
            table[summary.get( "model_time" )]["walked"].append( summary["result"] )
        for record in _lines( [path] ):
            wins = record.get( "wins" )
            if not wins:
                continue
            for role in ( "teacher", "student" ):
                for move in record.get( role ) or []:
                    rollouts = wins.get( str( move ) ) or []
                    if rollouts:
                        table[record.get( "model_time" )][role].append( sum( r == 1 for r in rollouts ) / len( rollouts ) )
    return table


def _stamp( model_time ) -> str:
    return time.strftime( "%m-%d %H:%M", time.localtime( model_time ) ) if model_time else "?"


def game_line( rows: list[tuple[int, float]] ) -> str:
    if not rows:
        return "-"
    wins = sum( o == 1 for o, _ in rows )
    losses = sum( o == -1 for o, _ in rows )
    duels = sum( r > 0 for _, r in rows )
    return ( f"{len( rows )} games: won {wins} ({wins / len( rows ):.1%}), lost {losses}, final duels won {duels} "
             f"({duels / len( rows ):.1%}), reward {statistics.fmean( r for _, r in rows ):+.2f}" )


def main() -> None:
    parser = argparse.ArgumentParser( description="Wins of the JOLT teacher and student per model version" )
    parser.add_argument( "--games", nargs="*", default=[] )
    parser.add_argument( "--battles", nargs="*", default=[] )
    args = parser.parse_args()

    games = game_table( args.games )
    if games:
        print( "strategy (whole games, our color):" )
        for model_time in sorted( games, key=lambda t: t or 0 ):
            for role in ( "teacher", "student" ):
                print( f"  model {_stamp( model_time )} {role:8s} {game_line( games[model_time][role] )}" )
    battles = battle_table( args.battles )
    if battles:
        print( "battles:" )
        for model_time in sorted( battles, key=lambda t: t or 0 ):
            row = battles[model_time]
            walked = row["walked"]
            text = f"  model {_stamp( model_time )} student walked {len( walked )} battles: won {sum( r == 1 for r in walked )}"
            if walked:
                text += f" ({sum( r == 1 for r in walked ) / len( walked ):.1%}), lost {sum( r == -1 for r in walked )}"
            for role in ( "teacher", "student" ):
                if row[role]:
                    text += f"; {role} moves: rollouts won {statistics.fmean( row[role] ):.1%} ({len( row[role] )})"
            print( text )


if __name__ == "__main__":
    main()
