"""Status of a running joint training (train_strategy_net.py) and its collectors, from their logs.

Run where the logs are (the training machine), from the repo root:

    rl/.venv/bin/python rl/run_status.py --train logs/joint16_train.log --collect 'logs/ggv2_c*.log' \\
        --data 'rl/data/game_grpo_v2_c*.jsonl' --mem logs/memwatch.log

From another machine (the address stays out of the repo — set it in the environment):

    FHEROES2_SERVER="user@host -p PORT" FHEROES2_DIR="~/path/to/fheroes2" rl/run_status.sh joint16 ggv2 game_grpo_v2

Prints: the tmux sessions, per report the strategic accuracy per query kind, battle imitation, the battle gate, the
strategic GRPO metrics and the whole-game check; the last step line (speed per part); collector progress (groups,
groups per hour, last line); memory (last and peak of the memwatch log).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import time

REPORT = re.compile( r"^(before|epoch \d+(?: step \d+)?):" )


def kinds( line: str ) -> str:
    match = re.search( r"strategic accuracy (\{[^}]*\})", line )
    if not match:
        return "-"
    accuracy = json.loads( match.group( 1 ) )
    return " ".join( f"{k} {accuracy[k].split()[0]}" for k in ( "target", "build", "hire", "army" ) if k in accuracy )


def grab( pattern: str, line: str, default: str = "-" ) -> str:
    match = re.search( pattern, line )
    return match.group( 1 ) if match else default


def report_lines( path: str, last: int ) -> list[str]:
    with open( path, errors="replace" ) as f:
        lines = [line.rstrip( "\n" ) for line in f if "warn" not in line.lower()]
    reports = [line for line in lines if REPORT.match( line ) and "strategic accuracy" in line]
    steps = [line for line in lines if line.startswith( "  step " )]
    errors = [line for line in lines if "Error" in line or "Traceback" in line]
    out = []
    for line in reports[-last:]:
        out.append( f"  {REPORT.match( line ).group( 1 ):18s} | {kinds( line )} | battle imitation "
                    f"{grab( r'battle imitation ([0-9.]+)', line )} | gate {grab( r'gate ([0-9/]+ of \d+ \(wins [^)]*\))', line )}" )
        for label, pattern in ( ( "game GRPO", r"(game GRPO: [^,]*, [^,(]*(?:\([^)]*\))?)" ),
                                ( "short-horizon", r"(short-horizon: [^,]*, advantage of the pick [^,]*, deviates in [0-9]+%)" ),
                                ( "value", r'value \{"states": \d+, ("mse": [0-9.]+, "mse_of_the_mean": [0-9.]+, "corr": [0-9.]+)' ),
                                ( "games", r"(games vs the built-in AI.*)$" ) ):
            value = grab( pattern, line, "" )
            if value:
                out.append( f"      {label}: {value[:400]}" )
    if steps:
        out.append( f"  last step: {steps[-1].strip()[:300]}" )
    for line in errors[-3:]:
        out.append( f"  ERROR: {line[:300]}" )
    return out


def collector_lines( log_glob: str, data_glob: str ) -> list[str]:
    out = []
    total = 0
    for path in sorted( glob.glob( data_glob ) ):
        with open( path, "rb" ) as f:
            groups = sum( 1 for line in f if line.strip() )
        total += groups
        age = max( time.time() - os.path.getctime( path ), 1.0 )
        out.append( f"  {os.path.basename( path )}: {groups} groups ({groups / age * 3600:.1f}/h)" )
    out.append( f"  total: {total} groups" )
    for path in sorted( glob.glob( log_glob ) ):
        with open( path, errors="replace" ) as f:
            lines = [line.strip() for line in f if line.strip() and "warn" not in line.lower()]
        if lines:
            out.append( f"  {os.path.basename( path )}: {lines[-1][:200]}" )
    return out


def memory_lines( path: str ) -> list[str]:
    with open( path ) as f:
        lines = [line.split() for line in f if line.strip()]
    if not lines:
        return []
    peak = max( lines, key=lambda row: int( row[row.index( "train_rss_mb" ) + 1] ) if "train_rss_mb" in row else 0 )
    return [f"  last: {' '.join( lines[-1] )}", f"  peak training process: {' '.join( peak )}"]


def main() -> None:
    parser = argparse.ArgumentParser( description="Status of a joint training run and its collectors" )
    parser.add_argument( "--train", required=True, help="train_strategy_net log" )
    parser.add_argument( "--reports", type=int, default=6, help="last reports shown" )
    parser.add_argument( "--collect", default="", help="glob of the collector logs" )
    parser.add_argument( "--data", default="", help="glob of the collected files (one group per line)" )
    parser.add_argument( "--mem", default="logs/memwatch.log" )
    args = parser.parse_args()

    print( time.strftime( "%Y-%m-%d %H:%M:%S %Z" ) )
    sessions = subprocess.run( ["tmux", "ls"], capture_output=True, text=True ).stdout.strip()
    print( "tmux:", ", ".join( line.split( ":" )[0] for line in sessions.splitlines() ) or "-" )
    print( f"training ({args.train}):" )
    print( "\n".join( report_lines( args.train, args.reports ) ) if os.path.exists( args.train ) else "  (no log yet)" )
    if args.data or args.collect:
        print( "collectors:" )
        print( "\n".join( collector_lines( args.collect, args.data ) ) )
    if os.path.exists( args.mem ):
        print( "memory:" )
        print( "\n".join( memory_lines( args.mem ) ) )


if __name__ == "__main__":
    main()
