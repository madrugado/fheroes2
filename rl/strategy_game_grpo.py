"""Whole-game GRPO for the strategic answers (user decision 2026-10-20: a single answer barely moves the end of a
game and the value cannot tell answers apart a few days later, so the reward is the END of the game and the group is a
set of games of the same seed).

  collect — per seed one color is ours; a group of `--group` games of that seed (every player on `--plan`, per-turn
            random streams so an answer changes the dice only until the next turn): game 0 answers greedily with the
            strategic net, games 1.. answer greedily too but leave the net's answer with probability `--eps` for a
            random other option (an exploration answer, recorded with the history the net saw). Reward of a game =
            minus the hero equivalent of our strongest hero at the end (strategy_games.equivalent_of) against ONE
            reference per group — the strongest rival hero of game 0 — so every game of a group is measured on the
            same scale. The model file is re-read when it changes (on-policy with a training run that saves it).
  The training side is `train_strategy_net --game-grpo FILES`: per group A_k = (R_k - mean) / (std + eps), and every
  exploration answer of game k is pushed by A_k (-A log pi(answer)), with a KL leash.

  rl/.venv/bin/python rl/strategy_game_grpo.py collect --model rl/models/unified_200m_joint12.pt --seeds 6001-6500 \\
      --out rl/data/game_grpo_c1.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

from harvest_battles import parse_seeds  # noqa: E402
from strategy_games import PolicyBranch, equivalent_of, play, strongest_rival_hero  # noqa: E402
from strategy_net import answer_of, query_options  # noqa: E402

COLORS = ( "Blue", "Green", "Red", "Yellow", "Orange", "Purple" )


class ExploreBranch( PolicyBranch ):
    """PolicyBranch whose policy leaves its own answer with probability `eps` for a random other option; such
    exploration answers are kept with the history the policy saw (the records the GRPO update needs)."""

    def __init__( self, policy, color: str, eps: float, rng: random.Random ):
        super().__init__( policy, color )
        self.eps = eps
        self.rng = rng
        self.explored: list[dict] = []
        self.answered = 0

    def _answer( self, kind: str, event: dict ):
        if event.get( "p" ) != self.color:
            return super()._answer( kind, event )
        self.count += 1
        options = query_options( kind, event )
        history = self.policy.history( self.color )
        snapshot = {"days": list( history["days"] ), "decisions": list( history["decisions"] )}
        context = self.contexts.get( self.color )
        choice = self.policy.decide( kind, event )
        if choice is not None and len( options ) >= 2:
            self.answered += 1
            if self.rng.random() < self.eps:
                own = choice
                choice = self.rng.choice( [i for i in range( len( options ) ) if i != own] )
                self.explored.append( {"kind": kind, "event": event, "context": context, "history": snapshot,
                                       "answer": choice, "own": own} )
            self._remember( kind, event, choice )
        self.policy.record( kind, event, choice )
        answer = None if choice is None else answer_of( options[choice] )
        if kind == "target" and answer is not None and answer is ( event.get( "cands" ) or [None] )[0]:
            answer = None  # the top candidate is the built-in choice
        return answer


def collect( args ) -> None:
    from engine_bridge import BattleEnv
    from strategy_net import NetStrategyPolicy

    rng = random.Random( args.rng_seed )
    colors = args.colors.split( "," )
    duel_env = BattleEnv( map_name=args.map )
    policy, loaded = None, None
    started = time.time()
    try:
        with open( args.out, "a" ) as out:
            for number, seed in enumerate( parse_seeds( args.seeds ) ):
                stamp = os.path.getmtime( args.model )
                if stamp != loaded:  # a training run saved a newer model: stay on-policy
                    policy, loaded = NetStrategyPolicy( args.model ), stamp
                color = colors[number % len( colors )]
                games = []
                reference = None
                for k in range( args.group ):
                    branch = ExploreBranch( policy, color, 0.0 if k == 0 else args.eps, random.Random( rng.random() ) )
                    try:
                        _, results, _, _ = play( args, seed, args.days, branch )
                    except ( TimeoutError, RuntimeError ) as error:
                        print( f"seed {seed} game {k}: failed ({error})", flush=True )
                        continue
                    if k == 0:
                        reference = strongest_rival_hero( results, color )
                    reward = -equivalent_of( duel_env, results, color, reference, seed )
                    games.append( {"k": k, "reward": round( reward, 4 ), "answered": branch.answered, "explored": branch.explored} )
                if len( games ) >= 2:
                    out.write( json.dumps( {"seed": seed, "map": args.map, "color": color, "model_time": loaded, "games": games},
                                           separators=( ",", ":" ) ) + "\n" )
                    out.flush()
                rewards = [g["reward"] for g in games]
                print( f"seed {seed} {color}: rewards {rewards} (sd {statistics.pstdev( rewards ) if rewards else 0:.2f}), "
                       f"explored {sum( len( g['explored'] ) for g in games )}, {time.time() - started:.0f}s", flush=True )
    finally:
        duel_env.close()


def group_records( group: dict, eps: float = 0.3 ) -> list[dict]:
    """The GRPO records of one group: every exploration answer of game k with A_k = (R_k - mean) / (std + eps)
    (`train_strategy_net` pg format: "adv" {option: advantage})."""
    rewards = [g["reward"] for g in group["games"]]
    if len( rewards ) < 2:
        return []
    mean, spread = statistics.fmean( rewards ), statistics.pstdev( rewards )
    records = []
    for game in group["games"]:
        advantage = ( game["reward"] - mean ) / ( spread + eps )
        for answer in game["explored"]:
            records.append( {"kind": answer["kind"], "event": answer["event"], "context": answer["context"],
                             "history": answer["history"], "seed": group["seed"], "map": group["map"],
                             "own": answer["own"], "adv": {answer["answer"]: advantage}} )
    return records


def main() -> None:
    parser = argparse.ArgumentParser( description="Whole-game GRPO data for the strategic answers" )
    sub = parser.add_subparsers( dest="cmd", required=True )
    c = sub.add_parser( "collect" )
    c.add_argument( "--binary", default="./fheroes2" )
    c.add_argument( "--map", default="Battlefi.mp2" )
    c.add_argument( "--days", type=int, default=30 )
    c.add_argument( "--seeds", required=True )
    c.add_argument( "--colors", default=",".join( COLORS ) )
    c.add_argument( "--plan", default="champion=1,split_singles=1,max_heroes=3" )
    c.add_argument( "--model", required=True, help="strategic net (re-read when the file changes)" )
    c.add_argument( "--group", type=int, default=8, help="games per seed" )
    c.add_argument( "--eps", type=float, default=0.1, help="exploration rate of games 1.." )
    c.add_argument( "--rng-seed", type=int, default=1 )
    c.add_argument( "--no-rng-streams", dest="rng_streams", action="store_false" )
    c.add_argument( "--out", required=True )
    args = parser.parse_args()
    collect( args )


if __name__ == "__main__":
    main()
