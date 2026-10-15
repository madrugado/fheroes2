"""Group-relative policy improvement for the battle policy (the GRPO idea, user request 2026-10-15): instead of ONE
rollout per move (battle_prefs.py / DPO, where a different move also reshuffles the dice), every candidate move of a
decision is played out K times to the end of the battle with different dice (battle server `restore` with
`"reseed"`), and the policy is pushed towards the moves whose MEAN outcome is above the group's mean.

  collect — walks real battles (harvest_battles.py setups) with the built-in AI; at a share `--rate` of the
            decisions the candidates (the built-in move, the policy's top moves, random moves) are each rolled out
            `--rollouts` times with reseed 1..K (the same K seeds for every candidate: common random numbers),
            score = battle_prefs.move_score (outcome + own strength left + enemy strength killed - 1).
  train   — loss per decision = -sum_i A_i log pi(option i) + kl x mean_i (log pi(i) - log ref(i))^2 over the options,
            A_i = (mean_i - mean over options) / (std over options + 0.05); decisions whose options all score alike
            are skipped. The squared log-ratio keeps the policy near the starting network (a plain advantage
            loss pushes the bad options to log pi -> -inf). Validation: the mean score of the network's preferred
            option among the scored ones vs the built-in move's.

  rl/.venv/bin/python rl/battle_grpo.py collect --setups rl/data/v4/battles_Arena.jsonl \\
      --model rl/models/az_battle_expert_v4.pt --out rl/data/v4/grpo_Arena.jsonl
  rl/.venv/bin/python rl/battle_grpo.py train --data rl/data/v4/grpo_*.jsonl --model rl/models/az_battle_expert_v4.pt \\
      --out rl/models/az_battle_grpo_v4.pt
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import statistics
import sys
import time

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

from battle_prefs import MAX_STEPS, SNAP_ID, move_score, mover_side, side_strength  # noqa: E402
from engine_bridge import BattleEnv, new_battle_from_setup  # noqa: E402
from gen_expert import STATE_KEYS, load_setups  # noqa: E402


def candidates( state: dict, expert: dict | None, policy_value, top: int, randoms: int, rng: random.Random ) -> list[int]:
    """Indexes into state["legal"]: the built-in move, the policy's top moves, random moves (no duplicates)."""
    legal = state["legal"]
    chosen: list[int] = []
    if expert is not None and expert in legal:
        chosen.append( legal.index( expert ) )
    if policy_value is not None:
        priors, _ = policy_value.evaluate( state )
        for i in sorted( priors, key=priors.get, reverse=True )[:top]:
            if i not in chosen:
                chosen.append( i )
    for _ in range( randoms ):
        extra = rng.randrange( len( legal ) )
        if extra not in chosen:
            chosen.append( extra )
    return chosen


def score_parts( root: dict, final: dict | None, side: str ) -> list[float] | None:
    """The parts of move_score: [outcome, own strength left (share of root), enemy strength left (share of root)]."""
    if final is None or final.get( "ev" ) != "state":
        return None
    enemy = "def" if side == "att" else "att"
    result = final.get( "result" )
    own0, enemy0 = side_strength( root, side ), side_strength( root, enemy )
    return [1.0 if result == side else ( -1.0 if result == enemy else 0.0 ),
            round( side_strength( final, side ) / own0, 4 ) if own0 else 0.0,
            round( side_strength( final, enemy ) / enemy0, 4 ) if enemy0 else 0.0]


def collect_battle( env: BattleEnv, setup: dict, policy_value, args, rng: random.Random, key: str ) -> list[dict]:
    records: list[dict] = []
    state = new_battle_from_setup( env, setup )
    for _ in range( MAX_STEPS ):
        if state is None or state.get( "ev" ) != "state" or state.get( "result" ) or not state.get( "legal" ):
            break
        expert = ( env.suggest() or {} ).get( "expert" )
        if len( state["legal"] ) > 1 and rng.random() < args.rate:
            env.snapshot_save( SNAP_ID )
            side = mover_side( state )
            options = candidates( state, expert, policy_value, args.top, args.randoms, rng )
            scores, parts = [], []
            for i in options:
                move = state["legal"][i]
                row, row_parts = [], []
                for k in range( 1, args.rollouts + 1 ):
                    final = env.snapshot_restore( SNAP_ID, path=[( move["act"], move["args"] )], rollout=True, reseed=k )
                    row.append( round( move_score( state, final, side ), 4 ) )
                    row_parts.append( score_parts( state, final, side ) )
                scores.append( row )
                parts.append( row_parts )
            records.append( {
                "state": {k: state[k] for k in STATE_KEYS if k in state},
                "legal": state["legal"],
                "options": options,
                "scores": scores,
                "parts": parts,  # per rollout [outcome, own left, enemy left] (value targets)
                "expert": state["legal"].index( expert ) if expert in state["legal"] else None,
                "side": side,
                "battle": key,
            } )
        if expert is None:
            break
        state = env.action( expert["act"], expert["args"] )
    env.snapshots_free()
    return records


def collect( args ) -> None:
    policy_value = None
    if args.model:
        import torch

        torch.set_num_threads( 2 )
        from selfplay import load_policy_value

        policy_value = load_policy_value( args.model, args.arch, "cpu" )

    rng = random.Random( args.seed )
    setups = load_setups( args.setups )
    if args.limit:
        setups = setups[: args.limit]
    envs: dict[str, BattleEnv] = {}
    total, t0 = 0, time.time()
    os.makedirs( os.path.dirname( args.out ) or ".", exist_ok=True )
    with open( args.out, "w" ) as out:
        try:
            for n, setup in enumerate( setups ):
                map_name = setup.get( "map", "Arena.mp2" )
                if map_name not in envs:
                    envs[map_name] = BattleEnv( map_name=map_name )
                key = f"real:{map_name}:{setup.get( 'game_seed' )}:{setup['bid']}"
                try:
                    records = collect_battle( envs[map_name], setup, policy_value, args, rng, key )
                except TimeoutError:
                    print( f"battle {key}: timeout, respawning the engine", flush=True )
                    envs.pop( map_name ).close()
                    continue
                for record in records:
                    out.write( json.dumps( record, separators=( ",", ":" ) ) + "\n" )
                total += len( records )
                if ( n + 1 ) % 50 == 0 or n == len( setups ) - 1:
                    print( f"battle {n + 1}/{len( setups )}: {total} decisions, {time.time() - t0:.0f}s", flush=True )
        finally:
            for env in envs.values():
                env.close()
    print( f"done: {total} decisions -> {args.out}" )


def group_advantages( record: dict, eps: float = 0.05 ) -> list[float] | None:
    means = [statistics.fmean( row ) for row in record["scores"]]
    if len( means ) < 2:
        return None
    spread = statistics.pstdev( means )
    if spread < 1e-6:
        return None
    center = statistics.fmean( means )
    return [( m - center ) / ( spread + eps ) for m in means]


def train( args ) -> None:
    import torch

    from train import split_records
    from train_dpo import resnet_move_logps, transformer_move_logps

    torch.set_num_threads( args.threads )
    sys.stdout.reconfigure( line_buffering=True )
    device = torch.device( "cpu" )

    records = []
    for path in args.data:
        with open( path ) as f:
            records.extend( json.loads( line ) for line in f if line.strip() )
    records = [r for r in records if group_advantages( r ) is not None]
    train_records, val_records = split_records( records, args.val )
    print( f"decisions with a spread: {len( train_records )} train, {len( val_records )} validation" )

    if args.arch == "transformer":
        from transformer_model import load_checkpoint, save_checkpoint

        model = load_checkpoint( args.model, "cpu" )
        move_logps = transformer_move_logps
    else:
        from model import AzBattleNet
        from transformer_model import grow_rows

        model = AzBattleNet()
        model.load_state_dict( grow_rows( model.state_dict(), torch.load( args.model, map_location="cpu" ) ) )
        move_logps = resnet_move_logps
    ref = copy.deepcopy( model ).eval()
    for param in ref.parameters():
        param.requires_grad_( False )

    def expand( batch: list[dict] ) -> tuple[list[dict], list[float], list[int]]:
        """One pseudo-record per (decision, option): its advantage and the decision it belongs to."""
        rows, advantages, owners = [], [], []
        for owner, record in enumerate( batch ):
            for option, advantage in zip( record["options"], group_advantages( record ) ):
                rows.append( {"state": record["state"], "legal": record["legal"], "opt": option} )
                advantages.append( advantage )
                owners.append( owner )
        return rows, advantages, owners

    def evaluate( net ) -> dict:
        """Mean score of the net's preferred scored option vs the built-in move's (when it was scored)."""
        net.eval()
        picked, builtin = [], []
        with torch.no_grad():
            for start in range( 0, len( val_records ), args.batch ):
                batch = val_records[start:start + args.batch]
                rows, _, owners = expand( batch )
                logps = move_logps( net, rows, "opt", device ).tolist()
                best: dict[int, tuple[float, int]] = {}
                for row_index, owner in enumerate( owners ):
                    if owner not in best or logps[row_index] > best[owner][0]:
                        best[owner] = ( logps[row_index], row_index )
                for owner, record in enumerate( batch ):
                    means = [statistics.fmean( r ) for r in record["scores"]]
                    first = owners.index( owner )
                    picked.append( means[best[owner][1] - first] )
                    if record["expert"] in record["options"]:
                        builtin.append( means[record["options"].index( record["expert"] )] )
        return {"policy": statistics.fmean( picked ) if picked else 0.0, "builtin": statistics.fmean( builtin ) if builtin else 0.0}

    before = evaluate( model )
    print( f"validation before: mean score of the net's pick {before['policy']:+.4f} (built-in move {before['builtin']:+.4f})" )
    optimizer = torch.optim.AdamW( [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0 )
    for epoch in range( args.epochs ):
        model.train()
        random.shuffle( train_records )
        total, batches = 0.0, 0
        for start in range( 0, len( train_records ), args.batch ):
            batch = train_records[start:start + args.batch]
            rows, advantages, _ = expand( batch )
            logp = move_logps( model, rows, "opt", device )
            with torch.no_grad():
                ref_logp = move_logps( ref, rows, "opt", device )
            adv = torch.tensor( advantages, dtype=torch.float32 )
            loss = -( adv * logp ).mean() + args.kl * ( ( logp - ref_logp ) ** 2 ).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_( model.parameters(), 1.0 )
            optimizer.step()
            total += loss.item()
            batches += 1
        after = evaluate( model )
        print( f"epoch {epoch + 1}: loss {total / max( batches, 1 ):+.4f}, validation pick {after['policy']:+.4f} "
               f"(built-in {after['builtin']:+.4f})" )

    os.makedirs( os.path.dirname( args.out ) or ".", exist_ok=True )
    if args.arch == "transformer":
        save_checkpoint( model, args.out )
    else:
        torch.save( model.state_dict(), args.out )
    print( f"model saved -> {args.out}" )


def main() -> None:
    parser = argparse.ArgumentParser( description="Group-relative (GRPO-style) battle policy improvement from repeated rollouts" )
    sub = parser.add_subparsers( dest="cmd", required=True )
    c = sub.add_parser( "collect" )
    c.add_argument( "--setups", nargs="+", required=True )
    c.add_argument( "--model", default=None, help="policy whose top moves are candidates" )
    c.add_argument( "--arch", choices=["resnet", "transformer"], default="resnet" )
    c.add_argument( "--top", type=int, default=3 )
    c.add_argument( "--randoms", type=int, default=1 )
    c.add_argument( "--rollouts", type=int, default=8, help="rollouts per candidate (reseed 1..K)" )
    c.add_argument( "--rate", type=float, default=0.2, help="share of decisions labeled" )
    c.add_argument( "--limit", type=int, default=0 )
    c.add_argument( "--seed", type=int, default=7 )
    c.add_argument( "--out", required=True )
    t = sub.add_parser( "train" )
    t.add_argument( "--data", nargs="+", required=True )
    t.add_argument( "--model", required=True, help="starting checkpoint (also the reference)" )
    t.add_argument( "--arch", choices=["resnet", "transformer"], default="resnet" )
    t.add_argument( "--kl", type=float, default=0.5, help="weight of the squared log-ratio to the reference" )
    t.add_argument( "--epochs", type=int, default=3 )
    t.add_argument( "--batch", type=int, default=64 )
    t.add_argument( "--lr", type=float, default=1e-5 )
    t.add_argument( "--val", type=float, default=0.1 )
    t.add_argument( "--threads", type=int, default=2 )
    t.add_argument( "--out", required=True )
    args = parser.parse_args()
    collect( args ) if args.cmd == "collect" else train( args )


if __name__ == "__main__":
    main()
