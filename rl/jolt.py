"""JOLT — Joint On-Policy Learning and Teaching (Ardywibowo, Dalal, Jiao 2026, arXiv 2610.10447; user request
2026-10-26): ONE network in two roles. The privileged TEACHER is the network given hints the STUDENT does not have; the
teacher learns from outcome rewards with a KL leash towards the student (a teacher that "meets the student where they
are"), the student learns by on-policy distillation from the teacher along its own decisions. JOLT+ (the user's
choice) also gives the student its own outcome advantage. The student is the player: it is what plays real games.

Hints (user design):
  battle   — the battle ResNet's move distribution (the grpo3 policy): per first-step token and per (attack cell,
             direction) (AzBattleTransformer.forward_batch `hints`, HINT_W; zero-initialised inputs, so an untrained
             teacher equals the student exactly);
  strategy — the built-in AI's answer (strategy_net views): the student has NO access to the built-in AI (no target
             values / ranks / value order, no hire `bi` flag, no "let the built-in AI build" option), the teacher sees
             its answer flagged (build: the engine's probe `bi`, AI::setBuildProbe).

Per decision (discrete options, so the KLs are exact instead of the paper's sampled log-ratios):
  teacher: -A_k log pi_T(a_k | s, c) + beta_T KL(pi_T(.|s,c) || sg pi_S(.|s))   (a_k sampled by the teacher)
  student:  beta_S KL(pi_S(.|s) || sg pi_T(.|s,c))  [- A_k log pi_S(a_k | s)  JOLT+, a_k sampled by the student]
with A the reward centred within the role's group (paper eq. 19; strategy: divided by the group's spread + eps like the
whole-game GRPO). Both roles update the shared parameters. Paper defaults: beta_T 0.001-0.03, beta_S 0.1.

  battle data   — `collect` below: real battle setups walked by the STUDENT on one side (built-in AI on the other); at a
                  share of its decisions G teacher samples (with the ResNet hint) and G student samples, each rolled out
                  K times to the end with the built-in AI (battle server `restore` rollout + reseed, as battle_grpo.py).
  strategy data — strategy_game_grpo.py collect --jolt: per seed --group teacher games and --group student games.
  training      — train_strategy_net.py --jolt-battles FILES --jolt-games FILES.

  rl/.venv/bin/python rl/jolt.py collect --setups rl/data/v4/battles_Arena.jsonl --model rl/models/<joint>.pt \\
      --hint-model rl/models/az_battle_grpo3_v4b.pt --out rl/data/jolt_battles_Arena.jsonl
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

import encoding as enc  # noqa: E402


def _torch():
    import torch

    return torch


# --- shared ---------------------------------------------------------------------------------------


def kl_per_group( logp, logq, owner, groups: int ):
    """KL(p || q) per group of rows (owner: the group of every row), p = exp(logp); rows are one distribution's support."""
    torch = _torch()
    terms = logp.exp() * ( logp - logq )
    return torch.zeros( groups, dtype=terms.dtype, device=terms.device ).index_add( 0, owner, terms )


def centred( rewards: list[float], eps: float | None = None ) -> list[float]:
    """Rewards centred over the group (paper eq. 19); with `eps` also divided by (spread + eps)."""
    if not rewards:
        return []
    mean = statistics.fmean( rewards )
    scale = 1.0 if eps is None else statistics.pstdev( rewards ) + eps
    if scale <= 0:
        return [0.0] * len( rewards )  # every game of the role scored the same: no signal
    return [( r - mean ) / scale for r in rewards]


# --- battle ---------------------------------------------------------------------------------------


def battle_hint( state: dict, legal: list, probs: list[float] ):
    """The hint of a battle decision from another player's probabilities over `legal`: (probability per first-step
    token, {attack cell: probability per direction given the cell}) in the transformer's decomposition."""
    import transformer_model as tfm

    cells = enc.unit_cells_map( state["units"] )
    tokens = [0.0] * tfm.NUM_POLICY_TOKENS
    dirs: dict[int, list[float]] = {}
    for move, p in zip( legal, probs ):
        act, args = ( move["act"], move["args"] ) if isinstance( move, dict ) else ( move[0], list( move[1] ) )
        parts = tfm.decompose_action( act, list( args ), cells )
        if parts is None or p <= 0:
            continue
        kind, cell, direction = parts
        token = tfm.SKIP_CELL if kind == "skip" else cell
        tokens[token] += p
        if kind == "attack":
            dirs.setdefault( cell, [0.0] * tfm.NUM_DIRECTIONS )[direction] += p
    total = sum( tokens )
    if total > 0:
        tokens = [t / total for t in tokens]
    for cell, row in dirs.items():
        mass = sum( row )
        dirs[cell] = [d / mass for d in row]
    return tokens, dirs


def battle_move_logps( model, records: list[dict], device, hinted: bool ):
    """log pi over every decomposable legal move of every record (one row per move): (logp rows, owner, legal index
    per row). hinted: the teacher (record["hint"], the hint net's probabilities over record["legal"])."""
    from train_strategy_net import battle_group_forward

    samples = [( r["state"], r["legal"], [0.0] * len( r["legal"] ), 0.0 ) for r in records]
    hints = [battle_hint( r["state"], r["legal"], r["hint"] ) for r in records] if hinted else None
    logq, owner, _, _, moves = battle_group_forward( model, samples, device, hints=hints, with_moves=True )
    return logq, owner, moves


def _scores( record: dict ) -> dict[int, float]:
    return {int( move ): statistics.fmean( rows ) for move, rows in record["scores"].items()}


def _sampled_term( logp, owner, moves, records: list[dict], key: str, device ):
    """-mean over the records of mean_k A_k log pi(a_k), a_k = record[key] (sampled moves), A centred over them."""
    torch = _torch()
    row_of = {( int( o ), m ): i for i, ( o, m ) in enumerate( zip( owner.tolist(), moves ) )}
    index, weights = [], []
    for n, record in enumerate( records ):
        sampled = record.get( key ) or []
        if len( sampled ) < 2:
            continue
        scores = _scores( record )
        advantages = centred( [scores[m] for m in sampled] )
        for move, advantage in zip( sampled, advantages ):
            if ( n, move ) in row_of and advantage != 0.0:
                index.append( row_of[( n, move )] )
                weights.append( advantage / len( sampled ) )
    if not index:
        return torch.zeros( (), device=device )
    picked = logp[torch.tensor( index, device=device )]
    return -( torch.tensor( weights, dtype=picked.dtype, device=device ) * picked ).sum() / len( records )


def battle_loss( model, records: list[dict], device, beta_t: float, beta_s: float, plus: bool = True ):
    """The JOLT loss of a batch of battle decisions (mean per decision; see the module doc)."""
    student, owner, moves = battle_move_logps( model, records, device, hinted=False )
    teacher, owner_t, moves_t = battle_move_logps( model, records, device, hinted=True )
    assert moves == moves_t and bool( ( owner == owner_t ).all() ), "teacher and student rows must line up"
    count = len( records )
    kl_s = kl_per_group( student, teacher.detach(), owner, count ).mean()
    kl_t = kl_per_group( teacher, student.detach(), owner, count ).mean()
    loss = _sampled_term( teacher, owner, moves, records, "teacher", device ) + beta_t * kl_t + beta_s * kl_s
    if plus:
        loss = loss + _sampled_term( student, owner, moves, records, "student", device )
    return loss


def battle_metrics( model, records: list[dict], device, chunk: int = 16 ) -> dict:
    """Held-out battle decisions: mean KL(student || teacher); for the student, the teacher and the hint net the
    expected rollout score of its distribution restricted to the scored moves; the hint scale."""
    torch = _torch()
    model.eval()
    kl, picks, count = 0.0, {"student": 0.0, "teacher": 0.0, "hint": 0.0}, 0
    for start in range( 0, len( records ), chunk ):
        part = records[start:start + chunk]
        with torch.no_grad():
            student, owner, moves = battle_move_logps( model, part, device, hinted=False )
            teacher, _, _ = battle_move_logps( model, part, device, hinted=True )
        kl += float( kl_per_group( student.float(), teacher.float(), owner, len( part ) ).sum() )
        owner_list = owner.tolist()
        for n, record in enumerate( part ):
            scores = _scores( record )
            rows = [i for i, o in enumerate( owner_list ) if o == n and moves[i] in scores]
            dists = {"student": {moves[i]: float( student[i].exp() ) for i in rows},
                     "teacher": {moves[i]: float( teacher[i].exp() ) for i in rows},
                     "hint": {m: record["hint"][m] for m in scores}}
            for name, dist in dists.items():
                mass = sum( dist.values() )
                picks[name] += sum( p * scores[m] for m, p in dist.items() ) / mass if mass > 0 else statistics.fmean( scores.values() )
            count += 1
    count = max( count, 1 )
    return {"kl": round( kl / count, 4 ), **{f"{k}_score": round( v / count, 4 ) for k, v in picks.items()},
            "hint_scale": round( float( model.hint_scale.detach() ), 4 )}


# --- strategy -------------------------------------------------------------------------------------


def strategic_log_probs( model, records: list[dict], obj_vocab: list[int], view: str ):
    """(B, max options) log-softmax over each query's canonical options as `view` shows them."""
    from strategy_net import MAX_STRATEGIC_TOKENS, canonical_logits, option_order, query_tokens
    from transformer_model import strategic_logits


    torch = _torch()
    window = model.config.get( "window", MAX_STRATEGIC_TOKENS )
    queries = [query_tokens( r["kind"], r["event"], r.get( "context" ), obj_vocab, r.get( "history" ), window, view ) for r in records]
    logits, _ = strategic_logits( model, queries )
    logits = canonical_logits( logits, [option_order( r["kind"], r["event"], view ) for r in records] )
    return torch.log_softmax( logits.float(), dim=1 )


def game_records( group: dict, eps: float ) -> list[dict]:
    """Training records of a JOLT group (strategy_game_grpo collect --jolt): every answer of every game with its role,
    the history it was given in, and the advantage of its game within the role ((R - mean) / (spread + eps))."""
    records = []
    for role in ( "teacher", "student" ):
        games = [g for g in group["games"] if g.get( "role" ) == role]
        if len( games ) < 2:
            continue
        advantages = centred( [g["reward"] for g in games], eps )
        for game, advantage in zip( games, advantages ):
            days, previous = game["days"], []
            for answer in game["answers"]:
                ctx = answer["ctx"]
                context = days[ctx] if ctx >= 0 else None
                records.append( {"kind": answer["kind"], "event": answer["event"], "context": context,
                                 "history": {"days": days[:max( ctx, 0 )], "decisions": list( previous )},
                                 "seed": group["seed"], "map": group["map"], "role": role, "answer": answer["answer"],
                                 "adv": advantage} )
                previous.append( {"kind": answer["kind"], "event": answer["event"], "context": context, "answer": answer["answer"]} )
    return records


def strategic_loss( model, records: list[dict], obj_vocab: list[int], beta_t: float, beta_s: float, plus: bool = True ):
    """The JOLT loss of a batch of strategic answers (mean per answer): a teacher answer trains the teacher (its
    advantage + KL to the student), a student answer trains the student (KL to the teacher + its advantage, JOLT+)."""
    torch = _torch()
    student = strategic_log_probs( model, records, obj_vocab, "student" )
    teacher = strategic_log_probs( model, records, obj_vocab, "teacher" )
    real = student > -1e8  # padding options

    def kl( p_log, q_log ):
        return torch.where( real, p_log.exp() * ( p_log - q_log ), torch.zeros_like( p_log ) ).sum( dim=1 )

    is_teacher = torch.tensor( [r["role"] == "teacher" for r in records], device=student.device )
    answers = torch.tensor( [r["answer"] for r in records], device=student.device ).unsqueeze( 1 )
    advantages = torch.tensor( [r["adv"] for r in records], dtype=student.dtype, device=student.device )
    teacher_terms = -advantages * teacher.gather( 1, answers ).squeeze( 1 ) + beta_t * kl( teacher, student.detach() )
    student_terms = beta_s * kl( student, teacher.detach() )
    if plus:
        student_terms = student_terms - advantages * student.gather( 1, answers ).squeeze( 1 )
    return torch.where( is_teacher, teacher_terms, student_terms ).mean()


def strategic_metrics( model, records: list[dict], obj_vocab: list[int], batch: int = 16 ) -> dict:
    """Held-out answers: mean KL(student || teacher) and how often the student's / the teacher's most probable option
    is the built-in AI's answer (the teacher has it as a hint; the student has to learn it or find better ones)."""
    from strategy_net import builtin_option, query_options

    torch = _torch()
    model.eval()
    kl, student_builtin, teacher_builtin, agree, known = 0.0, 0, 0, 0, 0
    for start in range( 0, len( records ), batch ):
        part = records[start:start + batch]
        with torch.no_grad():
            student = strategic_log_probs( model, part, obj_vocab, "student" )
            teacher = strategic_log_probs( model, part, obj_vocab, "teacher" )
        real = student > -1e8
        kl += float( torch.where( real, student.exp() * ( student - teacher ), torch.zeros_like( student ) ).sum() )
        s_pick, t_pick = student.argmax( dim=1 ).tolist(), teacher.argmax( dim=1 ).tolist()
        for record, s, t in zip( part, s_pick, t_pick ):
            agree += s == t
            builtin = builtin_option( record["kind"], query_options( record["kind"], record["event"], "student" ), record["event"] )
            if builtin is not None:
                known += 1
                student_builtin += s == builtin
                teacher_builtin += t == builtin
    count = max( len( records ), 1 )
    return {"kl": round( kl / count, 4 ), "agree": round( agree / count, 3 ), "student_builtin": round( student_builtin / max( known, 1 ), 3 ),
            "teacher_builtin": round( teacher_builtin / max( known, 1 ), 3 )}


# --- battle collector -----------------------------------------------------------------------------


def _distribution( model, record: dict, device, hinted: bool ) -> list[float]:
    """The network's probabilities over record["legal"] (0 for moves it cannot decompose)."""
    torch = _torch()
    from precision import autocast_for

    with torch.no_grad(), autocast_for( model ):  # bf16 collectors (--bf16)
        logp, _, moves = battle_move_logps( model, [record], device, hinted )
    probs = [0.0] * len( record["legal"] )
    for value, move in zip( logp.float().exp().tolist(), moves ):
        probs[move] = value
    return probs


def _sample( probs: list[float], count: int, temperature: float, rng: random.Random ) -> list[int]:
    weights = [p ** ( 1.0 / temperature ) if p > 0 else 0.0 for p in probs]
    return rng.choices( range( len( probs ) ), weights=weights, k=count )


def collect_battle( env, setup: dict, model, hint_net, args, rng: random.Random, key: str ) -> list[dict]:
    from battle_prefs import MAX_STEPS, SNAP_ID, move_score, mover_side
    from engine_bridge import new_battle_from_setup
    from gen_expert import STATE_KEYS
    from transformer_model import battle_history_entry

    records: list[dict] = []
    history: list[dict] = []
    ours = rng.choice( ( "att", "def" ) )  # the student's side; the built-in AI plays the other one
    state = new_battle_from_setup( env, setup )
    for _ in range( MAX_STEPS ):
        if state is None or state.get( "ev" ) != "state" or state.get( "result" ) or not state.get( "legal" ):
            break
        legal = state["legal"]
        kept = {k: state[k] for k in STATE_KEYS if k in state}
        kept["history"] = list( history )
        if mover_side( state ) == ours:
            record = {"state": kept, "legal": legal}
            student = _distribution( model, record, "cpu", hinted=False )
            if sum( student ) <= 0:
                move = ( env.suggest() or {} ).get( "expert" )  # nothing the network can play: the built-in move
            elif len( legal ) > 1 and rng.random() < args.rate:
                hint_state = dict( state )
                priors, _ = hint_net.evaluate( hint_state )
                record["hint"] = [round( priors.get( i, 0.0 ), 6 ) for i in range( len( legal ) )]
                teacher = _distribution( model, record, "cpu", hinted=True )
                record["teacher"] = _sample( teacher, args.group, args.temperature, rng )
                record["student"] = _sample( student, args.group, args.temperature, rng )
                side = mover_side( state )
                env.snapshot_save( SNAP_ID )
                scores = {}
                for i in sorted( set( record["teacher"] ) | set( record["student"] ) ):
                    path = [( legal[i]["act"], legal[i]["args"] )]
                    scores[str( i )] = [round( move_score( state, env.snapshot_restore( SNAP_ID, path=path, rollout=True, reseed=k ), side ), 4 )
                                        for k in range( 1, args.rollouts + 1 )]
                record.update( {"scores": scores, "side": side, "battle": key} )
                records.append( record )
                move = legal[record["student"][0]]  # the student's own decision continues the battle
            else:
                move = legal[_sample( student, 1, args.temperature, rng )[0]]
        else:
            move = ( env.suggest() or {} ).get( "expert" )
        if move is None:
            break
        entry = battle_history_entry( state, move["act"], list( move["args"] ) )
        if entry is not None:
            history.append( entry )
        state = env.action( move["act"], move["args"] )
    env.snapshots_free()
    return records


def collect( args ) -> None:
    torch = _torch()
    torch.set_num_threads( args.threads )
    from engine_bridge import BattleEnv
    from gen_expert import load_setups
    from precision import for_inference
    from selfplay import load_policy_value
    from transformer_model import load_checkpoint

    rng = random.Random( args.seed )
    hint_net = load_policy_value( args.hint_model, "resnet", "cpu" )
    setups = load_setups( args.setups )
    if args.limit:
        setups = setups[:args.limit]
    envs: dict[str, BattleEnv] = {}
    model, loaded, total, started = None, None, 0, time.time()
    os.makedirs( os.path.dirname( args.out ) or ".", exist_ok=True )
    with open( args.out, "a" ) as out:
        try:
            for n, setup in enumerate( setups ):
                stamp = os.path.getmtime( args.model )
                if stamp != loaded:  # a training run saved a newer model: stay on-policy
                    model, loaded = for_inference( load_checkpoint( args.model ), "cpu", args.bf16 ), stamp
                map_name = setup.get( "map", "Arena.mp2" )
                if map_name not in envs:
                    envs[map_name] = BattleEnv( map_name=map_name )
                key = f"real:{map_name}:{setup.get( 'game_seed' )}:{setup['bid']}"
                try:
                    records = collect_battle( envs[map_name], setup, model, hint_net, args, rng, key )
                except TimeoutError:
                    print( f"battle {key}: timeout, respawning the engine", flush=True )
                    envs.pop( map_name ).close()
                    continue
                for record in records:
                    record["model_time"] = loaded
                    out.write( json.dumps( record, separators=( ",", ":" ) ) + "\n" )
                out.flush()
                total += len( records )
                if ( n + 1 ) % 20 == 0 or n == len( setups ) - 1:
                    print( f"battle {n + 1}/{len( setups )}: {total} decisions, {time.time() - started:.0f}s", flush=True )
        finally:
            for env in envs.values():
                env.close()
    print( f"done: {total} decisions -> {args.out}" )


def main() -> None:
    parser = argparse.ArgumentParser( description="JOLT data for the battle policy" )
    sub = parser.add_subparsers( dest="cmd", required=True )
    c = sub.add_parser( "collect", help="battle decisions of the student with teacher and student samples rolled out" )
    c.add_argument( "--setups", nargs="+", required=True )
    c.add_argument( "--model", required=True, help="the joint transformer (student and teacher; re-read when the file changes)" )
    c.add_argument( "--hint-model", default="rl/models/az_battle_grpo3_v4b.pt", help="battle ResNet whose moves are the teacher's hint" )
    c.add_argument( "--group", type=int, default=4, help="sampled moves per role and decision (paper: 4 teacher + 4 student)" )
    c.add_argument( "--rollouts", type=int, default=4, help="rollouts per sampled move (reseed 1..K, common random numbers)" )
    c.add_argument( "--rate", type=float, default=0.2, help="share of the student's decisions labeled" )
    c.add_argument( "--temperature", type=float, default=1.0 )
    c.add_argument( "--limit", type=int, default=0 )
    c.add_argument( "--seed", type=int, default=7 )
    c.add_argument( "--threads", type=int, default=1 )
    c.add_argument( "--bf16", action="store_true" )
    c.add_argument( "--out", required=True )
    args = parser.parse_args()
    collect( args )


if __name__ == "__main__":
    main()
