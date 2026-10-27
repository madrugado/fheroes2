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


def hint_cross_entropy( logp, owner, moves, records: list[dict], device ):
    """Mean over the records of -sum_m h(m) log pi(m): the hint net's probabilities (record["hint"]) renormalised over
    the decomposable moves (the rows), the teacher's anchor to the player whose moves it sees."""
    torch = _torch()
    hint = torch.tensor( [records[o]["hint"][m] for o, m in zip( owner.tolist(), moves )], dtype=torch.float32, device=device )
    mass = torch.zeros( len( records ), device=device ).index_add( 0, owner, hint )
    target = hint / mass.clamp_min( 1e-12 )[owner]
    return -( target * logp.float() ).sum() / len( records )


def battle_loss( model, records: list[dict], device, beta_t: float, beta_s: float, plus: bool = True, anchor: float = 0.0 ):
    """The JOLT loss of a batch of battle decisions (mean per decision; see the module doc). `anchor`: the TEACHER also
    imitates the hint net's move distribution (cross-entropy, weight anchor) — jolt1 (2026-10-26) without it: the teacher
    ignored its hint (hint_scale 0.01 after a day), the battle policy drifted on its own advantages, wins 50% -> 3%."""
    student, owner, moves = battle_move_logps( model, records, device, hinted=False )
    teacher, owner_t, moves_t = battle_move_logps( model, records, device, hinted=True )
    assert moves == moves_t and bool( ( owner == owner_t ).all() ), "teacher and student rows must line up"
    count = len( records )
    kl_s = kl_per_group( student, teacher.detach(), owner, count ).mean()
    kl_t = kl_per_group( teacher, student.detach(), owner, count ).mean()
    loss = _sampled_term( teacher, owner, moves, records, "teacher", device ) + beta_t * kl_t + beta_s * kl_s
    if anchor > 0:
        loss = loss + anchor * hint_cross_entropy( teacher, owner, moves, records, device )
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


def strategic_logits_of( model, records: list[dict], obj_vocab: list[int], view: str, q: bool = False ):
    """(B, max options) logits over each query's canonical options as `view` shows them. With `q`
    also (B, max options) Q(s, option) of the pooled advantage head on the STUDENT view's option
    hidden states (the same values for every view; padding options 0)."""
    from strategy_net import MAX_STRATEGIC_TOKENS, canonical_logits, canonical_rows, option_order, query_tokens
    from transformer_model import strategic_logits, strategic_q_values

    torch = _torch()
    window = model.config.get( "window", MAX_STRATEGIC_TOKENS )
    queries = [query_tokens( r["kind"], r["event"], r.get( "context" ), obj_vocab, r.get( "history" ), window, view ) for r in records]
    if q:
        logits, mask, picked = strategic_logits( model, queries, with_hidden=True )
        q_values = strategic_q_values( model, picked )
    else:
        logits, mask = strategic_logits( model, queries )
        q_values = None
    logits = canonical_logits( logits, [option_order( r["kind"], r["event"], view ) for r in records] )
    if q_values is not None:
        # The Q head reads the student view's states: reorder them by the student's option order (0 on padding).
        q_values = canonical_rows( q_values, [option_order( r["kind"], r["event"], "student" ) for r in records] ) * mask
    return logits, q_values


def strategic_log_probs( model, records: list[dict], obj_vocab: list[int], view: str ):
    """(B, max options) log-softmax over each query's canonical options as `view` shows them."""
    torch = _torch()
    logits, _ = strategic_logits_of( model, records, obj_vocab, view )
    return torch.log_softmax( logits.float(), dim=1 )


# The pooled advantage head's regression target: the game's role-centred reward in hero log2 units
# divided by this (strategy_games.HERO_LOG2_RANGE; the target lies in ~[-1, 1]).
Q_TARGET_SCALE = 6.0


def game_records( group: dict, eps: float ) -> list[dict]:
    """Training records of a JOLT group (strategy_game_grpo collect --jolt): every answer of every game with its role,
    the history it was given in, and the advantage of its game within the role ((R - mean) / (spread + eps)). Every
    record also carries the same centred reward UNDIVIDED ("qtarget", hero log2 units — the pooled advantage head's
    regression target) and the answer's sampling temperature ("T", the collector's per-kind value, 1 for older data).
    An answer drawn by the collector's epsilon exploration ("eps") is off-policy: its advantage is 0 (no policy term),
    its reward still feeds the Q head."""
    records = []
    for role in ( "teacher", "student" ):
        games = [g for g in group["games"] if g.get( "role" ) == role]
        if len( games ) < 2:
            continue
        rewards = [g["reward"] for g in games]
        advantages = centred( rewards, eps )
        qtargets = centred( rewards )
        for game, advantage, qtarget in zip( games, advantages, qtargets ):
            days, previous = game["days"], []
            for answer in game["answers"]:
                ctx = answer["ctx"]
                context = days[ctx] if ctx >= 0 else None
                records.append( {"kind": answer["kind"], "event": answer["event"], "context": context,
                                 "history": {"days": days[:max( ctx, 0 )], "decisions": list( previous )},
                                 "seed": group["seed"], "map": group["map"], "role": role, "answer": answer["answer"],
                                 "adv": 0.0 if answer.get( "eps" ) else advantage, "qtarget": qtarget,
                                 **( {"T": answer["T"]} if "T" in answer else {} ),
                                 **( {"eps": 1} if answer.get( "eps" ) else {} )} )
                previous.append( {"kind": answer["kind"], "event": answer["event"], "context": context, "answer": answer["answer"]} )
    return records


def strategic_loss( model, records: list[dict], obj_vocab: list[int], beta_t: float, beta_s: float, plus: bool = True,
                    anchor: float = 0.0, q_weight: float = 0.0, q_scale: float = Q_TARGET_SCALE,
                    q_advantage: bool = False, q_roles: str = "both", q_norm: float = 1.07 ):
    """The JOLT loss of a batch of strategic answers (mean per answer): a teacher answer trains the teacher (its
    advantage + KL to the student), a student answer trains the student (KL to the teacher + its advantage, JOLT+).
    `anchor`: on every record the TEACHER also imitates the built-in AI's answer (its hint), -anchor log pi_T(built-in).

    Each answer's outcome term trains log softmax(log p / T) with its own sampling temperature ("T", 1 = the plain
    policy — the distribution that sampled it); the KL leashes and the anchor stay on the plain policy (what plays the
    real games). With `q_weight` > 0 the pooled advantage head is also trained on the records' centred rewards
    ("qtarget" / q_scale). With `q_advantage` the outcome terms take A = Q(s, a) - E_pi Q(s, .) from it (detached;
    `q_roles` "both" = the teacher's and the student's terms alike, "student" = only the student's, the teacher keeps
    its game advantage), while the collector's epsilon answers ("eps", off-policy) keep advantage 0. Q is in qtarget /
    q_scale units; the advantage is brought back to the game advantage's units, x q_scale / q_norm (q_norm = a typical
    group spread + eps: 0.77 + 0.3 in jolt2) — without it A_Q was ~100x smaller than the game advantage and the KL
    leashes would have silenced the outcome terms."""
    from strategy_net import builtin_option, query_options

    torch = _torch()
    student_logits, q_values = strategic_logits_of( model, records, obj_vocab, "student", q=q_weight > 0 or q_advantage )
    teacher_logits, _ = strategic_logits_of( model, records, obj_vocab, "teacher" )
    student = torch.log_softmax( student_logits.float(), dim=1 )
    teacher = torch.log_softmax( teacher_logits.float(), dim=1 )
    temperatures = torch.tensor( [[float( r.get( "T", 1.0 ) )] for r in records], dtype=student.dtype, device=student.device )
    student_t = torch.log_softmax( student_logits.float() / temperatures, dim=1 )
    teacher_t = torch.log_softmax( teacher_logits.float() / temperatures, dim=1 )
    real = student > -1e8  # padding options

    def kl( p_log, q_log ):
        return torch.where( real, p_log.exp() * ( p_log - q_log ), torch.zeros_like( p_log ) ).sum( dim=1 )

    is_teacher = torch.tensor( [r["role"] == "teacher" for r in records], device=student.device )
    answers = torch.tensor( [r["answer"] for r in records], device=student.device ).unsqueeze( 1 )
    advantages = torch.tensor( [float( r["adv"] ) for r in records], dtype=student.dtype, device=student.device )
    if q_advantage and q_values is not None:
        def q_advantage_of( log_probs ):
            probs = log_probs.exp().masked_fill( ~real, 0.0 )
            return ( q_values - ( probs * q_values ).sum( dim=1, keepdim=True ) ).detach() * ( q_scale / q_norm )

        from_q = torch.where( is_teacher, q_advantage_of( teacher ).gather( 1, answers ).squeeze( 1 ),
                              q_advantage_of( student ).gather( 1, answers ).squeeze( 1 ) )
        if q_roles == "both":
            advantages = from_q
        else:  # "student": the teacher keeps its game advantage
            advantages = torch.where( ~is_teacher, from_q, advantages )
        eps = torch.tensor( [bool( r.get( "eps" ) ) for r in records], device=student.device )
        advantages = torch.where( eps, torch.zeros_like( advantages ), advantages )
    teacher_terms = -advantages * teacher_t.gather( 1, answers ).squeeze( 1 ) + beta_t * kl( teacher, student.detach() )
    student_terms = beta_s * kl( student, teacher.detach() )
    if plus:
        student_terms = student_terms - advantages * student_t.gather( 1, answers ).squeeze( 1 )
    loss = torch.where( is_teacher, teacher_terms, student_terms ).mean()
    if anchor > 0:
        builtin = [builtin_option( r["kind"], query_options( r["kind"], r["event"], "student" ), r["event"] ) for r in records]
        rows = [n for n, b in enumerate( builtin ) if b is not None]
        if rows:
            picked = teacher[torch.tensor( rows, device=teacher.device ), torch.tensor( [builtin[n] for n in rows], device=teacher.device )]
            loss = loss - anchor * picked.sum() / len( records )
    if q_weight > 0 and q_values is not None:
        targets = [r.get( "qtarget" ) for r in records]
        rows = [n for n, t in enumerate( targets ) if t is not None]
        if rows:
            index = torch.tensor( rows, device=student.device )
            target = torch.tensor( [targets[n] / q_scale for n in rows], dtype=student.dtype, device=student.device )
            picked_q = q_values[index, answers.squeeze( 1 )[index]]
            loss = loss + q_weight * ( picked_q - target ).pow( 2 ).mean()
    return loss


def _pearson( xs: list[float], ys: list[float] ) -> float:
    import math
    import statistics

    mean_x, mean_y = statistics.fmean( xs ), statistics.fmean( ys )
    cov = sum( ( x - mean_x ) * ( y - mean_y ) for x, y in zip( xs, ys ) )
    var_x = sum( ( x - mean_x ) ** 2 for x in xs )
    var_y = sum( ( y - mean_y ) ** 2 for y in ys )
    return cov / math.sqrt( var_x * var_y ) if var_x > 0 and var_y > 0 else 0.0


def q_metrics( model, records: list[dict], obj_vocab: list[int], q_scale: float = Q_TARGET_SCALE, batch: int = 16 ) -> dict:
    """Held-out quality of the pooled advantage head against the realised centred rewards ("qtarget" /
    q_scale; the group-mean baseline predicts 0, its MSE is the targets' mean square): the head's MSE,
    their correlation, and per kind the mean over the answers DEVIATING from the built-in AI's minus the
    mean over its answers, of Q and of the realised target — does Q rank the deviations like the v2
    regression (army below 100% negative, explicit building positive)?"""
    from strategy_net import builtin_option, query_options

    torch = _torch()
    model.eval()
    targets, qs, kinds, devs = [], [], [], []
    for start in range( 0, len( records ), batch ):
        part = records[start:start + batch]
        with torch.no_grad():
            _, q_values = strategic_logits_of( model, part, obj_vocab, "student", q=True )
        for row, record in enumerate( part ):
            target = record.get( "qtarget" )
            if target is None:
                continue
            targets.append( target / q_scale )
            qs.append( float( q_values[row, record["answer"]] ) )
            kinds.append( record["kind"] )
            options = query_options( record["kind"], record["event"], "student" )
            builtin = builtin_option( record["kind"], options, record["event"] )
            devs.append( None if builtin is None else record["answer"] != builtin )
    if not targets:
        return {}
    out = {"mse": round( sum( ( q - t ) ** 2 for q, t in zip( qs, targets ) ) / len( targets ), 4 ),
           "baseline": round( sum( t * t for t in targets ) / len( targets ), 4 ),
           "corr": round( _pearson( qs, targets ), 3 )}
    for kind in sorted( set( kinds ) ):
        dev_q = [q for q, k, d in zip( qs, kinds, devs ) if k == kind and d is True]
        own_q = [q for q, k, d in zip( qs, kinds, devs ) if k == kind and d is False]
        dev_t = [t for t, k, d in zip( targets, kinds, devs ) if k == kind and d is True]
        own_t = [t for t, k, d in zip( targets, kinds, devs ) if k == kind and d is False]
        if dev_q and own_q and dev_t and own_t:
            out[kind] = {"dev_q": round( sum( dev_q ) / len( dev_q ) - sum( own_q ) / len( own_q ), 4 ),
                         "dev_target": round( sum( dev_t ) / len( dev_t ) - sum( own_t ) / len( own_t ), 4 ),
                         "n_dev": len( dev_q )}
    return out


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


def _sample_distinct( probs: list[float], count: int, temperature: float, rng: random.Random ) -> list[int]:
    """`count` DIFFERENT moves drawn without replacement from probs^(1/T) (Gumbel top-k), the most probable first; fewer
    when fewer moves have probability. With replacement a confident policy drew 3.2 distinct moves of 8 and one move in
    12% of the decisions (jolt2): nothing to compare there."""
    import math

    keys = [( math.log( p ) / temperature - math.log( -math.log( rng.random() or 1e-300 ) ), i ) for i, p in enumerate( probs ) if p > 0]
    keys.sort( reverse=True )
    return [i for _, i in keys[:count]]


def _rollout_scores( env, state: dict, legal: list, moves: list[int], side: str, reseeds: range, scores: dict, wins: dict ) -> None:
    """Rollouts of every move with the given reseeds (common random numbers), appended to scores / wins."""
    from battle_grpo import score_parts
    from battle_prefs import SNAP_ID, move_score

    for i in moves:
        path = [( legal[i]["act"], legal[i]["args"] )]
        finals = [env.snapshot_restore( SNAP_ID, path=path, rollout=True, reseed=k ) for k in reseeds]
        scores.setdefault( str( i ), [] ).extend( round( move_score( state, final, side ), 4 ) for final in finals )
        wins.setdefault( str( i ), [] ).extend( int( ( score_parts( state, final, side ) or [0] )[0] ) for final in finals )  # 1 won, -1 lost


def collect_battle( env, setup: dict, model, hint_net, args, rng: random.Random, key: str ) -> tuple[list[dict], dict]:
    """The battle's labeled decisions and its summary (the student's side, the result: 1 won, -1 lost, 0 neither)."""
    from battle_prefs import MAX_STEPS, SNAP_ID, mover_side
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
                draw = _sample_distinct if args.distinct else _sample
                record["teacher"] = draw( teacher, args.group, args.temperature, rng )
                record["student"] = draw( student, args.group, args.temperature, rng )
                side = mover_side( state )
                env.snapshot_save( SNAP_ID )
                scores, wins = {}, {}
                moves = sorted( set( record["teacher"] ) | set( record["student"] ) )
                # Adaptive rollouts: `--screen` rollouts per move first; a decision whose moves all score alike (spread of
                # the means below --min-spread; 80% of the jolt2 decisions differed by < 0.05) is dropped, the others get
                # the remaining rollouts up to --rollouts (same reseeds for every move: common random numbers).
                screen = min( args.screen, args.rollouts ) if args.screen else args.rollouts
                _rollout_scores( env, state, legal, moves, side, range( 1, screen + 1 ), scores, wins )
                means = [statistics.fmean( scores[str( i )] ) for i in moves]
                keep = len( moves ) >= 2 and ( not args.screen or max( means ) - min( means ) >= args.min_spread )
                if keep and screen < args.rollouts:
                    _rollout_scores( env, state, legal, moves, side, range( screen + 1, args.rollouts + 1 ), scores, wins )
                if keep:
                    record.update( {"scores": scores, "wins": wins, "side": side, "battle": key} )
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
    result = ( state or {} ).get( "result" )
    summary = {"battle": key, "side": ours, "result": 1 if result == ours else ( -1 if result in ( "att", "def" ) else 0 ),
               "labeled": len( records )}
    return records, summary


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
    if args.shuffle:
        random.Random( args.seed ).shuffle( setups )  # the maps mixed from the first decision on
    if args.limit:
        setups = setups[:args.limit]
    envs: dict[str, BattleEnv] = {}
    model, loaded, total, started = None, None, 0, time.time()
    os.makedirs( os.path.dirname( args.out ) or ".", exist_ok=True )
    # Every walked battle's result goes to <out>.battles (the student's own win rate; jolt_status.py).
    with open( args.out, "a" ) as out, open( os.path.splitext( args.out )[0] + ".battles", "a" ) as battles:
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
                    records, summary = collect_battle( envs[map_name], setup, model, hint_net, args, rng, key )
                except TimeoutError:
                    print( f"battle {key}: timeout, respawning the engine", flush=True )
                    envs.pop( map_name ).close()
                    continue
                for record in records:
                    record["model_time"] = loaded
                    out.write( json.dumps( record, separators=( ",", ":" ) ) + "\n" )
                out.flush()
                battles.write( json.dumps( dict( summary, model_time=loaded ), separators=( ",", ":" ) ) + "\n" )
                battles.flush()
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
    c.add_argument( "--distinct", action="store_true", help="each role's --group moves are DIFFERENT moves (without replacement)" )
    c.add_argument( "--screen", type=int, default=0, help="adaptive: this many rollouts per move first, the rest only when the "
                    "moves' means differ by >= --min-spread (else the decision is not recorded); 0 = always --rollouts" )
    c.add_argument( "--min-spread", type=float, default=0.05 )
    c.add_argument( "--rate", type=float, default=0.2, help="share of the student's decisions labeled" )
    c.add_argument( "--temperature", type=float, default=1.0 )
    c.add_argument( "--limit", type=int, default=0 )
    c.add_argument( "--shuffle", action="store_true", help="setups in random order (--seed)" )
    c.add_argument( "--seed", type=int, default=7 )
    c.add_argument( "--threads", type=int, default=1 )
    c.add_argument( "--bf16", action="store_true" )
    c.add_argument( "--out", required=True )
    args = parser.parse_args()
    collect( args )


if __name__ == "__main__":
    main()
