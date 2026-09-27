"""Strategic target-choice policies (hero target decisions, see src/fheroes2/ai/ai_decision.cpp).

A policy is a callable `policy(decision_event) -> chosen candidate dict | None` (None = keep the
built-in choice). Context-aware policies may also implement `observe_turn(turn_context_event)`,
which the runners call at the start of every AI kingdom turn.

    greedy  — the highest built-in value;
    random  — uniform over the candidates;
    builtin — always skip (the engine's own choice);
    tempo   — value/distance-aware, see TempoPolicy;
    learned — advantage model trained on counterfactual rollouts, see LearnedPolicy.
"""

from __future__ import annotations

import math
import random


def greedy_policy( decision: dict ) -> dict | None:
    cands = decision.get( "cands" ) or []
    return max( cands, key=lambda c: c["v"] ) if cands else None


# Explicit "do nothing" answer of a build/hire policy method (None means "built-in choice").
NOTHING = "nothing"


class RandomPolicy:
    """Uniform over the candidates of every strategic choice (hire: "nothing" is one option)."""

    def __init__( self, rng: random.Random ):
        self.rng = rng

    def __call__( self, decision: dict ) -> dict | None:
        cands = decision.get( "cands" ) or []
        return self.rng.choice( cands ) if cands else None

    def build( self, ev: dict ):
        cands = ev.get( "cands" ) or []
        return self.rng.choice( cands ) if cands else None

    def hire( self, ev: dict ):
        options = list( ev.get( "cands" ) or [] ) + [NOTHING]
        return self.rng.choice( options )


def random_policy_factory( rng: random.Random ):
    return RandomPolicy( rng )


def builtin_policy( _decision: dict ) -> dict | None:
    return None


class TempoPolicy:
    """Value/distance-aware target choice on top of the built-in evaluation.

    The built-in value `v` already folds in the path distance, but it is evaluated per hero in
    isolation. This policy adds two things the per-hero evaluation lacks:

    - tempo: a target reachable with the hero's move points is worth its full value, a farther
      one is discounted by `gamma` per extra turn of travel (turns = ceil((d - mp) / mmp));
    - coordination: a tile already claimed by another hero during the same kingdom turn is
      penalized by `claim_penalty`, so heroes spread over the map instead of racing each other.

    Move points come from the last `turn_context` (start of the kingdom turn), so for heroes that
    already moved this turn the reach is an optimistic estimate. Without a context for the hero
    the tempo factor is skipped (pure value + coordination).
    """

    def __init__( self, gamma: float = 0.85, claim_penalty: float = 0.5 ):
        self.gamma = gamma
        self.claim_penalty = claim_penalty
        self._heroes: dict[int, dict] = {}
        self._claims: dict[int, int] = {}  # tile -> hero id, reset every kingdom turn

    def observe_turn( self, turn_context: dict ) -> None:
        self._heroes = {hero["id"]: hero for hero in turn_context.get( "heroes" ) or []}
        self._claims = {}

    def score( self, decision: dict, candidate: dict ) -> float:
        value = float( candidate["v"] )

        hero = self._heroes.get( decision.get( "h" ) )
        if hero is not None and hero.get( "mmp", 0 ) > 0:
            extra = max( 0, candidate["d"] - hero.get( "mp", 0 ) )
            value *= self.gamma ** math.ceil( extra / hero["mmp"] )

        owner = self._claims.get( candidate["i"] )
        if owner is not None and owner != decision.get( "h" ):
            value *= self.claim_penalty

        return value

    def __call__( self, decision: dict ) -> dict | None:
        cands = decision.get( "cands" ) or []
        if not cands:
            return None

        # max() keeps the first of equal scores: the engine sends candidates sorted by value.
        best = max( cands, key=lambda c: self.score( decision, c ) )
        self._claims[best["i"]] = decision.get( "h" )
        return best


class LearnedPolicy:
    """Picks by the advantage model of strategy_model.py (trained on counterfactual rollouts):
    among the top-`top` candidates by built-in value, the one with the highest predicted
    advantage over the built-in choice if it beats it by more than `margin`; otherwise keeps the
    built-in choice (returns None)."""

    def __init__( self, model_path: str ):
        import json

        with open( model_path ) as f:
            self.model = json.load( f )
        self.top = int( self.model.get( "top", 4 ) )
        self.margin = float( self.model.get( "margin", 0.0 ) )
        self._contexts: dict[str, dict] = {}

    def observe_turn( self, turn_context: dict ) -> None:
        self._contexts[turn_context.get( "p" )] = turn_context

    def __call__( self, decision: dict ) -> dict | None:
        from strategy_model import candidate_features, predict  # numpy only at use time

        cands = decision.get( "cands" ) or []
        if len( cands ) < 2:
            return None

        context = self._contexts.get( decision.get( "p" ) )
        count = min( self.top, len( cands ) )
        rows = [candidate_features( decision, context, j, self.model["obj_vocab"] ) for j in range( count )]
        pred = predict( self.model, rows )
        best = max( range( count ), key=lambda j: pred[j] )
        if best == 0 or pred[best] - pred[0] <= self.margin:
            return None
        return cands[best]


class ForColor:
    """Applies `policy` only to the decisions of one player (the "p" color of the events); the
    other players keep the built-in choice. Used for head-to-head comparisons (strategy_bench.py)."""

    def __init__( self, policy, color: str ):
        self.policy = policy
        self.color = color

    def observe_turn( self, turn_context: dict ) -> None:
        if turn_context.get( "p" ) == self.color and hasattr( self.policy, "observe_turn" ):
            self.policy.observe_turn( turn_context )

    def __call__( self, decision: dict ) -> dict | None:
        if decision.get( "p" ) != self.color:
            return None
        return self.policy( decision )

    def build( self, ev: dict ):
        method = getattr( self.policy, "build", None )
        return method( ev ) if method is not None and ev.get( "p" ) == self.color else None

    def hire( self, ev: dict ):
        method = getattr( self.policy, "hire", None )
        return method( ev ) if method is not None and ev.get( "p" ) == self.color else None


STRATEGIC_QUERIES = ( "decision", "build", "hire" )


def strategic_reply( policy, ev: dict ) -> tuple[dict, dict]:
    """Answers one strategic query (`decision` = hero target, `build`, `hire`) with `policy`.

    Returns (reply operation for the engine, record of the choice). Policies answer targets via
    `policy(ev)` and may implement `build(ev)` / `hire(ev)` returning a candidate dict, NOTHING, or
    None (= built-in choice); a policy without the method keeps the built-in choice. Record field
    `chosen`: target tile / building id (0 = nothing) / hire candidate index (-1 = nothing), or
    None when the built-in AI decided.
    """
    kind = ev.get( "ev" )
    record = {"kind": "target" if kind == "decision" else kind, "t": ev.get( "t" ), "p": ev.get( "p" ), "cands": ev.get( "cands" )}

    if kind == "decision":
        chosen = policy( ev )
        record.update( h=ev.get( "h" ), chosen=None if chosen is None else chosen["i"] )
        record["from"] = ev.get( "from" )
        if chosen is None:
            return {"op": "skip"}, record
        return {"op": "pick", "h": ev["h"], "i": chosen["i"]}, record

    method = getattr( policy, kind, None )
    choice = method( ev ) if method is not None else None

    if kind == "build":
        record.update( castle=ev.get( "castle" ), res=ev.get( "res" ), defensive=ev.get( "defensive" ) )
        if choice is None:
            record["chosen"] = None
            return {"op": "skip"}, record
        building = 0 if choice == NOTHING else choice["b"]
        record["chosen"] = building
        return {"op": "build", "castle": ev.get( "castle" ), "b": building}, record

    if kind == "hire":
        record.update( res=ev.get( "res" ), heroes=ev.get( "heroes" ), bi=ev.get( "bi" ) )
        if choice is None:
            record["chosen"] = None
            return {"op": "skip"}, record
        if choice == NOTHING:
            record["chosen"] = -1
            return {"op": "hire", "castle": -1}, record
        record["chosen"] = ( ev.get( "cands" ) or [] ).index( choice )
        return {"op": "hire", "castle": choice["castle"], "slot": choice["slot"]}, record

    raise ValueError( f"not a strategic query: {kind}" )


def attach_build_result( records: list[dict], ev: dict ) -> None:
    """Stores a `build_result` event on the latest matching `build` record (if any)."""
    for record in reversed( records ):
        if record.get( "kind" ) == "build" and record.get( "castle" ) == ev.get( "castle" ) and record.get( "t" ) == ev.get( "t" ):
            if "result" not in record:
                record["result"] = ev.get( "b" )
                record["src"] = ev.get( "src" )
            return


STRATEGY_POLICIES = ( "greedy", "random", "builtin", "tempo", "learned" )
DEFAULT_MODEL = "az/models/strategy_model.json"


def make_strategy_policy( name: str, rng: random.Random, model_path: str = DEFAULT_MODEL ):
    if name == "greedy":
        return greedy_policy
    if name == "random":
        return random_policy_factory( rng )
    if name == "builtin":
        return builtin_policy
    if name == "tempo":
        return TempoPolicy()
    if name == "learned":
        return LearnedPolicy( model_path )
    raise ValueError( f"unknown strategy policy: {name}" )
