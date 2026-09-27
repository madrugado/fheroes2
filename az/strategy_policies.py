"""Strategic target-choice policies (hero target decisions, see src/fheroes2/ai/ai_decision.cpp).

A policy is a callable `policy(decision_event) -> chosen candidate dict | None` (None = keep the
built-in choice). Context-aware policies may also implement `observe_turn(turn_context_event)`,
which the runners call at the start of every AI kingdom turn.

    greedy  — the highest built-in value;
    random  — uniform over the candidates;
    builtin — always skip (the engine's own choice);
    tempo   — value/distance-aware, see TempoPolicy.
"""

from __future__ import annotations

import math
import random


def greedy_policy( decision: dict ) -> dict | None:
    cands = decision.get( "cands" ) or []
    return max( cands, key=lambda c: c["v"] ) if cands else None


def random_policy_factory( rng: random.Random ):
    def policy( decision: dict ) -> dict | None:
        cands = decision.get( "cands" ) or []
        return rng.choice( cands ) if cands else None

    return policy


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


STRATEGY_POLICIES = ( "greedy", "random", "builtin", "tempo" )


def make_strategy_policy( name: str, rng: random.Random ):
    if name == "greedy":
        return greedy_policy
    if name == "random":
        return random_policy_factory( rng )
    if name == "builtin":
        return builtin_policy
    if name == "tempo":
        return TempoPolicy()
    raise ValueError( f"unknown strategy policy: {name}" )
