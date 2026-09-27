"""External battle agent for real fheroes2 battles (see az/README.md and
src/fheroes2/battle/battle_agent.cpp).

The runner spawns the engine in the autonomous playtest mode with FHEROES2_BATTLE_AGENT=1.
At every AI unit activation the engine sends the battle state (units, obstacles, legal
moves — the same wire format as the headless battle server) and blocks until this process
replies. Decision policies:

    random   — uniform over the legal moves (channel smoke tests);
    planner  — delegate every decision to the built-in AI (in-game baseline);
    policy   — trained policy/value network, argmax over the legal moves;
    mcts     — full MCTS (optionally net-guided) in a headless engine replica reconstructed
               from the "battle_start" setup. Exact for commander-less armies; in hero
               battles the replica lacks commander stats, so a state mismatch degrades the
               rest of the battle to the policy network.

Every decision and battle outcome is recorded to az/data/battle_agent_<policy>.jsonl.

Usage:
    az/.venv/bin/python az/battle_agent.py --policy planner --days 7
    az/.venv/bin/python az/battle_agent.py --policy policy --model az/models/<ckpt> --arch transformer --device mps
    az/.venv/bin/python az/battle_agent.py --policy mcts --sims 32 --model az/models/<ckpt>
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time

sys.path.insert( 0, os.path.dirname( os.path.abspath( __file__ ) ) )

from engine_bridge import BattleEnv  # noqa: E402
from line_reader import LineReader as _LineReader  # noqa: E402

# Battle rounds after which the built-in AI takes over the rest of the battle (see decide()).
DEFAULT_MAX_BATTLE_TURNS = 30

# State fields compared to detect that the replica diverged from the real battle.
STATE_SYNC_FIELDS = ( "turn", "cur", "units", "obstacles" )


def format_stacks( stacks: list ) -> str:
    """[slot, monId, count] rows -> 'slot:mon x count' CSV for the battle server "new" op."""
    return ",".join( f"{slot}:{mon}x{count}" for slot, mon, count in stacks )


def states_equal( real: dict, replica: dict ) -> bool:
    return all( real.get( key ) == replica.get( key ) for key in STATE_SYNC_FIELDS )


class BattleAgentRunner:
    """Spawns the engine with FHEROES2_BATTLE_AGENT=1 and answers its battle decision queries."""

    def __init__( self, binary: str, map_name: str, days: int, playthroughs: int, policy: str,
                  model=None, sims: int = 32, seed: int = 2026, extra_env: dict | None = None,
                  max_battle_turns: int = DEFAULT_MAX_BATTLE_TURNS ):
        # The engine gets the battle agent flag; the headless replica must NOT have it (it
        # speaks the battle-server protocol instead).
        base_env = dict( os.environ )
        base_env.pop( "FHEROES2_BATTLE_AGENT", None )
        # A stray strategic flag would make the engine block on decisions nobody answers.
        base_env.pop( "FHEROES2_STRATEGY_SERVER", None )
        base_env["FHEROES2_AUTO_PLAYTEST"] = str( playthroughs )
        base_env["FHEROES2_AUTO_PLAYTEST_DAYS"] = str( days )
        base_env["FHEROES2_AUTO_PLAYTEST_MAP"] = map_name
        base_env["FHEROES2_BATTLE_AGENT"] = "1"
        base_env.update( extra_env or {} )  # e.g. the strategic channel for the game agent

        self.proc = subprocess.Popen(
            [binary],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=False,
            env=base_env,
            bufsize=0,
        )
        self._reader = _LineReader( self.proc.stdout.fileno() )

        self.policy_name = policy
        self.model = model
        self.sims = sims
        self.max_battle_turns = max_battle_turns
        self.rng = random.Random( seed )
        self.map_name = map_name

        self._setup: dict | None = None  # battle_start payload of the current battle
        self._pending_action: tuple | None = None  # real action not yet mirrored into the replica
        self._replica_env: BattleEnv | None = None
        self._replica_tried = False  # replica start attempted for the current battle
        self._replica_synced = False
        self._replica_desynced = False
        self._replica_state: dict | None = None
        self._records: list[dict] = []

    # --- replica management ---------------------------------------------------------------

    def _replica_start( self, state: dict ) -> bool:
        """Reconstructs the current battle in the headless replica; returns False when the
        battle cannot be replicated (siege) or the replica process is gone. Called once per
        battle (at its first decision point)."""
        self._replica_tried = True

        if not self._setup or not self._setup.get( "searchable" ):
            return False

        if self._replica_env is None:
            try:
                # The replica must load the same map as the real game: obstacle placement on
                # the battle tile derives from the terrain and objects of this very tile.
                self._replica_env = BattleEnv( map_name=self.map_name )
            except OSError as error:
                print( f"battle_agent: replica spawn failed ({error}); policy mode", flush=True )
                return False

        att, dfd = self._setup["att"], self._setup["def"]
        reply = self._replica_env.new_battle(
            seed=self._setup["seed"],
            attacker=format_stacks( att["stacks"] ),
            defender=format_stacks( dfd["stacks"] ),
            tile=self._setup["tile"],
            world_seed=self._setup["wseed"],
            spread_att=bool( att["spread"] ),
            spread_def=bool( dfd["spread"] ),
        )
        if reply is None or "legal" not in reply:
            print( "battle_agent: replica battle failed; policy mode", flush=True )
            return False

        self._replica_desynced = False
        self._replica_synced = states_equal( state, reply )
        self._replica_state = reply
        if not self._replica_synced:
            print( "battle_agent: replica root state mismatch; policy mode", flush=True )
            return False

        return True

    def _replica_mirror( self, state: dict ) -> None:
        """Applies the previous real action to the replica and checks that the replica state
        still matches the real one."""
        if self._replica_env is None or self._replica_desynced or self._pending_action is None:
            return

        act, args = self._pending_action
        reply = self._replica_env.action( act, list( args ) )
        self._pending_action = None

        if reply is None:
            self._replica_desynced = True
            return

        if not states_equal( state, reply ):
            self._replica_desynced = True
            print( "battle_agent: replica state mismatch (hero battle?); policy mode", flush=True )
            return

        self._replica_state = reply

    def _replica_close( self ) -> None:
        if self._replica_env is not None:
            try:
                self._replica_env.close()
            except Exception:
                pass
            self._replica_env = None

    # --- decisions -------------------------------------------------------------------------

    def _policy_argmax( self, state: dict ) -> tuple | None:
        """Argmax of the trained policy over the legal moves."""
        if self.model is None:
            return None

        priors, _ = self.model.evaluate( state )
        legal = state["legal"]
        best, best_score = 0, -1.0
        for i in range( len( legal ) ):
            score = priors.get( i, 0.0 )
            if score > best_score:
                best, best_score = i, score

        move = legal[best]
        return move["act"], tuple( move["args"] )

    def decide( self, state: dict ) -> tuple | None:
        """Returns (act, args) or None to delegate the decision to the built-in AI."""
        if state.get( "turn", 0 ) > self.max_battle_turns:
            # Battles in the engine have no round limit: two sides driven by a weak agent (e.g. MCTS
            # without a network) can dance forever without engaging (seen: 131k moves in one
            # battle). The built-in AI finishes long battles.
            return None
        if self.policy_name == "planner":
            return None

        if self.policy_name == "random":
            move = self.rng.choice( state["legal"] )
            return move["act"], tuple( move["args"] )

        if self.policy_name == "policy":
            return self._policy_argmax( state )

        # MCTS mode: the search runs in the replica, which must be in sync with the real battle.
        if self._replica_state is None and not self._replica_tried and not self._replica_start( state ):
            return self._policy_argmax( state )

        if self._replica_desynced or self._replica_state is None:
            return self._policy_argmax( state )

        from mcts import Mcts  # deferred: torch import only when the search is actually used

        legal, counts = Mcts( self._replica_env, policy_value=self.model, rng=self.rng, root_noise=0.0 ).run( self._replica_state, self.sims )
        if not legal:
            return None

        best = max( range( len( legal ) ), key=lambda i: counts[i] )
        act, args = legal[best]

        # The search optimized the replica physics; the engine has the final word on legality.
        real_moves = {( m["act"], tuple( m["args"] ) ) for m in state["legal"]}
        if ( act, args ) not in real_moves:
            self._replica_desynced = True
            print( "battle_agent: replica move is illegal in the real battle; policy mode", flush=True )
            return self._policy_argmax( state )

        return act, args

    # --- engine event loop -------------------------------------------------------------------

    def run( self, on_record=None ) -> list[dict]:
        """Runs the game(s) to completion; returns the game_end events."""
        summaries: list[dict] = []
        assert self.proc.stdin is not None

        while True:
            line = self._reader.read_line()
            if line is None:
                break

            try:
                ev = json.loads( line )
            except json.JSONDecodeError:
                continue

            if self._handle_event( ev ):
                continue

            kind = ev.get( "ev" )
            if kind == "battle_start":
                self._setup = ev
                self._pending_action = None
                self._replica_tried = False
                self._replica_state = None
                self._replica_synced = False
                self._replica_desynced = False
            elif kind == "state" and "bid" in ev:
                # A decision query: the wire format is exactly the battle-server state reply
                # extended with the battle id and the searchability flag.
                if self.policy_name == "mcts":
                    self._replica_mirror( ev )

                start = time.monotonic()
                decision = self.decide( ev )
                elapsed = time.monotonic() - start

                record = {
                    "bid": ev.get( "bid" ),
                    "turn": ev.get( "turn" ),
                    "cur": ev.get( "cur" ),
                    "n_legal": len( ev.get( "legal", [] ) ),
                    "chosen": None if decision is None else list( decision[1] ),
                    "act": None if decision is None else decision[0],
                    "policy_ms": round( elapsed * 1000, 1 ),
                    "replica_synced": None if self.policy_name != "mcts" else ( not self._replica_desynced and self._replica_synced ),
                }
                if on_record is not None:
                    on_record( record )
                self._records.append( record )

                if decision is None:
                    self._send( {"op": "planner"} )
                else:
                    act, args = decision
                    self._send( {"op": "action", "act": act, "args": list( args )} )
                    self._pending_action = ( act, args )
            elif kind == "battle_fallback":
                # The engine rejected our action and used the planner instead: the replica can
                # no longer follow this battle.
                self._pending_action = None
                self._replica_desynced = True
            elif kind == "battle_end":
                self._setup = None
                self._pending_action = None
            elif kind == "game_end":
                summaries.append( ev )
                for record in self._records:
                    record["game_end"] = {"day": ev.get( "day" ), "results": ev.get( "results" )}
                self._records = []

        return summaries

    def _handle_event( self, ev: dict ) -> bool:
        """Extension point for subclasses serving more channels over the same pipe (see
        game_agent.py). Returns True when the event was consumed."""
        return False

    def _send( self, obj: dict ) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write( ( json.dumps( obj, separators=( ",", ":" ) ) + "\n" ).encode() )
        self.proc.stdin.flush()

    def close( self ) -> None:
        self._replica_close()

        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.wait( timeout=10 )
            except subprocess.TimeoutExpired:
                # The engine fell back to the built-in AI and would finish the whole playtest
                # on its own; terminate it.
                self.proc.terminate()
                try:
                    self.proc.wait( timeout=5 )
                except subprocess.TimeoutExpired:
                    self.proc.kill()

        if self.proc.stdin is not None and not self.proc.stdin.closed:
            try:
                self.proc.stdin.close()
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser( description="External battle agent for real fheroes2 battles" )
    parser.add_argument( "--policy", choices=["random", "planner", "policy", "mcts"], default="planner" )
    parser.add_argument( "--binary", type=str, default="./fheroes2" )
    parser.add_argument( "--map", type=str, default="Arena.mp2" )
    parser.add_argument( "--days", type=int, default=7 )
    parser.add_argument( "--playthroughs", type=int, default=1 )
    parser.add_argument( "--sims", type=int, default=32, help="MCTS simulations per decision" )
    parser.add_argument( "--max-battle-turns", type=int, default=DEFAULT_MAX_BATTLE_TURNS, help="rounds before the built-in AI takes over a battle" )
    parser.add_argument( "--model", type=str, default=None, help="trained network checkpoint (policy/mcts)" )
    parser.add_argument( "--arch", choices=["resnet", "transformer"], default="resnet" )
    parser.add_argument( "--device", type=str, default="cpu" )
    parser.add_argument( "--seed", type=int, default=2026 )
    parser.add_argument( "--out", type=str, default="az/data" )
    args = parser.parse_args()

    if args.policy == "policy" or ( args.policy == "mcts" and args.model is not None ):
        # The network is required for the policy mode; in the MCTS mode it is optional
        # (without it the search uses uniform priors + the material heuristic).
        from selfplay import load_policy_value

        model = load_policy_value( args.model, args.arch, args.device )
        print( f"model loaded: {args.model} ({args.arch}) on {args.device}", flush=True )
    else:
        model = None

    runner = BattleAgentRunner(
        binary=args.binary,
        map_name=args.map,
        days=args.days,
        playthroughs=args.playthroughs,
        policy=args.policy,
        model=model,
        sims=args.sims,
        seed=args.seed,
        max_battle_turns=args.max_battle_turns,
    )

    os.makedirs( args.out, exist_ok=True )
    out_path = os.path.join( args.out, f"battle_agent_{args.policy}.jsonl" )
    records: list[dict] = []

    t0 = time.time()
    try:
        for item in runner.run( on_record=records.append ):
            print( f"game_end: day {item.get( 'day' )} results={item.get( 'results' )}", flush=True )
    finally:
        runner.close()

        with open( out_path, "w" ) as out:
            for record in records:
                out.write( json.dumps( record ) + "\n" )

    desynced = sum( 1 for r in records if r.get( "replica_synced" ) is False )
    print( f"done in {time.time() - t0:.1f}s: {len( records )} decisions -> {out_path}" )
    if args.policy == "mcts":
        print( f"replica: {desynced}/{len( records )} decisions degraded to policy mode" )


if __name__ == "__main__":
    main()
