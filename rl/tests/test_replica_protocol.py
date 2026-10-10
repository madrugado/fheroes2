"""Integration tests for full real-battle replication (requires the ./fheroes2 binary and game
data; skipped automatically when the binary is missing).

One seeded 14-day Battlefi playtest (built-in AI everywhere) harvests real `battle_start`
setups: hero battles, battles on castle/town tiles and a siege. Every setup is rebuilt in the
headless battle server, and random games — preferring hero spells whenever one is legal — check
the invariants the MCTS replica relies on in the new situations (commander state, hero spells,
towers/bridge/walls/catapult):

- snapshot restore reproduces the walked state at every prefix, restore + suffix reaches the
  final state;
- the main-line fast path equals the full replay from the battle root.

Run against the Debug build too: every spell the enumeration offers must pass the engine's own
spell validation (asserts in Debug).
"""

import os
import random
import sys

import pytest

REPO_ROOT = os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) ) )
BINARY = os.path.join( REPO_ROOT, "fheroes2" )
MAP_NAME = "Battlefi.mp2"
DAYS = 14
SEED = "1"

pytestmark = pytest.mark.skipif( not os.path.exists( BINARY ), reason="fheroes2 binary not built" )

sys.path.insert( 0, os.path.join( REPO_ROOT, "rl" ) )

import harvest_battles  # noqa: E402
from engine_bridge import BattleEnv, new_battle_from_setup  # noqa: E402

SPELLCAST = 2
RETREAT = 6
SURRENDER = 7


def new_battle( env, setup ):
    return new_battle_from_setup( env, setup )


@pytest.fixture( scope="module" )
def setups():
    harvested = harvest_battles.harvest( BINARY, MAP_NAME, DAYS, int( SEED ) )
    assert harvested, "the playtest produced no battles"
    return harvested


@pytest.fixture( scope="module" )
def env():
    env = BattleEnv( binary=BINARY, map_name=MAP_NAME )
    yield env
    env.close()


def pick_move( rng, state ):
    """Random legal move; a hero spell whenever one is legal (half of the time)."""
    spells = [m for m in state["legal"] if m["act"] == SPELLCAST]
    if spells and rng.random() < 0.5:
        return rng.choice( spells )
    # The commander's escapes end the battle at once: random walks keep fighting (test_escapes covers them).
    return rng.choice( [m for m in state["legal"] if m["act"] not in ( RETREAT, SURRENDER )] )


def walk_and_check( env, setup, rng, steps ):
    """Random game from the setup's root with the snapshot/fast-path invariants checked at every
    step. Returns the walked states."""
    state = new_battle( env, setup )
    assert state is not None and state.get( "legal" ), f"battle {setup['bid']} did not rebuild"

    path = []
    finals = [state]
    env.snapshot_save( 1000 )
    while state.get( "legal" ) and not state.get( "result" ) and len( path ) < steps:
        move = pick_move( rng, state )
        probe = [( move["act"], tuple( move["args"] ) )]
        assert env.replay( probe ) == env.replay( probe, full=True ), f"fast path != full replay (battle {setup['bid']})"

        path.append( ( move["act"], list( move["args"] ) ) )
        state = env.action( move["act"], move["args"] )
        assert state.get( "ev" ) == "state", f"a legal move was rejected: {move} -> {state}"
        finals.append( state )
        env.snapshot_save( 1000 + len( path ) )

    for step in range( len( path ) + 1 ):
        assert env.snapshot_restore( 1000 + step ) == finals[step], f"restore to step {step} (battle {setup['bid']})"
    for step in range( 0, len( path ), max( 1, len( path ) // 8 ) ):
        assert env.snapshot_restore( 1000 + step, path=path[step:] ) == finals[-1], f"restore {step} + suffix (battle {setup['bid']})"
    env.snapshots_free()

    return path, finals


def test_every_real_battle_is_searchable_and_rebuilds( setups, env ):
    kinds = {"hero": 0, "castle": 0, "garrison": 0}
    for setup in setups:
        assert setup["searchable"] == 1
        state = new_battle( env, setup )
        assert state is not None and state.get( "ev" ) == "state" and state.get( "legal" ), f"battle {setup['bid']}: {state}"
        kinds["hero"] += any( "hid" in setup[side] for side in ( "att", "def" ) )
        kinds["castle"] += "castle" in setup
        kinds["garrison"] += bool( setup["def"].get( "garrison" ) )
    assert kinds["hero"] > 0 and kinds["castle"] > 0 and kinds["garrison"] > 0, kinds


def test_siege_snapshots_and_fast_path( setups, env ):
    """Towers, bridge, walls, catapult and the per-round catapult/tower flags survive
    snapshot/restore; the siege state is part of the state reply."""
    rng = random.Random( 7 )
    sieges = [s for s in setups if "castle" in s and "siege" in ( new_battle( env, s ) or {} )]
    assert sieges, "no siege in the harvested battles"

    for setup in sieges:
        _, finals = walk_and_check( env, setup, rng, steps=80 )
        root_siege = finals[0]["siege"]
        assert set( root_siege ) == {"cells", "towers", "bridge"}
        # The catapult or the towers changed the siege state during the walk.
        assert any( state.get( "siege" ) != root_siege for state in finals[1:] ), "the siege state never changed"


def test_hero_spells_snapshots_and_fast_path( setups, env ):
    """Hero spells are legal moves; casting one updates the commander state ("heroes": spell
    points, cast flag) and keeps snapshots/fast path exact (summoned units, mirror images,
    resurrections change the unit set)."""
    rng = random.Random( 11 )
    casts = 0
    for setup in setups:
        root = new_battle( env, setup )
        if not any( m["act"] == SPELLCAST for m in ( root or {} ).get( "legal", [] ) ):
            continue

        path, finals = walk_and_check( env, setup, rng, steps=40 )
        for ( act, _ ), before, after in zip( path, finals, finals[1:] ):
            if act != SPELLCAST:
                continue
            casts += 1
            side = next( u["side"] for u in before["units"] if u["u"] == before["cur"] )
            hero_before = next( h for h in before["heroes"] if h["side"] == side )
            hero_after = next( h for h in after["heroes"] if h["side"] == side )
            assert hero_before["cast"] == 0 and hero_after["cast"] == 1
            assert hero_after["sp"] < hero_before["sp"]
            # One spell per round: no further spell of that side is legal right after the cast.
            if after.get( "legal" ) and after["cur"] == before["cur"]:
                assert not any( m["act"] == SPELLCAST for m in after["legal"] )
        if casts >= 8:
            break

    assert casts > 0, "no hero spell was cast"


def test_builtin_ai_actions_are_legal_in_hero_battles( setups, env ):
    """The built-in AI's action (the "suggest" op, used by the gate runner) must be in the legal
    list, including its hero spells: the spell enumeration covers the targets the AI picks."""
    rng = random.Random( 3 )
    checked = spells = 0
    for setup in setups:
        state = new_battle( env, setup )
        if not any( m["act"] == SPELLCAST for m in ( state or {} ).get( "legal", [] ) ):
            continue
        for _ in range( 15 ):
            if not state.get( "legal" ) or state.get( "result" ):
                break
            expert = env.suggest()["expert"]
            if expert["act"] == SPELLCAST:
                # Spells exactly (the target is part of the move).
                assert expert in state["legal"], f"built-in spell {expert} is not legal (battle {setup['bid']})"
                spells += 1
            # (The built-in AI leaves ATTACK fields for the engine to resolve (-1), so its unit actions
            # are not literally in the list; the action op below must still accept them.)
            checked += 1
            move = expert if rng.random() < 0.7 else rng.choice( state["legal"] )
            state = env.action( move["act"], move["args"] )
            assert state.get( "ev" ) == "state", f"{move} was rejected: {state}"
    assert checked > 0 and spells > 0, ( checked, spells )


def test_escapes_are_legal_moves_that_end_the_battle( setups, env ):
    """A hero commander may retreat (not from a defended castle) or surrender (to a hero, if the kingdom
    can pay): both are in the legal list exactly then, end the battle for the other side, and replay
    deterministically from a snapshot."""
    seen = {RETREAT: 0, SURRENDER: 0}
    for setup in setups:
        state = new_battle( env, setup )
        if not state or not state.get( "legal" ):
            continue
        mover = next( u["side"] for u in state["units"] if u["u"] == state["cur"] )
        escapes = [m for m in state["legal"] if m["act"] in ( RETREAT, SURRENDER )]
        if "hero" not in setup[mover]:
            assert not escapes, f"escape without a hero commander (battle {setup['bid']})"
            continue
        env.snapshot_save( 1 )
        for move in escapes:
            assert move["args"] == []
            final = env.snapshot_restore( 1, path=[( move["act"], [] )] )
            assert final["result"] == ( "def" if mover == "att" else "att" ), f"{move} did not end the battle: {final.get('result')}"
            assert env.replay( [( move["act"], () )], full=True ) == final
            seen[move["act"]] += 1
        env.snapshots_free()
    assert seen[RETREAT] > 0, seen


def test_surrender_follows_the_real_games_gold( setups, env ):
    """battle_start carries each side's kingdom gold and the replica uses it: surrender is legal with the gold to
    pay for it and not without (the map's starting gold used to decide)."""
    affordable = 0
    for setup in setups:
        state = new_battle( env, setup )
        if not state or not state.get( "legal" ):
            continue
        mover = next( u["side"] for u in state["units"] if u["u"] == state["cur"] )
        if "hero" not in setup[mover]:
            continue
        assert setup[mover].get( "gold" ) is not None
        rich = dict( setup, **{mover: dict( setup[mover], gold=10 ** 7 )} )
        poor = dict( setup, **{mover: dict( setup[mover], gold=0 )} )
        assert not any( m["act"] == SURRENDER for m in new_battle( env, poor )["legal"] )
        affordable += any( m["act"] == SURRENDER for m in new_battle( env, rich )["legal"] )
    assert affordable > 0
