"""Unit tests for the strategic output of the unified transformer (no engine)."""

import json
import math
import os
import sys

sys.path.insert( 0, os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), ".." ) )

import pytest

torch = pytest.importorskip( "torch" )
pytest.importorskip( "transformers" )

import strategy_net  # noqa: E402
from transformer_model import STRAT_FEATURE_W, STRAT_TOKEN_W, AzBattleTransformer, load_checkpoint, save_checkpoint, strategic_logits  # noqa: E402

CONTEXT = {"ev": "turn_context", "t": 3, "p": "Blue", "res": [5, 0, 5, 0, 0, 0, 3000],
           "castles": [{"i": 100}], "heroes": [{"id": 7, "mp": 900, "mmp": 1200, "str": 400.0}]}
TARGET = {"ev": "decision", "t": 3, "p": "Blue", "h": 7,
          "cands": [{"i": 10, "obj": 138, "v": 300.0, "d": 200}, {"i": 11, "obj": 134, "v": 90.0, "d": 100},
                    {"i": 12, "obj": 138, "v": 20.0, "d": 900}]}
ARMY = {"ev": "army", "t": 3, "p": "Blue", "castle": 100, "reason": "visit", "garrison": 50.0, "hero": 400.0,
        "offer": [{"mon": 1, "avail": 10, "n": 10, "str": 11.0}]}


def test_query_tokens_have_the_model_width():
    prefix, context, options = strategy_net.query_tokens( "target", TARGET, CONTEXT, [138] )
    assert len( context ) == STRAT_TOKEN_W and token_type( context ) == "context"
    assert len( options ) == 3 and all( len( o ) == STRAT_TOKEN_W and token_type( o ) == "option" for o in options )
    assert len( prefix ) == 2 and prefix[0][2] == 1.0  # the query's hero, then the castle
    assert token_type( prefix[0] ) == "hero" and token_type( prefix[1] ) == "castle"


def token_type( token: list[float] ) -> str:
    types = strategy_net.STRAT_TOKEN_TYPES
    return types[token[STRAT_FEATURE_W - len( types ):STRAT_FEATURE_W].index( 1.0 )]


def test_strategic_logits_are_masked_and_independent_of_padding():
    model = AzBattleTransformer().eval()
    target = strategy_net.query_tokens( "target", TARGET, CONTEXT, [] )
    army = strategy_net.query_tokens( "army", ARMY, CONTEXT, [] )
    with torch.no_grad():
        logits, mask = strategic_logits( model, [target, army] )
        alone, _ = strategic_logits( model, [army] )
    assert mask.tolist() == [[True, True, True], [True, True, True]]
    assert torch.allclose( logits[1], alone[0], atol=1e-5 )


def test_option_scores_see_every_option():
    """The second copy of the options is scored: changing the LAST option changes the first score."""
    model = AzBattleTransformer().eval()
    prefix, context, options = strategy_net.query_tokens( "target", TARGET, CONTEXT, [] )
    changed = [list( o ) for o in options]
    changed[-1][0] += 5.0
    with torch.no_grad():
        a, _ = strategic_logits( model, [( prefix, context, options )] )
        b, _ = strategic_logits( model, [( prefix, context, changed )] )
    assert abs( float( a[0, 0] - b[0, 0] ) ) > 1e-6


def test_checkpoints_without_the_strategic_head_still_load( tmp_path ):
    model = AzBattleTransformer()
    state = {k: v for k, v in model.state_dict().items() if not k.startswith( ( "strat_proj.", "strat_head." ) )}
    path = tmp_path / "old.pt"
    torch.save( {"arch": "transformer", "config": model.config, "state_dict": state}, path )
    loaded = load_checkpoint( str( path ) )
    assert hasattr( loaded, "strat_head" )
    save_checkpoint( loaded, str( tmp_path / "new.pt" ) )
    load_checkpoint( str( tmp_path / "new.pt" ) )


def test_preference_pairs_from_rollout_labels():
    base = {"seed": 1, "map": "m", "horizon": 7, "n": 5, "kind": "army", "event": ARMY, "context": CONTEXT,
            "builtin_index": 2, "base": {}}
    records = [dict( base, option_index=0, option=0, delta={"outcome": 0, "k": 0, "h": 0, "str": -900, "g": 0} ),
               dict( base, option_index=1, option=50, delta={"outcome": 0, "k": 1, "h": 0, "str": 0, "g": 0} )]
    ( pair, ) = strategy_net.preference_pairs( records, margin=100 )
    assert pair["chosen"] == 1 and pair["rejected"] == 0  # 50% (+1 castle) > 100% (0) > 0% (-900)
    assert pair["scores"] == {"2": 0.0, "0": -900.0, "1": 2000.0}


def test_net_policy_answers_every_kind( tmp_path ):
    path = tmp_path / "m.pt"
    save_checkpoint( AzBattleTransformer(), str( path ) )
    policy = strategy_net.NetStrategyPolicy( str( path ) )
    policy.observe_turn( CONTEXT )
    probs = policy.probabilities( "target", TARGET )
    assert len( probs ) == 3 and abs( sum( probs ) - 1.0 ) < 1e-5
    assert policy.army( ARMY ) in ( 0, 50, 100 )
    choice = policy( TARGET )
    assert choice is None or choice in TARGET["cands"]


def test_policy_branch_answers_only_its_color_and_the_picked_query():
    import strategy_games

    class Fixed:
        """decide/record/history interface of NetStrategyPolicy with fixed choices."""

        def __init__( self ):
            self.reset()

        def reset( self ):
            self.recorded = []

        def observe_turn( self, ctx ):
            pass

        def history( self, color ):
            return {"days": [], "decisions": list( self.recorded )}

        def decide( self, kind, event ):
            return 1

        def record( self, kind, event, index ):
            self.recorded.append( ( kind, index ) )

    other = dict( ARMY, p="Red" )
    policy = Fixed()
    branch = strategy_games.PolicyBranch( policy, "Blue", pick_at=2, answer_index=0, expected=ARMY )
    assert branch( TARGET ) == TARGET["cands"][1]       # query 0: our policy (option 1)
    assert branch.army( other ) is None                  # query 1: another color -> built-in
    assert branch.army( ARMY ) == 0                      # query 2: the branch's answer (option 0 = 0%)
    assert branch.army( ARMY ) == 50                     # query 3: our policy again (option 1 = 50%)
    assert [q["n"] for q in branch.queries] == [0, 2, 3] and not branch.diverged
    # The policy's history follows the answers actually given, including the forced one.
    assert policy.recorded == [( "target", 1 ), ( "army", 0 ), ( "army", 1 )]
    assert len( branch.queries[2]["history"]["decisions"] ) == 2


def test_option_index_of_builtin_answers():
    import strategy_games

    options = strategy_net.query_options( "army", ARMY )
    assert strategy_games.option_index( "army", options, None, ARMY ) == options.index( 100 )
    assert strategy_games.option_index( "target", strategy_net.query_options( "target", TARGET ), None, TARGET ) == 0
    build = {"ev": "build", "t": 3, "p": "Blue", "castle": 100, "race": 1, "res": [5] * 7,
             "cands": [{"b": 16, "name": "Statue", "trade": 0, "cost": [0, 0, 0, 0, 0, 0, 1250]}]}
    options = strategy_net.query_options( "build", build )
    assert options[0] == strategy_net.BUILTIN and options[-1] == "nothing"
    assert strategy_games.builtin_index( "build", options, build ) == 0
    assert strategy_net.answer_of( options[0] ) is None
    _, _, tokens = strategy_net.query_tokens( "build", build, CONTEXT, [] )
    assert len( tokens ) == 3 and tokens[0][strategy_net.STRAT_FEATURES - 1] == 1.0


def test_real_hero_battles_from_the_ai_log():
    import strategy_games

    events = [
        {"ev": "battle_start", "bid": 1, "t": 3, "att": {"hero": {}, "c": "B"}, "def": {"hero": {}, "c": "R"}},
        {"ev": "battle_end", "bid": 1, "t": 3, "winner": "B", "att0": 1000, "att1": 600, "def0": 800, "def1": 0},
        {"ev": "battle_start", "bid": 2, "t": 4, "att": {"hero": {}, "c": "B"}, "def": {"c": "N"}},  # neutrals
        {"ev": "battle_end", "bid": 2, "t": 4, "winner": "B", "att0": 1, "att1": 1, "def0": 1, "def1": 0},
        {"ev": "battle_start", "bid": 3, "t": 20, "att": {"hero": {}, "c": "R"}, "def": {"hero": {}, "c": "B"}},  # too late
        {"ev": "battle_end", "bid": 3, "t": 20, "winner": "R", "att0": 1, "att1": 1, "def0": 1, "def1": 0},
    ]
    ( score, ) = strategy_games.real_hero_battles( events, "Blue", 1, 10 )
    assert abs( score - ( 1.0 + 0.6 - 0.0 ) ) < 1e-9


def test_duel_score_plays_both_sides_and_handles_missing_heroes():
    import strategy_games

    class FakeDuelEnv:
        def __init__( self ):
            self.calls = []

        def new_battle( self, **kwargs ):
            self.calls.append( ( kwargs["hero_att"][0], kwargs["hero_def"][0] ) )
            self.seeds = getattr( self, "seeds", [] ) + [kwargs["seed"]]
            return {"ev": "state", "result": "att",
                    "units": [{"u": 1, "side": "att", "str": 100, "q": 1}, {"u": 2, "side": "def", "str": 0, "q": 0}]}

    env = FakeDuelEnv()
    results = {"Blue": {"top": {"hid": 5, "hero": "aa", "str": 900}},
               "Red": {"top": {"hid": 7, "hero": "bb", "str": 800}}, "Green": {"top": {"hid": 9, "hero": "cc", "str": 50}}}
    score = strategy_games.duel_score( env, results, "Blue", 1 )
    assert sorted( set( env.seeds ) ) == list( range( 1, 1 + strategy_games.DUEL_SEEDS ) )
    assert env.calls == [( 5, 7 ), ( 7, 5 )] * strategy_games.DUEL_SEEDS  # the strongest rival (Red), both sides, every seed
    assert score == 0.0  # the attacker wins either way: +win once, -loss once
    assert strategy_games.duel_score( env, {"Blue": {}, "Red": results["Red"]}, "Blue", 1 ) == -2.0
    assert strategy_games.duel_score( env, {"Blue": results["Blue"]}, "Blue", 1 ) == 2.0


def test_smoothed_nll_ignores_padding_options():
    import torch.nn.functional as F

    import train_strategy_net

    logits = torch.tensor( [[2.0, 0.0, -1e9], [1.0, 1.0, 1.0]] )  # the first query has 2 real options
    log_probs = F.log_softmax( logits, dim=1 )
    plain = train_strategy_net.smoothed_nll( log_probs, [0, 1], 0.0 )
    assert abs( float( plain ) - float( F.nll_loss( log_probs, torch.tensor( [0, 1] ) ) ) ) < 1e-6
    smooth = train_strategy_net.smoothed_nll( log_probs, [0, 1], 0.5 )
    assert float( smooth ) < 1e6  # padding (-1e9) is not part of the uniform term


def test_attach_history_orders_by_game_and_query():
    records = [dict( kind="army", event=dict( ARMY, t=day ), context=dict( CONTEXT, t=day ), target=2, map="m", seed=1, n=n )
               for n, day in ( ( 5, 2 ), ( 1, 1 ), ( 9, 3 ) )]
    strategy_net.attach_history( records )
    by_n = {r["n"]: r["history"] for r in records}
    assert [len( by_n[n]["decisions"] ) for n in ( 1, 5, 9 )] == [0, 1, 2]
    assert [len( by_n[n]["days"] ) for n in ( 1, 5, 9 )] == [0, 1, 2]
    prefix, _, _ = strategy_net.query_tokens( "army", records[2]["event"], records[2]["context"], [], by_n[9] )
    # Time order: each previous day (kingdom, hero, castle) followed by its answer, then today's snapshot.
    assert [token_type( t ) for t in prefix] == ["day", "hero", "castle", "decision"] * 2 + ["hero", "castle"]
    assert [t[strategy_net.DAY_SLOT] * 30 for t in prefix] == pytest.approx( [1, 1, 1, 1, 2, 2, 2, 2, 3, 3] )


def test_history_keeps_the_newest_days_that_fit_the_window():
    days = [dict( CONTEXT, t=day ) for day in range( 1, 41 )]
    decisions = [{"kind": "army", "event": dict( ARMY, t=day ), "context": days[day - 1], "answer": 1} for day in range( 1, 41 )]
    history = {"days": days, "decisions": decisions}
    tokens = strategy_net.history_tokens( history, [] )
    assert len( tokens ) == 40 * 4  # everything fits: day + hero + castle + decision per day
    cut = strategy_net.history_tokens( history, [], budget=41 )
    assert len( cut ) == 40 and cut[-1][strategy_net.DAY_SLOT] * 30 == pytest.approx( 40 )  # the 10 newest whole days
    assert token_type( cut[0] ) == "day" and cut[0][strategy_net.DAY_SLOT] * 30 == pytest.approx( 31 )
    prefix, _, options = strategy_net.query_tokens( "army", dict( ARMY, t=41 ), dict( CONTEXT, t=41 ), [], history )
    assert len( prefix ) + 1 + 2 * len( options ) <= strategy_net.MAX_STRATEGIC_TOKENS


def test_own_heroes_and_castles_as_the_player_sees_them():
    hero = {"id": 7, "i": 25, "mp": 600, "mmp": 1200, "str": 400.0, "race": 2, "lvl": 5, "a": 3, "d": 2, "pw": 1, "k": 1, "sp": 10, "msp": 10,
            "mor": 1, "luck": 0, "book": 1, "army": [[13, 5, 11.7, 2, 2, 1, 0], [17, 3, 40.8, 4, 4, 0, 1]], "sk": [2, 0, 1] + [0] * 11, "art": [33, 41]}
    castle = {"i": 55, "race": 2, "castle": 1, "b": 0b101, "army": [[12, 10, 2.9, 1, 4, 0, 0]],
              "dw": [[12, 12, 2.86, 1, 4, 0, 0], [0, 0, 0, 0, 0, 0, 0]]}
    context = dict( CONTEXT, w=10, wd=1, wk=2, heroes=[hero], castles=[castle],
                    rcastles=[{"c": "Red", "i": 99, "race": 4, "castle": 0, "vis": 1, "army": [[30, 0, 5.0, 1, 5, 0, 0]], "est": 5}] )
    hero_token, castle_token, rcastle_token = strategy_net.snapshot_tokens( TARGET, context )
    assert [token_type( t ) for t in ( hero_token, castle_token, rcastle_token )] == ["hero", "castle", "rcastle"]
    assert hero_token[2] == 1.0 and hero_token[3] == 0.5 and hero_token[12] == 1.0  # query hero, level 5, spell book
    assert hero_token[14 + 1] == 1.0  # race BARB
    assert hero_token[21] == pytest.approx( 2 / 3 ) and hero_token[23] == pytest.approx( 1 / 3 )  # pathfinding 2, logistics 1
    stacks = hero_token[35:60]
    assert stacks[0] == pytest.approx( math.log1p( 5 ) ) and stacks[3] == 1.0 and stacks[9] == 1.0 and stacks[10:] == [0.0] * 15
    assert castle_token[9] == 1.0 and castle_token[10] == 0.0 and castle_token[11] == 1.0  # building bits 0 and 2
    assert castle_token[41] == pytest.approx( math.log1p( 12 ) ) and castle_token[43] == 0.0  # dwelling 1 has 12, dwelling 2 none
    assert rcastle_token[0] == 0.0 and rcastle_token[9 + 1] == 1.0  # an enemy castle, defenders seen as types
    context_token = strategy_net.query_tokens( "target", TARGET, context, [] )[1]
    assert context_token[6] == 1.0 and context_token[13] == 0.5  # Monday of week 2
    assert all( t[strategy_net.DAY_SLOT] == pytest.approx( 0.1 ) for t in ( hero_token, castle_token, rcastle_token, context_token ) )


def test_checkpoints_with_another_strategic_width_load_with_a_fresh_head( tmp_path ):
    model = AzBattleTransformer()
    state = model.state_dict()
    state["strat_proj.weight"] = torch.zeros( state["strat_proj.weight"].shape[0], STRAT_FEATURE_W - 2 )
    path = tmp_path / "old.pt"
    torch.save( {"arch": "transformer", "config": {}, "state_dict": state}, str( path ) )
    loaded = load_checkpoint( str( path ) )
    assert loaded.strat_proj.weight.shape[1] == STRAT_FEATURE_W


def test_history_prefix_changes_the_scores_and_padding_does_not():
    model = AzBattleTransformer().eval()
    history = {"days": [dict( CONTEXT, t=1 ), dict( CONTEXT, t=2 )], "decisions": []}
    plain = strategy_net.query_tokens( "target", TARGET, CONTEXT, [] )
    rich = strategy_net.query_tokens( "target", TARGET, CONTEXT, [], history )
    with torch.no_grad():
        a, _ = strategic_logits( model, [plain] )
        b, _ = strategic_logits( model, [rich] )
        batch, _ = strategic_logits( model, [rich, plain] )
    assert not torch.allclose( a, b )
    assert torch.allclose( batch[0], b[0], atol=1e-5 ) and torch.allclose( batch[1], a[0], atol=1e-5 )


def growing_game_queries( days: int = 20 ):
    """Queries of one game whose history grows day by day (several prefix chunks)."""
    history = {"days": [], "decisions": []}
    queries = []
    for day in range( 1, days + 1 ):
        context = dict( CONTEXT, t=day, res=[5, 0, 5, 0, 0, 0, 1000 * day] )
        for kind, event in ( ( "target", dict( TARGET, t=day ) ), ( "army", dict( ARMY, t=day ) ) ):
            queries.append( strategy_net.query_tokens( kind, event, context, [138], history, 512 ) )
            history["decisions"].append( {"kind": kind, "event": event, "context": context, "answer": day % 3} )
        history["days"].append( context )
    return queries


def test_prefix_cache_matches_the_plain_forward_and_is_deterministic():
    from transformer_model import STRAT_KV_CHUNK, StrategicPrefixCache, strategic_logits_cached

    torch.manual_seed( 0 )
    model = AzBattleTransformer().eval()
    queries = growing_game_queries()
    assert len( queries[-1][0] ) > 4 * STRAT_KV_CHUNK
    warm = StrategicPrefixCache()
    with torch.no_grad():
        for query in queries:
            plain, _ = strategic_logits( model, [query] )
            cached = strategic_logits_cached( model, query, warm )
            assert torch.allclose( cached, plain[0][:len( query[2] )], atol=1e-5 )
            # A cold cache gives bit-identical logits: the result never depends on the cache state.
            assert torch.equal( strategic_logits_cached( model, query, StrategicPrefixCache() ), cached )
    assert warm.hits > warm.misses  # the queries of a game share their history


def test_net_policy_caches_answers_by_their_input( tmp_path ):
    path = tmp_path / "m.pt"
    save_checkpoint( AzBattleTransformer(), str( path ) )
    cached = strategy_net.NetStrategyPolicy( str( path ) )
    plain = strategy_net.NetStrategyPolicy( str( path ), cache=False )
    for _ in range( 2 ):  # the same game twice: the second time every answer comes from the cache
        for policy in ( cached, plain ):
            policy.reset()
        for day in range( 1, 6 ):
            for policy in ( cached, plain ):
                policy.observe_turn( dict( CONTEXT, t=day ) )
            for kind, event in ( ( "target", dict( TARGET, t=day ) ), ( "army", dict( ARMY, t=day ) ) ):
                a, b = cached.probabilities( kind, event ), plain.probabilities( kind, event )
                assert max( abs( x - y ) for x, y in zip( a, b ) ) < 1e-5
                for policy in ( cached, plain ):
                    policy.record( kind, event, 0 )
    assert cached.answer_hits == 10


def test_net_policy_keeps_the_game_history( tmp_path ):
    path = tmp_path / "m.pt"
    save_checkpoint( AzBattleTransformer(), str( path ) )
    policy = strategy_net.NetStrategyPolicy( str( path ) )
    policy.observe_turn( dict( CONTEXT, t=1 ) )
    policy.army( dict( ARMY, t=1 ) )
    policy.observe_turn( dict( CONTEXT, t=2 ) )
    history = policy.history( "Blue" )
    assert len( history["days"] ) == 1 and len( history["decisions"] ) == 1
    policy.reset()
    assert policy.history( "Blue" ) == {"days": [], "decisions": []}


def test_branch_scores_use_the_day_report_and_game_end_per_horizon():
    import types

    import strategy_games

    calls = []

    def fake_duel( env, state, color, seed ):
        calls.append( state["Blue"]["top"]["str"] )
        return float( state["Blue"]["top"]["str"] )

    original = strategy_games.duel_score
    strategy_games.duel_score = fake_duel
    try:
        args = types.SimpleNamespace( horizons="7,14", label="duel", color="Blue" )
        week = {"Blue": {"c": "Blue", "top": {"str": 1}}}
        end = {"Blue": {"c": "Blue", "top": {"str": 2}}}
        played = ( {}, end, [], {3 + 7 + 1: week} )  # the query on day 3: report of day 11, game_end at day 17
        assert strategy_games.branch_scores( args, None, played, 3, 1 ) == [1.0, 2.0]
        assert calls == [1, 2]
    finally:
        strategy_games.duel_score = original


def test_branch_scores_fall_back_to_game_end_when_the_game_ended_early():
    import types

    import strategy_games

    original = strategy_games.duel_score
    strategy_games.duel_score = lambda env, state, color, seed: float( state["Blue"]["top"]["str"] )
    try:
        args = types.SimpleNamespace( horizons="7,14", label="duel", color="Blue" )
        end = {"Blue": {"c": "Blue", "top": {"str": 5}}}
        assert strategy_games.branch_scores( args, None, ( {}, end, [], {} ), 3, 1 ) == [5.0, 5.0]
    finally:
        strategy_games.duel_score = original


def test_rival_tokens_carry_only_what_the_player_sees():
    context = dict( CONTEXT, w=10, heroes=[{"id": 7, "i": 0, "mp": 900, "mmp": 1200, "str": 400.0}], castles=[{"i": 99}],
                    rivals=[{"c": "Red", "i": 23, "full": 0, "army": [[13, 5]], "est": 134},
                            {"c": "Red", "i": 55, "full": 1, "army": [[13, 7]], "est": 190, "a": 3, "lvl": 4}] )
    tokens = strategy_net.rival_tokens( TARGET, context )
    assert len( tokens ) == 2
    hidden, shown = tokens
    assert hidden[1] == 0.0 and shown[1] == 1.0
    assert hidden[3] == 3 / 50.0  # tile 23 = (3, 2) vs the query hero at (0, 0): Chebyshev 3
    assert hidden[6:14] == [0.0] * 8 and shown[6] == 0.4 and shown[7] == 0.3  # skills only with full information
    assert all( len( t ) == STRAT_TOKEN_W and token_type( t ) == "rival" for t in tokens )
    assert hidden[STRAT_FEATURE_W:] == [14.0, 0.0, 0.0, 0.0, 0.0]  # the creature is seen even through the size word


def test_army_creatures_share_the_battle_creature_embedding():
    """Every army (own heroes, rivals, garrisons) carries its slots' creature ids, embedded by the
    battle's mon_embed: the creature changes the strategic scores."""
    context = dict( CONTEXT, heroes=[dict( CONTEXT["heroes"][0], army=[[39, 6, 4.1, 1, 3, 1, 0], [3, 20, 22.3, 2, 4, 1, 0]] )],
                    castles=[{"i": 100, "army": [[5, 10, 8.0, 2, 3, 0, 0]]}] )
    prefix, context_token, options = strategy_net.query_tokens( "target", TARGET, context, [138] )
    hero, castle = prefix
    assert hero[STRAT_FEATURE_W:] == [40.0, 4.0, 0.0, 0.0, 0.0] and castle[STRAT_FEATURE_W:] == [6.0, 0.0, 0.0, 0.0, 0.0]
    assert context_token[STRAT_FEATURE_W:] == [0.0] * 5 and all( o[STRAT_FEATURE_W:] == [0.0] * 5 for o in options )
    # The same creature in a battle gets the same row: one encoding (enc.monster_token) for both.
    import encoding as enc

    battle = {"turn": 1, "cur": 1, "obstacles": [], "units": [{"u": 1, "side": "att", "mon": 39, "q": 6, "hpl": 5, "i": 12, "ti": -1, "sp": 3, "shots": 0,
                                                                   "moved": 0}]}
    assert enc.battle_tokens( battle )[12][enc.MON_COL] == hero[STRAT_FEATURE_W] == enc.monster_token( 39 )

    torch.manual_seed( 0 )
    model = AzBattleTransformer().eval()
    other = [hero[:STRAT_FEATURE_W] + [41.0] + hero[STRAT_FEATURE_W + 1:], castle]  # another creature in slot 1
    with torch.no_grad():
        scores, _ = strategic_logits( model, [( prefix, context_token, options ), ( other, context_token, options )] )
    assert not torch.allclose( scores[0], scores[1] )


def test_war_score_rules():
    import strategy_games

    class Env:
        def __init__( self ):
            self.fights = []

        def new_battle( self, **kwargs ):
            self.fights.append( kwargs["hero_def"][0] if kwargs["hero_att"][0] == 5 else kwargs["hero_att"][0] )
            return {"ev": "state", "result": "att",
                    "units": [{"u": 1, "side": "att", "str": 100, "q": 1}, {"u": 2, "side": "def", "str": 0, "q": 0}]}

    ours = {"hid": 5, "hero": "aa", "str": 900}
    state = {"Blue": {"s": "2", "str": 900, "top": ours},
             "Red": {"s": "2", "str": 3000, "top": {"hid": 7, "hero": "bb", "str": 800}},
             "Green": {"s": "2", "str": 100, "top": {"hid": 9, "hero": "cc", "str": 50}},
             "Yellow": {"s": "1", "str": 0}}  # lost: not a rival any more
    assert strategy_games.war_score( None, dict( state, Blue={"s": "0"} ), [], "Blue", 1, 22, 1 ) == 2.0
    assert strategy_games.war_score( None, dict( state, Blue={"s": "1"} ), [], "Blue", 1, 22, 1 ) == -2.0
    # A battle against the strongest rival (Red, by total strength) in the window decides.
    events = [{"ev": "battle_start", "bid": 1, "t": 5, "att": {"hero": {}, "c": "B"}, "def": {"hero": {}, "c": "R"}},
              {"ev": "battle_end", "bid": 1, "t": 5, "winner": "R", "att0": 100, "att1": 0, "def0": 100, "def1": 50}]
    assert strategy_games.war_score( None, state, events, "Blue", 1, 22, 1 ) == -1.5
    # No battle against the strongest rival: duels against every active rival instead.
    env = Env()
    strategy_games.war_score( env, state, [], "Blue", 1, 22, 1 )
    assert sorted( set( env.fights ) ) == [7, 9]


def test_length_batches_group_similar_histories_and_keep_every_record():
    import random

    import train_strategy_net

    records = [{"id": i, "history": {"days": [{}] * ( i % 40 ), "decisions": [{}] * i}} for i in range( 100 )]
    batches = train_strategy_net.length_batches( records, 8, random.Random( 0 ) )
    assert sorted( r["id"] for b in batches for r in b ) == list( range( 100 ) )
    lengths = [[train_strategy_net.history_length( r ) for r in b] for b in batches]
    assert max( max( l ) - min( l ) for l in lengths ) < 60  # a random batch would span ~0..330
    assert [b[0]["id"] for b in batches] != sorted( b[0]["id"] for b in batches )  # batch order is shuffled


class HandicapDuelEnv:
    """A duel server whose battles our hero wins when log2(our army / theirs) + the seed's luck > the
    true balance point (the luck spreads the seeds around it): a known answer for final_duel_label."""

    def __init__( self, balance: float ):
        self.balance = balance
        self.battles = 0
        self.ours = None

    def new_battle( self, seed, attacker, defender, hero_att, hero_def, att_scale=100, def_scale=100, **_ ):
        import math

        self.battles += 1
        our_scale, their_scale = ( att_scale, def_scale ) if hero_att[0] == 5 else ( def_scale, att_scale )
        luck = ( seed % 3 - 1 ) * 0.4 + ( 0.2 if hero_att[0] == 5 else -0.2 )  # every battle within -0.6 .. +0.6
        won = math.log2( our_scale / their_scale ) + luck > self.balance
        winner = "att" if ( hero_att[0] == 5 ) == won else "def"
        return {"ev": "state", "result": winner, "units": []}


def test_final_duel_label_searches_the_even_army_ratio():
    import strategy_games

    ours, theirs = {"hid": 5, "hero": "aa"}, {"hid": 7, "hero": "bb"}
    assert strategy_games.final_duel_label( HandicapDuelEnv( -0.9 ), ours, theirs, 1 ) == 1.0  # every battle won
    assert strategy_games.final_duel_label( HandicapDuelEnv( 0.9 ), ours, theirs, 1 ) == -1.0
    for balance in ( -0.3, 0.05, 0.4 ):
        env = HandicapDuelEnv( balance )
        label = strategy_games.final_duel_label( env, ours, theirs, 1 )
        # Even = half the battles won, which holds within +-0.2 of the balance (the luck steps), plus the
        # bisection precision.
        assert abs( label - ( -balance ) ) <= 0.27, ( balance, label )
        assert label * -balance > 0 or abs( balance ) < 0.2
        assert env.battles <= 2 * strategy_games.DUEL_SEEDS * ( 1 + strategy_games.FINAL_DUEL_STEPS )


def test_final_label_rules():
    import strategy_games

    top = {"hid": 5, "hero": "aa", "str": 100}
    results = {"Blue": {"s": "2", "top": top}, "Red": {"s": "2", "str": 900, "top": {"hid": 7, "hero": "bb"}}}
    assert strategy_games.final_label( None, dict( results, Blue={"s": "0"} ), "Blue", 1 ) == 1.0
    assert strategy_games.final_label( None, dict( results, Blue={"s": "1"} ), "Blue", 1 ) == -1.0
    assert strategy_games.final_label( None, dict( results, Blue={"s": "2"} ), "Blue", 1 ) == -1.0  # no hero left
    assert strategy_games.final_label( HandicapDuelEnv( -0.9 ), results, "Blue", 1 ) == 1.0


def test_query_scores_average_the_answer_differences_over_the_luck_replays( monkeypatch ):
    import types

    import strategy_games

    class Branch:
        def __init__( self, policy, color, pick_at=None, answer_index=None, expected=None ):
            self.answer, self.diverged = answer_index, False

    replays = []

    def fake_play( args, seed, until, branch, report_days, reseed ):
        replays.append( ( branch.answer, reseed ) )
        salt = reseed[1] if reseed else 0
        if branch.answer == 2 and salt == 1:
            raise TimeoutError( "stuck" )
        return ( branch.answer, salt )

    def fake_scores( args, duel_env, played, first_day, seed ):
        answer, salt = played
        # Luck shifts both answers alike (cancels in the paired difference); answer 1 is 0.5 better.
        return [salt * 0.3 + ( 0.5 if answer == 1 else 0.0 ) + ( -0.2 if answer == 2 else 0.0 )]

    monkeypatch.setattr( strategy_games, "PolicyBranch", Branch )
    monkeypatch.setattr( strategy_games, "play", fake_play )
    monkeypatch.setattr( strategy_games, "branch_scores", fake_scores )
    args = types.SimpleNamespace( color="Blue", salts=3 )
    query = {"n": 4, "event": {"t": 6}}
    baselines: dict = {}
    scores, salt_scores, salt_parts = strategy_games.query_scores( args, None, 1, query, [1, 2], None, baselines, 45, [] )
    assert salt_parts == {}
    assert scores == pytest.approx( {1: 0.5, 2: -0.2} )
    assert salt_scores["1"] == pytest.approx( [0.5, 0.5, 0.5] ) and len( salt_scores["2"] ) == 2  # the stuck replay is dropped
    assert sorted( baselines ) == [( 6, 0 ), ( 6, 1 ), ( 6, 2 )]  # one baseline per luck, shared by the answers
    assert ( None, None ) in replays and ( None, ( 7, 1 ) ) in replays  # salt 0 plain, salt k from the next day
    assert sum( 1 for answer, _ in replays if answer is None ) == 3


class HeroDuelEnv( HandicapDuelEnv ):
    """HandicapDuelEnv whose balance point depends on which version of our hero (hid 5) fights:
    a stronger hero has a lower balance (it needs less army)."""

    def __init__( self, balances: dict ):
        super().__init__( 0.0 )
        self.balances = balances

    def new_battle( self, seed, attacker, defender, hero_att, hero_def, att_scale=100, def_scale=100, **kwargs ):
        ours = hero_att if hero_att[0] == 5 else hero_def
        self.balance = self.balances[ours[1]]
        return super().new_battle( seed, attacker, defender, hero_att, hero_def, att_scale, def_scale, **kwargs )


def test_hero_equivalent_finds_the_army_factor_of_half_the_wins():
    import strategy_games

    ours, reference = {"hid": 5, "hero": "aa"}, {"hid": 7, "hero": "rr"}
    for balance in ( -1.5, -0.3, 0.4, 2.0 ):
        value = strategy_games.hero_equivalent( HeroDuelEnv( {"aa": balance} ), ours, reference, 1 )
        assert abs( value - balance ) <= 0.3, ( balance, value )  # +-0.2 luck steps + the bisection resolution
    # Out of range: clipped near the ends.
    assert strategy_games.hero_equivalent( HeroDuelEnv( {"aa": 9.0} ), ours, reference, 1 ) > 2.9
    assert strategy_games.hero_equivalent( HeroDuelEnv( {"aa": -9.0} ), ours, reference, 1 ) < -2.9


def test_hero_label_parts_measure_both_versions_against_the_baseline_reference():
    import math

    import strategy_games

    env = HeroDuelEnv( {"base": 0.5, "better": -0.5} )
    rival = {"s": "2", "str": 900, "top": {"hid": 7, "hero": "rr", "str": 800}}
    baseline = {"Blue": {"s": "2", "top": {"hid": 5, "hero": "base", "str": 100}}, "Red": rival}
    branch = {"Blue": {"s": "2", "top": {"hid": 5, "hero": "better", "str": 200}},
              "Red": {"s": "2", "str": 5, "top": {"hid": 9, "hero": "other"}}}  # the branch's own rival is ignored
    base = strategy_games.hero_state( env, baseline, "Blue", 1 )
    assert base["ref"] == rival["top"] and abs( base["m"] - 0.5 ) <= 0.3 and base["str"] == 100
    hero, army = strategy_games.hero_components( env, base, branch, "Blue", 1 )
    assert abs( hero - 1.0 ) <= 0.4 and army == pytest.approx( math.log2( 201 / 101 ) )
    # Our hero gone in the branch: the weakest equivalent and the army floor.
    hero, army = strategy_games.hero_components( env, base, {"Blue": {"s": "1"}}, "Blue", 1 )
    assert hero == pytest.approx( base["m"] - strategy_games.HERO_LOG2_RANGE ) and army == -strategy_games.HERO_LOG2_RANGE  # clipped
    # A won game is the strongest end of the scale; a rival without heroes cannot fight (strongest too).
    won = strategy_games.hero_state( env, {"Blue": dict( baseline["Blue"], s="0" ), "Red": rival}, "Blue", 1 )
    assert won["m"] == -strategy_games.HERO_LOG2_RANGE
    lonely = strategy_games.hero_state( env, {"Blue": baseline["Blue"], "Red": {"s": "2", "str": 0}}, "Blue", 1 )
    assert lonely["ref"] is None and lonely["m"] == -strategy_games.HERO_LOG2_RANGE
    # ... and a branch of it that did not win is measured against its own rival's hero.
    hero, _ = strategy_games.hero_components( env, lonely, dict( branch, Red=rival ), "Blue", 1 )
    assert abs( hero - ( -strategy_games.HERO_LOG2_RANGE - ( -0.5 ) ) ) <= 0.3
    assert strategy_games.combine_hero_parts( None, 0.5, "hero" ) == 0.5  # no hero part: the army
    assert strategy_games.combine_hero_parts( 1.0, 0.5, "mean" ) == 0.75
    assert strategy_games.combine_hero_parts( 1.0, 0.5, "army" ) == 0.5


def test_query_scores_record_both_hero_label_parts( monkeypatch ):
    import types

    import strategy_games

    class Branch:
        def __init__( self, policy, color, pick_at=None, answer_index=None, expected=None ):
            self.answer, self.diverged = answer_index, False

    def fake_play( args, seed, until, branch, report_days, reseed ):
        return None, {"answer": branch.answer, "salt": reseed[1] if reseed else 0}, [], {}

    def fake_state( duel_env, results, color, seed ):
        return {"salt": results["salt"]}

    def fake_components( duel_env, base, results, color, seed ):
        assert base["salt"] == results["salt"]  # the baseline of the same luck
        return ( 0.4, -0.2 ) if results["answer"] == 1 else ( -0.1, 0.3 )

    monkeypatch.setattr( strategy_games, "PolicyBranch", Branch )
    monkeypatch.setattr( strategy_games, "play", fake_play )
    monkeypatch.setattr( strategy_games, "hero_state", fake_state )
    monkeypatch.setattr( strategy_games, "hero_components", fake_components )
    args = types.SimpleNamespace( color="Blue", salts=2, label="hero", hero_rule="hero" )
    scores, salt_scores, parts = strategy_games.query_scores( args, None, 1, {"n": 4, "event": {"t": 6}}, [1, 2], None, {}, 45, [] )
    assert scores == pytest.approx( {1: 0.4, 2: -0.1} )
    assert parts["1"] == {"hero": [0.4, 0.4], "army": [-0.2, -0.2]}
    # 1 beats 2 in the hero part but loses in the army part: "agree" rejects the pair.
    parts["0"] = {"hero": [0.0, 0.0], "army": [0.0, 0.0]}
    assert not strategy_games.parts_agree( parts, 1, 2 )
    assert strategy_games.parts_agree( {"1": {"hero": [0.4], "army": [0.1]}, "2": {"hero": [None], "army": [0.0]}}, 1, 2 )


def test_reliable_pairs_keep_only_clear_per_luck_gaps():
    import train_strategy_net

    def pair( chosen_scores, rejected_scores ):
        return {"chosen": 1, "rejected": 0, "salt_scores": {"1": chosen_scores, "0": rejected_scores}}

    clear = pair( [0.5, 0.6, 0.4, 0.5], [0.0] * 4 )
    one_lucky_replay = pair( [2.0, 0.0, 0.0, 0.0], [0.0] * 4 )  # the mean 0.5 rests on one replay
    single = {"chosen": 1, "rejected": 0, "scores": {"1": 1.0}}  # no per-luck scores
    kept = train_strategy_net.reliable_pairs( [clear, one_lucky_replay, single], 2.0 )
    assert kept == [clear]


def test_policy_branch_records_every_player_for_the_value_data():
    import strategy_games

    class Net:
        def reset( self ):
            pass

        def observe_turn( self, ctx ):
            pass

        def history( self, color ):
            return {"days": [], "decisions": []}

        def decide( self, kind, event ):
            return 2

        def record( self, kind, event, index ):
            pass

    branch = strategy_games.PolicyBranch( Net(), "Blue" )
    branch.observe_turn( {"p": "Blue", "t": 1} )
    branch.observe_turn( {"p": "Green", "t": 1} )
    army = {"ev": "army", "t": 1, "castle": 100, "offer": [{"mon": 1, "avail": 10, "n": 10, "str": 11.0}]}
    branch.army( dict( army, p="Blue" ) )
    branch.army( dict( army, p="Green" ) )
    blue, green = branch.trajectory( "Blue" ), branch.trajectory( "Green" )
    assert [d["answer"] for d in blue["decisions"]] == [2] and blue["decisions"][0]["ctx"] == 0
    options = strategy_games.query_options( "army", army )
    assert [d["answer"] for d in green["decisions"]] == [options.index( 100 )]  # the built-in AI's answer
    assert len( blue["days"] ) == len( green["days"] ) == 1
