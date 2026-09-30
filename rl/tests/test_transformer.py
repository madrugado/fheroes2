"""Unit tests for rl/transformer_model.py (torch + transformers required)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import encoding as enc  # noqa: E402
import transformer_model as tfm  # noqa: E402
from transformer_model import AzBattleTransformer  # noqa: E402


def make_state():
    return {
        "turn": 3,
        "cur": 1,
        "obstacles": [40],
        "units": [
            {"u": 1, "side": "att", "mon": 13, "q": 30, "hpl": 10, "i": 0, "ti": -1, "sp": 2, "shots": 8, "moved": 0},
            {"u": 2, "side": "def", "mon": 22, "q": 20, "hpl": 20, "i": 6, "ti": -1, "sp": 2, "shots": 0, "moved": 0},
        ],
        "legal": [
            {"act": 0, "args": [1, 1]},
            {"act": 1, "args": [0, -1, -1, 2, 1]},  # ranged shot at unit 2
            {"act": 8, "args": [1]},
        ],
    }


def test_kv_cache_matches_full_forward():
    """The decode step with a KV-cache must produce the same hidden state as a full forward
    over the sequence with the decode token appended."""
    model = AzBattleTransformer()
    model.eval()

    state = make_state()
    cell_tokens = model.cell_tokens(state)

    # Full forward with the decode token appended (cell identity 5).
    full = model.body(inputs_embeds=model._embed_sequence(cell_tokens, decode_cell=5))
    full_hidden = full.last_hidden_state[:, -1, :]

    # Prefill + one cached decode step.
    prefill = model.body(inputs_embeds=model._embed_sequence(cell_tokens), use_cache=True)
    decode_tok = model.cell_id_embed(torch.tensor([5])).unsqueeze(0)  # (1, 1, D)
    step = model.body(inputs_embeds=decode_tok, past_key_values=prefill.past_key_values, use_cache=True)
    cached_hidden = step.last_hidden_state[:, -1, :]

    assert torch.allclose(full_hidden, cached_hidden, atol=1e-4)


def test_evaluate_interface():
    model = AzBattleTransformer()
    model.eval()

    state = make_state()
    priors, value = model.evaluate(state)

    assert len(priors) == len(state["legal"])
    assert all(i in priors for i in range(len(state["legal"])))
    assert all(p >= 0 for p in priors.values())
    assert abs(sum(priors.values()) - 1.0) < 1e-4
    assert -1.0 <= value <= 1.0

    # The ranged shot (index 1) must get a non-zero prior (the net saw the unit map resolve it).
    assert priors[1] > 0.0


def test_forward_train_shapes():
    model = AzBattleTransformer()
    model.train()

    state = make_state()

    # Attack target: both cell and direction losses apply.
    cell_logits, dir_logits, value = model.forward_train(state, {"kind": "attack", "cell": 6, "dir": 1})
    assert cell_logits.shape == (1, tfm.NUM_POLICY_TOKENS)
    assert dir_logits.shape == (1, tfm.NUM_DIRECTIONS)
    assert value.shape == (1,)

    # Skip target: no direction loss.
    cell_logits, dir_logits, value = model.forward_train(state, {"kind": "skip", "cell": None, "dir": None})
    assert dir_logits is None


def test_overfit_single_batch():
    """The cell head must learn the target cell of one fixed state within a few steps."""
    model = AzBattleTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    state = make_state()
    target = {"kind": "attack", "cell": 6, "dir": 1}

    first_loss = None
    for _ in range(60):
        cell_logits, dir_logits, value = model.forward_train(state, target)
        loss = torch.nn.functional.cross_entropy(cell_logits, torch.tensor([6]))
        if dir_logits is not None:
            loss = loss + torch.nn.functional.cross_entropy(dir_logits, torch.tensor([1]))
        if first_loss is None:
            first_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert loss.item() < first_loss * 0.5


def test_decompose_action():
    unit_cells = {2: 6}

    assert tfm.decompose_action(0, [12, 1]) == ("move", 12, None)
    assert tfm.decompose_action(8, [1]) == ("skip", None, None)
    assert tfm.decompose_action(1, [0, -1, -1, 2, 1], unit_cells) == ("attack", 6, tfm.DIR_INDEX_RANGED)

    # Melee with an explicit direction flag.
    parts = tfm.decompose_action(1, [4, 7, -1, 2, 1], unit_cells)
    assert parts is not None and parts[1] == 7 and parts[2] == enc._DIR_FLAGS.index(4)


def test_decompose_action_rejects_garbage():
    unit_cells = {2: 6}

    # MOVE outside the board.
    assert tfm.decompose_action(0, [200, 1]) is None
    # ATTACK with an impossible direction flag.
    assert tfm.decompose_action(1, [64, 40, -1, 2, 1], unit_cells) is None
    # ATTACK whose target unit is not on the board.
    assert tfm.decompose_action(1, [0, -1, -1, 9, 1], unit_cells) is None
    # ATTACK with an unresolvable target and no unit map.
    assert tfm.decompose_action(1, [0, -1, -1, 9, 1]) is None
    # Unknown command type.
    assert tfm.decompose_action(5, [1]) is None
    # SPELLCAST without a valid spell id.
    assert tfm.decompose_action(2, [40, 0]) is None


def test_decompose_spellcast_and_evaluate_splits_the_spell_token():
    assert tfm.decompose_action(2, [40, 3]) == ("spell", tfm.SPELL_TOKEN_BASE + 3, None)
    assert tfm.decompose_action(2, [60, 40, 5]) == ("spell", tfm.SPELL_TOKEN_BASE + 5, None)

    model = AzBattleTransformer()
    model.eval()
    state = make_state()
    state["legal"] = state["legal"] + [{"act": 2, "args": [6, 1]}, {"act": 2, "args": [7, 1]}]
    priors, _ = model.evaluate(state)
    assert priors[3] > 0.0 and abs(priors[3] - priors[4]) < 1e-9
    assert abs(sum(priors.values()) - 1.0) < 1e-4


def test_decompose_action_derives_melee_direction():
    # Defender at cell 6, attacker moves from cell 7 (adjacent LEFT): direction is derived.
    assert tfm.decompose_action(1, [-1, -1, 7, 2, 1], {2: 6}) == ("attack", 6, enc._DIR_FLAGS.index(32))


def test_dir_subindex():
    assert tfm._dir_subindex(None) == tfm.DIR_INDEX_RANGED
    assert tfm._dir_subindex(4) == enc._DIR_FLAGS.index(4)


def test_forward_batch():
    model = AzBattleTransformer()
    model.train()
    state = make_state()

    # Mixed batch: row 0 attacks (direction decode), row 1 skips (no decode).
    cell_logits, dir_out, value = model.forward_batch([state, state], [6, None])
    assert cell_logits.shape == (2, tfm.NUM_POLICY_TOKENS)
    assert value.shape == (2,)
    rows, dir_logits = dir_out
    assert rows == [0]
    assert dir_logits.shape == (1, tfm.NUM_DIRECTIONS)

    # All rows decode directions.
    _, dir_out, _ = model.forward_batch([state, state], [6, 6])
    rows, dir_logits = dir_out
    assert rows == [0, 1]
    assert dir_logits.shape == (2, tfm.NUM_DIRECTIONS)

    # No attack rows at all -> no direction decode.
    _, dir_out, _ = model.forward_batch([state, state], [None, None])
    assert dir_out is None


def test_decompose_action_defensive_move_cell():
    # The engine may emit move cells outside the v0 board bounds; they must not crash.
    assert tfm.decompose_action(1, [0, 6, 500, 2, 1], {2: 6}) == ("attack", 6, tfm.DIR_INDEX_RANGED)


def test_evaluate_ignores_unmappable_moves():
    model = AzBattleTransformer()
    model.eval()

    state = make_state()
    state["legal"] = state["legal"] + [{"act": 2, "args": [0]}]  # SPELLCAST of Spell::NONE: unmappable

    priors, _ = model.evaluate(state)

    assert 3 not in priors
    assert abs(sum(priors.values()) - 1.0) < 1e-4


def test_evaluate_skip_only_and_mode_restore():
    model = AzBattleTransformer()
    model.train()  # evaluate must put the model in eval mode and restore training afterwards

    state = make_state()
    state["legal"] = [{"act": 8, "args": [1]}]

    priors, value = model.evaluate(state)

    assert abs(priors[0] - 1.0) < 1e-4
    assert -1.0 <= value <= 1.0
    assert model.training


def test_forward_batch_direction_decode_matches_the_full_forward():
    """The batched teacher-forced decode must see the board (regression: batch_select_indices
    returns None in transformers 5.x, and the decode silently ran without the cache)."""
    model = AzBattleTransformer()
    model.eval()
    states = [make_state(), make_state()]
    with torch.no_grad():
        _, (rows, dir_logits), _ = model.forward_batch(states, [None, 6])
        full = model.body(inputs_embeds=model._embed_sequence(model.cell_tokens(states[1]), decode_cell=6))
        reference = model.dir_head(full.last_hidden_state[:, -1, :])
    assert rows == [1]
    assert torch.allclose(dir_logits, reference, atol=1e-4)


def test_the_battle_output_sees_the_creature_the_health_and_the_hero_mana():
    torch.manual_seed(0)
    model = AzBattleTransformer()
    model.eval()
    base = dict(make_state(), heroes=[{"side": "att", "sp": 10, "cast": 0}])

    def outputs(state):
        with torch.no_grad():
            cell_logits, _, value = model.forward_batch([state], [None])
        return cell_logits, value

    reference = outputs(base)
    other_creature = dict(base, units=[dict(base["units"][0], mon=60)] + base["units"][1:])
    more_health = dict(base, units=[dict(base["units"][0], hpl=180)] + base["units"][1:])
    more_mana = dict(base, heroes=[{"side": "att", "sp": 40, "cast": 0}])
    enemy_hero = dict(base, heroes=base["heroes"] + [{"side": "def", "sp": 5, "cast": 0}])
    for changed in (other_creature, more_health, more_mana, enemy_hero):
        logits, value = outputs(changed)
        assert not torch.allclose(logits, reference[0]) and not torch.allclose(value, reference[1])


def test_a_checkpoint_without_the_unit_and_hero_inputs_loads(tmp_path):
    model = AzBattleTransformer()
    state = {key: value for key, value in model.state_dict().items() if not key.startswith(("unit_proj.", "mon_embed.", "hero_proj."))}
    path = tmp_path / "old.pt"
    torch.save({"arch": "transformer", "config": model.config, "state_dict": state}, path)
    loaded = tfm.load_checkpoint(str(path))
    assert torch.equal(loaded.cell_proj.weight, model.cell_proj.weight)
    loaded.evaluate(make_state())
