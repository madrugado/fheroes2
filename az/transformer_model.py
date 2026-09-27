"""Transformer policy/value model for battles (HuggingFace Qwen3 body).

Body: ready-made Qwen3Model in a tiny configuration (RMSNorm, SwiGLU, RoPE, GQA —
no learned position table, so sequences cannot overflow it the way GPT2's wpe could).

Tokenization: one token per board cell (99), plus a global [CLS] token (value) and an
[ACTION] query token (policy). Per-cell inputs are continuous feature vectors (the 11
encoding planes flattened per cell), projected to the model dimension.

Action decoding is autoregressive in two steps, which is where the KV-cache pays off:
  1. prefill: [cells..., CLS, ACTION] -> cell logits (target cell) + value (from CLS);
  2. for each distinct legal target cell, a one-token cell-identity embedding is appended
     and the transformer is re-run with `use_cache=True`, reusing the cached keys/values —
     the direction distribution for that cell is read from the new token.

The model never sees discrete vocabulary tokens: everything goes through inputs_embeds.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Qwen3Config, Qwen3Model

import encoding as enc

D_MODEL = 128
N_LAYER = 4
N_HEAD = 4

NUM_CELL_TOKENS = 100  # 99 board cells + one "skip" pseudo-cell (index 99)
SKIP_CELL = enc.NUM_CELLS  # 99
# The first decoding step also chooses between the hero's spells: tokens [100, 173) = spell id
# (all legal targets of a spell share its token and split its probability).
SPELL_TOKEN_BASE = NUM_CELL_TOKENS
NUM_POLICY_TOKENS = NUM_CELL_TOKENS + enc.NUM_SPELLS
NUM_DIRECTIONS = enc.ATTACK_SLOTS  # 6 head-cell + 6 tail-cell strike directions + ranged
DIR_INDEX_RANGED = enc.RANGED_DIR

_CLS_ID = NUM_CELL_TOKENS  # 100: global token
_ACT_ID = NUM_CELL_TOKENS + 1  # 101: policy query token


def _dir_subindex(direction_flag: int | None) -> int:
    """Maps an engine direction flag (None means ranged) to the direction sub-index."""
    if direction_flag is None:
        return DIR_INDEX_RANGED
    return enc._DIR_FLAGS.index(direction_flag)


def decompose_action(act: int, args: list[int], unit_cells: dict[int, int] | None = None):
    """Splits an engine command into (kind, cell, dir_subindex) or None.

    kind: "move" | "attack" | "skip" | "spell". For attacks, cell is the TARGET cell and dir_subindex
    encodes the hex direction (or ranged); for spells, cell is the spell token (SPELL_TOKEN_BASE +
    spell id). Cell/dir are resolved like encoding.action_index;
    `args` are in the engine wire order (see encoding.ctor_args).
    """
    args = enc.ctor_args(args)
    if act == 0 and len(args) >= 2:
        if not (0 <= args[1] < enc.NUM_CELLS):
            return None
        return "move", args[1], None
    if act == 1 and len(args) >= 5:
        # The same resolution as the ResNet's action slots (encoding.attack_parts): the direction
        # sub-index also tells a wide attacker's head and tail strikes apart.
        parts = enc.attack_parts(args, unit_cells)
        if parts is None:
            return None
        return "attack", parts[0], parts[1]
    if act == 8:
        return "skip", None, None
    if act == enc.SPELLCAST and len(args) >= 1:
        if not (0 < args[0] < enc.NUM_SPELLS):
            return None
        return "spell", SPELL_TOKEN_BASE + args[0], None
    return None


def _clone_cache(past):
    """Builds a fresh DynamicCache from the prefill cache without mutating the original:
    HF appends to the cache it receives, so each decode call needs its own copy."""
    from transformers import DynamicCache

    cache = DynamicCache()
    for layer_idx, layer in enumerate(past.layers):
        cache.update(layer.keys.clone(), layer.values.clone(), layer_idx)
    return cache


# Model sizes. The body is a plain Qwen3 stack fed through inputs_embeds (no vocabulary), so the
# parameter count is the transformer layers plus the small input/output heads.
#   small — the prototype (~1.0M parameters);
#   50m   — hidden 512, 12 layers, 8 query / 4 KV heads of 64, SwiGLU 2048: ~47M parameters,
#           window 2048; the largest size that trains in reasonable time on the M1 Pro laptop
#           (the 0.5b model needs ~4 h per epoch with Adafactor and 0.28 s per evaluation);
#   0.5b  — Qwen3-0.6B's layer shape (hidden 1024, 16 query / 8 KV heads of 128, SwiGLU 3072)
#           with 32 layers: ~0.50B parameters. Context window 2048 (a battle state is 102 tokens;
#           the headroom is for longer inputs such as state histories).
PRESETS: dict[str, dict] = {
    "small": {"d_model": D_MODEL, "n_layer": N_LAYER, "n_head": N_HEAD, "n_kv_head": N_HEAD // 2,
              "head_dim": D_MODEL // N_HEAD, "ffn": 4 * D_MODEL, "window": enc.NUM_CELLS + 3},
    "50m": {"d_model": 512, "n_layer": 12, "n_head": 8, "n_kv_head": 4, "head_dim": 64, "ffn": 2048,
            "window": 2048},
    "0.5b": {"d_model": 1024, "n_layer": 32, "n_head": 16, "n_kv_head": 8, "head_dim": 128, "ffn": 3072,
             "window": 2048},
}


class AzBattleTransformer(nn.Module):
    def __init__(self, size: str = "small", **overrides):
        super().__init__()

        # Stored in checkpoints (see save_checkpoint()) so that the loader rebuilds the same shape.
        self.config = dict(PRESETS[size], **overrides)
        cfg = self.config
        d_model = cfg["d_model"]

        config = Qwen3Config(
            vocab_size=1,  # unused: all inputs go through inputs_embeds
            hidden_size=d_model,
            num_hidden_layers=cfg["n_layer"],
            num_attention_heads=cfg["n_head"],
            num_key_value_heads=cfg["n_kv_head"],  # grouped-query attention
            head_dim=cfg["head_dim"],  # explicit: Qwen3Config defaults to 128, not d_model / heads
            intermediate_size=cfg["ffn"],
            max_position_embeddings=cfg["window"],
            attention_bias=False,
            mlp_bias=False,
            attention_dropout=0.0,
        )
        self.body: Qwen3Model = Qwen3Model(config)

        self.cell_proj = nn.Linear(enc.NUM_PLANES, d_model)
        self.special_embed = nn.Embedding(2, d_model)  # CLS, ACTION
        self.cell_id_embed = nn.Embedding(NUM_CELL_TOKENS, d_model)  # decode-step cell identity

        self.value_head = nn.Linear(d_model, 1)
        self.cell_head = nn.Linear(d_model, NUM_POLICY_TOKENS)
        self.dir_head = nn.Linear(d_model, NUM_DIRECTIONS)

    @staticmethod
    def cell_tokens(state: dict) -> torch.Tensor:
        """(1, NUM_CELLS, NUM_PLANES) per-cell features from the state encoding planes."""
        planes = enc.state_planes(state)  # (P, H, W)
        t = torch.tensor(planes, dtype=torch.float32)
        return t.permute(1, 2, 0).reshape(1, enc.NUM_CELLS, enc.NUM_PLANES)

    def _embed_sequence(self, cell_tokens: torch.Tensor, decode_cell: int | None = None):
        """Builds the input embedding sequence: [cells..., CLS, ACTION] (+ optional decode token)."""
        batch = cell_tokens.shape[0]
        cell_embeds = self.cell_proj(cell_tokens)
        # The special-token embedding table has two rows: 0 = CLS, 1 = ACTION.
        specials = self.special_embed(torch.tensor([0, 1], device=cell_tokens.device))
        specials = specials.unsqueeze(0).expand(batch, -1, -1)

        parts = [cell_embeds, specials]
        if decode_cell is not None:
            tok = self.cell_id_embed(torch.tensor([decode_cell], device=cell_tokens.device))
            parts.append(tok.unsqueeze(0))

        return torch.cat(parts, dim=1)

    def forward_train(self, state: dict, target: dict):
        """Teacher-forced training pass.

        target: {"kind": "move"|"attack"|"skip"|"spell", "cell": int|None, "dir": int|None}.
        Returns (cell_logits, dir_logits_or_None, value); the caller computes the losses.
        """
        cell_tokens = self.cell_tokens(state)

        kind = target["kind"]
        decode_cell = target["cell"] if (kind == "attack" and target["cell"] is not None) else None

        inputs = self._embed_sequence(cell_tokens, decode_cell)
        out = self.body(inputs_embeds=inputs)

        hidden = out.last_hidden_state
        act_pos = enc.NUM_CELLS + 1  # the ACTION token
        cls_pos = enc.NUM_CELLS  # the CLS token

        cell_logits = self.cell_head(hidden[:, act_pos, :])
        value = torch.tanh(self.value_head(hidden[:, cls_pos, :])).squeeze(1)

        if kind == "skip" or decode_cell is None:
            return cell_logits, None, value

        dir_logits = self.dir_head(hidden[:, -1, :])  # read from the decode token position
        return cell_logits, dir_logits, value

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def forward_batch(self, states: list[dict], decode_cells: list[int | None]):
        """Batched teacher-forced prefill. decode_cells marks the attack rows whose direction
        decode reuses the prefill KV-cache, mirroring inference.

        Returns (cell_logits (B, NUM_POLICY_TOKENS), (rows, dir_logits (K, 7)) | None, value (B,)).
        """
        device = self._device()
        cell_tokens = torch.cat([self.cell_tokens(s) for s in states]).to(device)  # (B, 99, P)
        inputs = self._embed_sequence(cell_tokens)

        out = self.body(inputs_embeds=inputs, use_cache=True)
        hidden = out.last_hidden_state
        past = out.past_key_values

        cell_logits = self.cell_head(hidden[:, enc.NUM_CELLS + 1, :])
        value = torch.tanh(self.value_head(hidden[:, enc.NUM_CELLS, :])).squeeze(1)

        rows = [i for i, c in enumerate(decode_cells) if c is not None]
        if not rows:
            return cell_logits, None, value

        decode_embeds = self.cell_id_embed(torch.tensor([decode_cells[i] for i in rows], device=cell_tokens.device))
        past_subset = past.batch_select_indices(torch.tensor(rows, device=cell_tokens.device))

        step = self.body(inputs_embeds=decode_embeds.unsqueeze(1), past_key_values=past_subset, use_cache=False)
        dir_logits = self.dir_head(step.last_hidden_state[:, -1, :])

        return cell_logits, (rows, dir_logits), value

    @torch.no_grad()
    def evaluate(self, state: dict):
        """MCTS interface: returns (slot -> prior for the legal moves, value in [-1, 1]).

        The prefill runs once with use_cache=True; the direction distribution for each
        distinct legal target cell is then decoded incrementally, reusing the KV-cache.
        """
        was_training = self.training
        self.eval()

        cell_tokens = self.cell_tokens(state).to(self._device())

        inputs = self._embed_sequence(cell_tokens)
        out = self.body(inputs_embeds=inputs, use_cache=True)
        hidden = out.last_hidden_state
        past = out.past_key_values

        act_hidden = hidden[:, enc.NUM_CELLS + 1, :]
        cls_hidden = hidden[:, enc.NUM_CELLS, :]

        cell_probs = torch.softmax(self.cell_head(act_hidden).squeeze(0), dim=0)  # (NUM_POLICY_TOKENS,)
        value = float(torch.tanh(self.value_head(cls_hidden).squeeze(0)))

        unit_cells = enc.unit_cells_map(state["units"])

        # Decompose all legal moves; collect the distinct attack cells for cached decoding.
        decomposed = []
        attack_cells = set()
        for i, move in enumerate(state["legal"]):
            act, args = (move["act"], move["args"]) if isinstance(move, dict) else (move[0], move[1])
            parts = decompose_action(act, list(args), unit_cells)
            decomposed.append((i, parts))
            if parts is not None and parts[0] == "attack":
                attack_cells.add(parts[1])

        # One incremental decode per distinct attack cell (KV-cache makes this cheap). Each
        # decode gets its own clone of the prefill cache: HF appends to the cache it receives
        # even with use_cache=False, so a shared cache would advance the decode position.
        dir_probs: dict[int, torch.Tensor] = {}
        for cell in attack_cells:
            decode_tok = self.cell_id_embed(torch.tensor([cell], device=cell_tokens.device)).unsqueeze(0)  # (1, 1, D)
            step = self.body(inputs_embeds=decode_tok, past_key_values=_clone_cache(past), use_cache=False)
            dir_probs[cell] = torch.softmax(self.dir_head(step.last_hidden_state[:, -1, :]).squeeze(0), dim=0)

        skip_prob = float(cell_probs[SKIP_CELL])
        spell_moves: dict[int, int] = {}
        for _, parts in decomposed:
            if parts is not None and parts[0] == "spell":
                spell_moves[parts[1]] = spell_moves.get(parts[1], 0) + 1

        priors = {}
        for i, parts in decomposed:
            if parts is None:
                continue
            kind, cell, dir_sub = parts
            if kind == "move":
                priors[i] = float(cell_probs[cell])
            elif kind == "skip":
                priors[i] = skip_prob
            elif kind == "spell":
                priors[i] = float(cell_probs[cell]) / spell_moves[cell]
            else:
                cell_prob = float(cell_probs[cell])
                dir_prob = float(dir_probs[cell][dir_sub]) if cell in dir_probs else 0.0
                priors[i] = cell_prob * dir_prob

        total = sum(priors.values())
        if total > 0:
            priors = {k: v / total for k, v in priors.items()}

        if was_training:
            self.train()
        return priors, value


def save_checkpoint(model: AzBattleTransformer, path: str) -> None:
    """Weights + the model shape (a 0.5b checkpoint cannot be loaded into the default shape)."""
    torch.save({"arch": "transformer", "config": model.config, "state_dict": model.state_dict()}, path)


def load_checkpoint(path: str, device: str = "cpu") -> AzBattleTransformer:
    """Loads save_checkpoint() output; a bare state dict (older checkpoints) is the small model."""
    data = torch.load(path, map_location=device)
    if isinstance(data, dict) and "state_dict" in data:
        model = AzBattleTransformer("small", **data["config"])  # the stored config overrides every field
        model.load_state_dict(data["state_dict"])
    else:
        model = AzBattleTransformer()
        model.load_state_dict(data)
    return model.to(device)
