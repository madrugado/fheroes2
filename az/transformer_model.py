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
NUM_DIRECTIONS = 7  # 6 hex directions + ranged
DIR_INDEX_RANGED = NUM_DIRECTIONS - 1

_CLS_ID = NUM_CELL_TOKENS  # 100: global token
_ACT_ID = NUM_CELL_TOKENS + 1  # 101: policy query token


def _dir_subindex(direction_flag: int | None) -> int:
    """Maps an engine direction flag (None means ranged) to the direction sub-index."""
    if direction_flag is None:
        return DIR_INDEX_RANGED
    return enc._DIR_FLAGS.index(direction_flag)


def decompose_action(act: int, args: list[int], unit_cells: dict[int, int] | None = None):
    """Splits an engine command into (kind, cell, dir_subindex) or None.

    kind: "move" | "attack" | "skip". For attacks, cell is the TARGET cell and dir_subindex
    encodes the hex direction (or ranged). Cell/dir are resolved like encoding.action_index.
    """
    if act == 0 and len(args) >= 2:
        if not (0 <= args[1] < enc.NUM_CELLS):
            return None
        return "move", args[1], None
    if act == 1 and len(args) >= 5:
        _, target_uid, move_cell, target_cell, direction = args[:5]
        if target_cell < 0 and unit_cells is not None:
            target_cell = unit_cells.get(target_uid)
        if target_cell is None or not (0 <= target_cell < enc.NUM_CELLS):
            return None
        if move_cell is not None and move_cell >= enc.NUM_CELLS:
            # Defensive: the engine may emit cells outside the v0 board bounds.
            move_cell = -1
        if direction in enc._DIR_FLAGS:
            return "attack", target_cell, _dir_subindex(direction)
        if direction <= 0:
            if move_cell >= 0:
                derived = enc.direction_between(move_cell, target_cell)
                if derived is not None:
                    return "attack", target_cell, _dir_subindex(derived)
            return "attack", target_cell, DIR_INDEX_RANGED
        return None
    if act == 8:
        return "skip", None, None
    return None


def _clone_cache(past):
    """Builds a fresh DynamicCache from the prefill cache without mutating the original:
    HF appends to the cache it receives, so each decode call needs its own copy."""
    from transformers import DynamicCache

    cache = DynamicCache()
    for layer_idx, layer in enumerate(past.layers):
        cache.update(layer.keys.clone(), layer.values.clone(), layer_idx)
    return cache


class AzBattleTransformer(nn.Module):
    def __init__(self, d_model: int = D_MODEL, n_layer: int = N_LAYER, n_head: int = N_HEAD):
        super().__init__()

        config = Qwen3Config(
            vocab_size=1,  # unused: all inputs go through inputs_embeds
            hidden_size=d_model,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            num_key_value_heads=max(n_head // 2, 1),  # grouped-query attention
            head_dim=d_model // n_head,
            intermediate_size=4 * d_model,
            max_position_embeddings=enc.NUM_CELLS + 3,
            attention_bias=False,
            mlp_bias=False,
            attention_dropout=0.0,
        )
        self.body: Qwen3Model = Qwen3Model(config)

        self.cell_proj = nn.Linear(enc.NUM_PLANES, d_model)
        self.special_embed = nn.Embedding(2, d_model)  # CLS, ACTION
        self.cell_id_embed = nn.Embedding(NUM_CELL_TOKENS, d_model)  # decode-step cell identity

        self.value_head = nn.Linear(d_model, 1)
        self.cell_head = nn.Linear(d_model, NUM_CELL_TOKENS)
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

        target: {"kind": "move"|"attack"|"skip", "cell": int|None, "dir": int|None}.
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

        Returns (cell_logits (B, 100), (rows, dir_logits (K, 7)) | None, value (B,)).
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

        cell_probs = torch.softmax(self.cell_head(act_hidden).squeeze(0), dim=0)  # (100,)
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

        priors = {}
        for i, parts in decomposed:
            if parts is None:
                continue
            kind, cell, dir_sub = parts
            if kind == "move":
                priors[i] = float(cell_probs[cell])
            elif kind == "skip":
                priors[i] = skip_prob
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
