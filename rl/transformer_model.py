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

import math

import torch
import torch.nn as nn
from transformers import Qwen3Config, Qwen3Model

import encoding as enc

D_MODEL = 128
N_LAYER = 4
N_HEAD = 4

NUM_CELL_TOKENS = 100  # 99 board cells + one "skip" pseudo-cell (index 99)
SKIP_CELL = enc.NUM_CELLS  # 99
# Battle input sequence: [history..., 99 cells, attacker's hero, defender's hero, CLS, ACTION].
BATTLE_TOKENS = enc.NUM_CELLS + 2
# The first decoding step also chooses between the hero's spells: tokens [100, 173) = spell id
# (all legal targets of a spell share its token and split its probability).
SPELL_TOKEN_BASE = NUM_CELL_TOKENS
NUM_POLICY_TOKENS = NUM_CELL_TOKENS + enc.NUM_SPELLS
NUM_DIRECTIONS = enc.ATTACK_SLOTS  # 6 head-cell + 6 tail-cell strike directions + ranged
DIR_INDEX_RANGED = enc.RANGED_DIR

# Battle history (user request 2026-09-30): the actions of this battle so far, one token each, in
# front of the board (state["history"], battle_history_entry). Rows of the battle input are
# BATTLE_ROW_W wide: the board columns of enc.battle_tokens, then the history columns, then the row
# kind. A batch is left-padded with ROW_PAD rows (masked out, positions count real tokens only).
MAX_BATTLE_HISTORY = 256  # the longest expert battle has 191 decisions
HIST_KINDS = ("move", "attack", "skip", "spell")
HIST_FEATURES = 2 + len(HIST_KINDS) + NUM_DIRECTIONS + 2  # own/enemy, kind, strike direction, round, stack size
HIST_ACTION_COL = enc.BATTLE_TOKEN_W  # first-step token of the action: target cell / skip / spell
HIST_MON_COL = HIST_ACTION_COL + 1  # the acting creature (enc.monster_token)
HIST_COLS = slice(HIST_MON_COL + 1, HIST_MON_COL + 1 + HIST_FEATURES)
ROW_KIND_COL = HIST_MON_COL + 1 + HIST_FEATURES
BATTLE_ROW_W = ROW_KIND_COL + 1
ROW_PAD, ROW_HISTORY, ROW_CELL, ROW_HERO = 0, 1, 2, 3

# Strategic queries (rl/strategy_net.py) share the body: history tokens (previous days with what the
# player saw and the decisions it made), today's heroes/castles/visible enemies, one context token
# and one token per answer option; each
# a feature vector of this width (features padded, the query kind one-hot, the token type one-hot),
# then the creatures of an army's slots (creature id + 1, 0: none) for the battle's creature embedding.
STRAT_FEATURES = 64
STRAT_KINDS = ("target", "build", "hire", "army")
STRAT_TOKEN_TYPES = ("context", "option", "decision", "day", "hero", "rival", "castle", "rcastle")
STRAT_FEATURE_W = STRAT_FEATURES + len(STRAT_KINDS) + len(STRAT_TOKEN_TYPES)  # what strat_proj reads
STRAT_MON_SLOTS = 5  # army slots of a hero or a castle garrison
STRAT_TOKEN_W = STRAT_FEATURE_W + STRAT_MON_SLOTS

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


def battle_history_entry(state: dict, act: int, args: list[int]) -> dict | None:
    """One action of a battle as its history keeps it: who acted (side, creature, stack size), what it
    did (kind, first-step token — target cell, skip or spell — and strike direction) and in which
    round. None when the command cannot be decoded."""
    actor = next((unit for unit in state["units"] if unit["u"] == state.get("cur")), None)
    parts = decompose_action(act, list(args), enc.unit_cells_map(state["units"]))
    if actor is None or parts is None:
        return None
    kind, cell, direction = parts
    return {"side": actor["side"], "mon": actor["mon"], "q": actor["q"], "turn": state.get("turn", 0), "kind": kind,
            "token": SKIP_CELL if kind == "skip" else cell, "dir": direction}


def battle_rows(state: dict) -> list[list[float]]:
    """The battle input rows: the last MAX_BATTLE_HISTORY actions of state["history"], seen from the
    side to move (own / enemy), then the board (enc.battle_tokens: cells and the two commanders)."""
    mover = enc.side_to_move(state)
    rows = []
    for entry in (state.get("history") or [])[-MAX_BATTLE_HISTORY:]:
        row = [0.0] * BATTLE_ROW_W
        row[HIST_ACTION_COL] = float(entry["token"])
        row[HIST_MON_COL] = enc.monster_token(entry["mon"])
        own = entry["side"] == mover
        direction = [0.0] * NUM_DIRECTIONS
        if entry.get("dir") is not None:
            direction[entry["dir"]] = 1.0
        row[HIST_COLS] = ([float(own), float(not own)] + [float(entry["kind"] == kind) for kind in HIST_KINDS] + direction
                          + [entry.get("turn", 0) / 50.0, math.log2(entry.get("q", 0) + 1) / 10.0])
        row[ROW_KIND_COL] = float(ROW_HISTORY)
        rows.append(row)
    for index, token in enumerate(enc.battle_tokens(state)):
        rows.append(token + [0.0] * (BATTLE_ROW_W - len(token) - 1) + [float(ROW_CELL if index < enc.NUM_CELLS else ROW_HERO)])
    return rows


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
#           window 512 (user decision 2026-09-29; was 2048: the strategic history of a 45-day game
#           is < 490 tokens); the largest size that trains in reasonable time on the M1 Pro laptop
#           (the 0.5b model needs ~4 h per epoch with Adafactor and 0.28 s per evaluation);
#   0.5b  — Qwen3-0.6B's layer shape (hidden 1024, 16 query / 8 KV heads of 128, SwiGLU 3072)
#           with 32 layers: ~0.50B parameters. Context window 2048 (a battle state is 102 tokens;
#           the headroom is for longer inputs such as state histories).
PRESETS: dict[str, dict] = {
    "small": {"d_model": D_MODEL, "n_layer": N_LAYER, "n_head": N_HEAD, "n_kv_head": N_HEAD // 2,
              "head_dim": D_MODEL // N_HEAD, "ffn": 4 * D_MODEL, "window": BATTLE_TOKENS + 3},
    "50m": {"d_model": 512, "n_layer": 12, "n_head": 8, "n_kv_head": 4, "head_dim": 64, "ffn": 2048,
            "window": 512},
    "100m": {"d_model": 768, "n_layer": 12, "n_head": 12, "n_kv_head": 4, "head_dim": 64, "ffn": 3072,
             "window": 512},
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
        # What a player sees of every stack and commander (enc.battle_tokens): the creature, exact
        # count, health, speed, shots; per side the hero's spell points and whether it cast this round.
        self.unit_proj = nn.Linear(enc.NUM_UNIT_FEATURES, d_model)
        self.mon_embed = nn.Embedding(enc.NUM_MONSTER_IDS + 1, d_model, padding_idx=0)  # 0: no unit
        self.hero_proj = nn.Linear(enc.NUM_HERO_FEATURES, d_model)
        # History tokens: the action's first-step token, the acting creature, the rest of its features.
        self.hist_action_embed = nn.Embedding(NUM_POLICY_TOKENS, d_model)
        self.hist_proj = nn.Linear(HIST_FEATURES, d_model)
        self.special_embed = nn.Embedding(2, d_model)  # CLS, ACTION
        self.cell_id_embed = nn.Embedding(NUM_CELL_TOKENS, d_model)  # decode-step cell identity

        self.value_head = nn.Linear(d_model, 1)
        self.cell_head = nn.Linear(d_model, NUM_POLICY_TOKENS)
        self.dir_head = nn.Linear(d_model, NUM_DIRECTIONS)

        # Strategic decisions (the same body): token projection and a score per answer option.
        self.strat_proj = nn.Linear(STRAT_FEATURE_W, d_model)
        # The creatures of an army (heroes, garrisons): the battle's creature embedding through one
        # projection per slot, so a creature stays paired with the count of the same slot.
        self.strat_mon_slots = nn.ModuleList(nn.Linear(d_model, d_model, bias=False) for _ in range(STRAT_MON_SLOTS))
        self.strat_head = nn.Linear(d_model, 1)
        # Strategic value (rl/strategy_value.py): the player's expected end-of-game score from its
        # history and today's snapshot, read at the context token.
        self.strat_value_head = nn.Linear(d_model, 1)
        # Starts at 0, where tanh passes the full gradient: a fresh 100m run pushed the random head
        # into saturation (every state at +-2) within 4 steps and it never came back.
        nn.init.zeros_(self.strat_value_head.weight)
        nn.init.zeros_(self.strat_value_head.bias)

    @staticmethod
    def cell_tokens(state: dict) -> torch.Tensor:
        """(1, rows, BATTLE_ROW_W): the battle's history and board (battle_rows; _embed_sequence
        splits the columns by row kind)."""
        return torch.tensor(battle_rows(state), dtype=torch.float32).unsqueeze(0)

    @staticmethod
    def batch_rows(states: list[dict]) -> torch.Tensor:
        """(B, longest, BATTLE_ROW_W): battle_rows of every state, left-padded with ROW_PAD rows so
        that the board, CLS and ACTION end every row at the same positions."""
        rows = [AzBattleTransformer.cell_tokens(state)[0] for state in states]
        length = max(row.shape[0] for row in rows)
        batch = torch.zeros(len(rows), length, BATTLE_ROW_W)
        for index, row in enumerate(rows):
            batch[index, length - row.shape[0]:] = row
        return batch

    @staticmethod
    def _attention(tokens: torch.Tensor, extra: int):
        """Attention mask (padding rows out) and position ids (real tokens only) for the rows plus
        `extra` real tokens after them."""
        real = tokens[..., ROW_KIND_COL] != ROW_PAD
        mask = torch.cat([real, torch.ones(real.shape[0], extra, dtype=torch.bool, device=real.device)], dim=1).long()
        return mask, (mask.cumsum(1) - 1).clamp(min=0)

    def _embed_sequence(self, cell_tokens: torch.Tensor, decode_cell: int | None = None):
        """Builds the input embedding sequence: [history..., cells..., attacker's hero, defender's
        hero, CLS, ACTION] (+ optional decode token); padding rows embed to zero."""
        batch = cell_tokens.shape[0]
        kinds = cell_tokens[..., ROW_KIND_COL].long().unsqueeze(-1)
        cell_embeds = (self.cell_proj(cell_tokens[..., enc.PLANE_COLS])
                       + self.unit_proj(cell_tokens[..., enc.UNIT_COLS])
                       + self.mon_embed(cell_tokens[..., enc.MON_COL].long()))
        hero_embeds = self.hero_proj(cell_tokens[..., enc.HERO_COLS])
        history_embeds = (self.hist_action_embed(cell_tokens[..., HIST_ACTION_COL].long())
                          + self.mon_embed(cell_tokens[..., HIST_MON_COL].long())
                          + self.hist_proj(cell_tokens[..., HIST_COLS]))
        rows = torch.where(kinds == ROW_CELL, cell_embeds, torch.zeros_like(cell_embeds))
        rows = torch.where(kinds == ROW_HERO, hero_embeds, rows)
        rows = torch.where(kinds == ROW_HISTORY, history_embeds, rows)
        # The special-token embedding table has two rows: 0 = CLS, 1 = ACTION.
        specials = self.special_embed(torch.tensor([0, 1], device=cell_tokens.device))
        specials = specials.unsqueeze(0).expand(batch, -1, -1)

        parts = [rows, specials]
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
        cls_pos = cell_tokens.shape[1]
        cell_logits = self.cell_head(hidden[:, cls_pos + 1, :])
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
        cell_tokens = self.batch_rows(states).to(device)  # (B, rows, BATTLE_ROW_W), left-padded
        inputs = self._embed_sequence(cell_tokens)
        mask, positions = self._attention(cell_tokens, 2)

        out = self.body(inputs_embeds=inputs, attention_mask=mask, position_ids=positions, use_cache=True)
        hidden = out.last_hidden_state
        past = out.past_key_values

        cls_pos = cell_tokens.shape[1]
        cell_logits = self.cell_head(hidden[:, cls_pos + 1, :])
        value = torch.tanh(self.value_head(hidden[:, cls_pos, :])).squeeze(1)

        rows = [i for i, c in enumerate(decode_cells) if c is not None]
        if not rows:
            return cell_logits, None, value

        decode_embeds = self.cell_id_embed(torch.tensor([decode_cells[i] for i in rows], device=cell_tokens.device))
        # transformers 5.x: batch_select_indices() filters the cache IN PLACE and returns None. It
        # used to be assigned (`past_subset = past.batch_select_indices(...)`), so the direction
        # decode ran with no cache — without the board — during training while inference decoded
        # with it (the direction loss of the 50m run stalled near the label entropy).
        selected = torch.tensor(rows, device=cell_tokens.device)
        past.batch_select_indices(selected)
        step_mask = torch.cat([mask[selected], torch.ones(len(rows), 1, dtype=mask.dtype, device=mask.device)], dim=1)
        step_positions = mask[selected].sum(dim=1, keepdim=True)

        step = self.body(inputs_embeds=decode_embeds.unsqueeze(1), attention_mask=step_mask, position_ids=step_positions,
                         past_key_values=past, use_cache=False)
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

        act_hidden = hidden[:, cell_tokens.shape[1] + 1, :]
        cls_hidden = hidden[:, cell_tokens.shape[1], :]

        cell_logits = self.cell_head(act_hidden).squeeze(0)  # (NUM_POLICY_TOKENS,)
        value = float(torch.tanh(self.value_head(cls_hidden).squeeze(0)))

        unit_cells = enc.unit_cells_map(state["units"])

        # Decompose all legal moves; collect the distinct attack cells for cached decoding and the
        # legal options of both steps: the probabilities are softmaxes over the LEGAL options only
        # (the masked losses of train.py / train_dpo.py train exactly these).
        decomposed = []
        legal_dirs: dict[int, set[int]] = {}
        legal_tokens: set[int] = set()
        for i, move in enumerate(state["legal"]):
            act, args = (move["act"], move["args"]) if isinstance(move, dict) else (move[0], move[1])
            parts = decompose_action(act, list(args), unit_cells)
            decomposed.append((i, parts))
            if parts is None:
                continue
            legal_tokens.add(SKIP_CELL if parts[0] == "skip" else parts[1])
            if parts[0] == "attack":
                legal_dirs.setdefault(parts[1], set()).add(parts[2])

        token_mask = torch.zeros_like(cell_logits, dtype=torch.bool)
        token_mask[sorted(legal_tokens)] = True
        cell_probs = torch.softmax(cell_logits.masked_fill(~token_mask, -1e9), dim=0)

        # One incremental decode per distinct attack cell (KV-cache makes this cheap). Each
        # decode gets its own clone of the prefill cache: HF appends to the cache it receives
        # even with use_cache=False, so a shared cache would advance the decode position.
        dir_probs: dict[int, torch.Tensor] = {}
        for cell, dirs in legal_dirs.items():
            decode_tok = self.cell_id_embed(torch.tensor([cell], device=cell_tokens.device)).unsqueeze(0)  # (1, 1, D)
            step = self.body(inputs_embeds=decode_tok, past_key_values=_clone_cache(past), use_cache=False)
            dir_logits = self.dir_head(step.last_hidden_state[:, -1, :]).squeeze(0)
            dir_mask = torch.zeros_like(dir_logits, dtype=torch.bool)
            dir_mask[sorted(dirs)] = True
            dir_probs[cell] = torch.softmax(dir_logits.masked_fill(~dir_mask, -1e9), dim=0)

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


def _strategic_embeds(model: "AzBattleTransformer", tokens: torch.Tensor) -> torch.Tensor:
    """(B, L, STRAT_TOKEN_W) strategic tokens -> input embeddings: the features through strat_proj
    plus, per army slot, the slot's creature (the battle's mon_embed; 0 = none embeds to zero)."""
    embeds = model.strat_proj(tokens[..., :STRAT_FEATURE_W])
    for slot, projection in enumerate(model.strat_mon_slots):
        embeds = embeds + projection(model.mon_embed(tokens[..., STRAT_FEATURE_W + slot].long()))
    return embeds


def strategic_logits(model: "AzBattleTransformer", queries: list[tuple]):
    """Scores of the answer options of a batch of strategic queries.

    queries: (prefix tokens, context token, option tokens) per query (strategy_net.query_tokens;
    the prefix is the history: days, decisions, heroes — may be empty; a 2-tuple means no prefix).
    The body is causal, so the sequence is [prefix, context, options, options]: the scores are
    read from the SECOND copy of the options, where every option has seen the history, the
    context and all options. Returns (logits (B, max options), mask (B, max options) True for real
    options)."""
    device = next(model.parameters()).device
    queries = [q if len(q) == 3 else ([], q[0], q[1]) for q in queries]
    n_max = max(len(options) for _, _, options in queries)
    length = max(len(prefix) + 1 + 2 * len(options) for prefix, _, options in queries)
    tokens = torch.zeros(len(queries), length, STRAT_TOKEN_W, device=device)
    mask = torch.zeros(len(queries), n_max, dtype=torch.bool, device=device)
    index = torch.zeros(len(queries), n_max, dtype=torch.long, device=device)
    for row, (prefix, context, options) in enumerate(queries):
        n, start = len(options), len(prefix)
        if prefix:
            tokens[row, :start] = torch.tensor(prefix, dtype=torch.float32, device=device)
        tokens[row, start] = torch.tensor(context, device=device)
        block = torch.tensor(options, dtype=torch.float32, device=device)
        # Real tokens first, zero padding at the end (causal: padding never affects real tokens).
        tokens[row, start + 1:start + 1 + n] = block
        tokens[row, start + 1 + n:start + 1 + 2 * n] = block
        mask[row, :n] = True
        index[row, :n] = torch.arange(start + 1 + n, start + 1 + 2 * n, device=device)

    hidden = model.body(inputs_embeds=_strategic_embeds(model, tokens), use_cache=False).last_hidden_state
    picked = hidden.gather(1, index.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
    logits = model.strat_head(picked).squeeze(-1)
    return logits.masked_fill(~mask, -1e9), mask


STRAT_VALUE_SCALE = 1.0  # the final label (strategy_games.final_label) lies in [-1, 1]


def strategic_value(model: "AzBattleTransformer", states: list[tuple]):
    """Values of a batch of strategic states: (prefix tokens, context token) per state (the prefix
    is the player's history and today's snapshot, strategy_net.value_tokens). Causal body: the value
    is read at the context token, the last real token. Returns (B,) in [-1, 1]."""
    device = next(model.parameters()).device
    length = max(len(prefix) + 1 for prefix, _ in states)
    tokens = torch.zeros(len(states), length, STRAT_TOKEN_W, device=device)
    last = torch.zeros(len(states), dtype=torch.long, device=device)
    for row, (prefix, context) in enumerate(states):
        if prefix:
            tokens[row, :len(prefix)] = torch.tensor(prefix, dtype=torch.float32, device=device)
        tokens[row, len(prefix)] = torch.tensor(context, device=device)
        last[row] = len(prefix)
    hidden = model.body(inputs_embeds=_strategic_embeds(model, tokens), use_cache=False).last_hidden_state
    picked = hidden[torch.arange(len(states), device=device), last]
    return STRAT_VALUE_SCALE * torch.tanh(model.strat_value_head(picked).squeeze(-1))


def save_checkpoint(model: AzBattleTransformer, path: str) -> None:
    """Weights + the model shape (a 0.5b checkpoint cannot be loaded into the default shape)."""
    torch.save({"arch": "transformer", "config": model.config, "state_dict": model.state_dict()}, path)


def load_checkpoint(path: str, device: str = "cpu") -> AzBattleTransformer:
    """Loads save_checkpoint() output; a bare state dict (older checkpoints) is the small model."""
    data = torch.load(path, map_location=device)
    if isinstance(data, dict) and "state_dict" in data:
        model = AzBattleTransformer("small", **data["config"])  # the stored config overrides every field
        # Checkpoints from before the strategic head (or with another strategic token width): those
        # modules start fresh.
        own = model.state_dict()
        state = {key: value for key, value in data["state_dict"].items()
                 if not (key.startswith("strat_") and key in own and own[key].shape != value.shape)}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or any(not key.startswith(("strat_proj.", "strat_head.", "strat_value_head.", "unit_proj.", "mon_embed.", "hero_proj.", "strat_mon_slots.",
                                                                        "hist_action_embed.", "hist_proj.")) for key in missing):
            raise RuntimeError(f"checkpoint mismatch: missing {missing}, unexpected {unexpected}")
    else:
        model = AzBattleTransformer()
        model.load_state_dict(data, strict=False)
    return model.to(device)
