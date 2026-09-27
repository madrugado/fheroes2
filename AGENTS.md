# AGENTS.md

Working notes for AI coding agents (and humans) in this fork of **fheroes2** — an open-source
reimplementation of the Heroes of Might and Magic II engine (GPLv2+).

This fork carries a research workstream on top of upstream: **AI-vs-AI self-play and an
AlphaZero-style battle engine**. Keep upstream conventions; keep local work in its own commits.

## Build (macOS, Apple Silicon)

```sh
# Dependencies: cmake >= 3.24, sdl2, sdl2_mixer (brew install cmake sdl2_mixer sdl2)
cmake -B build-release -DCMAKE_BUILD_TYPE=Release   # use for autonomous runs / experiments
cmake --build build-release -j8
cmake -B build -DCMAKE_BUILD_TYPE=Debug             # Debug enables WITH_DEBUG (asserts + DEBUG_LOG)

# The CMake post-build step copies the binary to the repo root.
# ALWAYS run the game as ./fheroes2 from the repo root: the engine resolves game data
# relative to the executable location, so running build-release/fheroes2 directly fails
# with "No AGG data files found".
```

Game data: original HoMM2 assets are NOT in the repo. `bash script/demo/download_demo_version.sh`
installs the free demo (DATA/ + one .mp2 map). The demo lacks Price-of-Loyalty assets.

## Running AI-vs-AI games

Autonomous playtest mode (no UI, exits when done):

```sh
FHEROES2_AI_LOG=/tmp/fh2_ai.jsonl \
FHEROES2_AUTO_PLAYTEST=1 \
FHEROES2_AUTO_PLAYTEST_DAYS=7 \
FHEROES2_AUTO_PLAYTEST_MAP="Arena.mp2" \
./fheroes2
```

- `FHEROES2_AUTO_PLAYTEST` — value = number of playthroughs (all players become AI-controlled).
- `FHEROES2_AUTO_PLAYTEST_DAYS` — day limit per playthrough (default 365).
- `FHEROES2_AUTO_PLAYTEST_MAP` — file name from `maps/` (case-insensitive); default: first
  `.fh2m`/`.mp2` in alphabetical order. Both original `.mp2` and Resurrection `.fh2m` maps are
  supported in this fork.
- `FHEROES2_AUTO_PLAYTEST_SEED` — re-seeds the engine RNG (`Rand::SeedCurrentThread`) with
  `seed + playthrough id` before every playthrough: equal seeds (+ equal agent choices) replay
  byte-identical games (verified on the AI log). Unset = random games, as upstream.
- `FHEROES2_AI_LOG` — JSON-lines event log of the AI (see `AI_LLM_PROTOCOL.md`), written by
  `AILog` (`src/fheroes2/ai/ai_log.*`). Unset = disabled, zero cost.

Notes:
- Speed: the autonomous mode sets `AIMoveSpeed(0)` like upstream's interactive playtest does.
  Before 2026-09-27 it did not, and `AI::HeroesMove` slept in movement-animation delays ~95% of
  the time: a 7-day 2kings game took ~63 s, now 1.6 s (identical game log). If playtests get
  slow again, `sample <pid>` the engine and look for `SDL_Delay` under `HeroesMove`.
- Maps that work well for multi-player benchmarks: `Battlefi.mp2` (6 players), `Thechaos.mp2`
  (5), `2kings.mp2` (2, small), `Arena.mp2` (4, tiny). Some maps (e.g. Crystalemania,
  Champions Charge) produce no events at all in the autonomous mode — probe before using.
- Use **Release** for playtests: `.fh2m` maps may reference Price-of-Loyalty artifacts, which
  trigger an `assert(0)` in Debug builds (`maps_tiles_helper.cpp`) but are skipped in Release.
- Maps live in `maps/` (untracked data: `*.mp2`, `*.mx2` are git-ignored; `*.fh2m` from upstream
  are tracked). Do not commit map packs.

## Workstream layout

- Branch `feature/az-battle-engine` — AlphaZero-style battle engine research (all commits below
  are on top of upstream `master`, commit `96cf68495`):
  - `858f1fad2` AILog event stream + autonomous playtest mode + AGENTS.md
  - `5479006f` headless battle server + `az/` prototype skeleton
  - `5558e14d` strategic decision protocol (`AIDecision`)
  - `d27c37d6` batched replay op + 60s protocol watchdogs
  - `d12cfd475` transformer body GPT2 → Qwen3 + test-coverage pass (63 tests)
  - `48f2a26cd` battle-state snapshot/restore + `suggest` op + `az/gate.py` (see "Gate runner" below)
  - battle-agent channel for real battles + `az/battle_agent.py` (see "Battle-agent channel")
  - unified game agent + `tempo` strategic policy + channel-fallback fixes (see "Unified game agent")
  - C++ hardening: legal moves == engine validation, illegal commands answered with an error,
    `wseed` reset, robust stack parsing, SIGPIPE-safe agent channels (+ integration tests)
- `az/` — Python side of the research:
  - `engine_bridge.py` — battle environment client (`BattleEnv`), with 60s read watchdogs
  - `mcts.py` — PUCT search; node states materialize via battle-server snapshots when the
    engine supports them (fallback: batched replay on top of the main line)
  - `selfplay.py` — battle self-play runner, records `az/data/games.jsonl`
    (state, legal moves, MCTS visit counts, outcome) and verifies determinism per game
  - `gate.py` — win-rate runner: our MCTS (optional trained net) vs the built-in BattlePlanner
    (via `suggest`), sides alternate, determinism-checked per battle
  - `battle_agent.py` — external agent for real battles (random|planner|policy|mcts)
  - `game_agent.py` — one agent for both channels; `strategy_policies.py` — strategic policies
  - `strategy_env.py` / `strategy_run.py` — full-game strategic layer: policies
    `greedy|random|builtin`, records `az/data/strategy_<policy>.jsonl`
- C++ side:
  - `src/fheroes2/battle/battle_server.*` — headless battle server
  - `src/fheroes2/battle/battle_arena.*` — snapshot capture/apply + mid-round resume
  - `src/fheroes2/ai/ai_log.*` — JSONL event log
  - `src/fheroes2/ai/ai_decision.*` — strategic decision protocol
  - `src/fheroes2/battle/battle_agent.*` — battle-agent channel for real battles
  - `Arena::Turns()/UnitTurn()` action-provider overloads — the seam for external drivers
- `AI_LLM_PROTOCOL.md` — event-log schema (the observation infra is reused for training data).

## Battle-agent channel (real battles, committed 2026-09-27)

`FHEROES2_BATTLE_AGENT=1` (with `FHEROES2_AUTO_PLAYTEST`) — at every AI unit activation
`Arena::UnitTurn` asks an external agent for the action (hook in the AI branch; `battle_action`
log events carry `src:"agent"|"planner"`). Runner: `az/battle_agent.py --policy
random|planner|policy|mcts` (full wire format in `az/README.md`, "Real-battle integration").

- C++: `battle_agent.{h,cpp}`; `Battle::Loader` sends `battle_start` (seed, tile, wseed,
  `searchable`, stacks as `[slot,mon,count]` + spread formation) and `battle_end`. Agent gone =>
  permanent built-in fallback; an illegal action falls back for that decision only and reports
  `battle_fallback`. Agent replies: `action` / `planner` / `skip`.
- C++: `battle_server.{h,cpp}` — shared `SerializeArenaState`/`EnumerateLegalMoves`; the `new`
  op accepts army slots (`"0:13x30,2:21x25"`), formation flags `sat`/`sdf` and `wseed`
  (0 = keep the pinned default).
- MCTS mode searches in a headless replica rebuilt from `battle_start`; every agent move is
  mirrored and the state diffed (`turn/cur/units/obstacles`) — first mismatch degrades the rest
  of the battle to policy/planner.
- Gotchas: decision queries arrive as `"ev":"state"` WITH a `"bid"` field (don't wait for a
  `battle_state` event); the reader must be a byte-level line assembler (states exceed the pipe
  buffer, buffered readline + select() starve); the replica must load the same map
  (`BattleEnv(map_name=...)`) or obstacles mismatch at the root.
- Verified: `random` — full 7-day playtest, 466 decisions; `mcts --sims 4` — ~300 ms per
  searched decision, replica synced in monster-only battles. Release build, zero warnings.
- Agent death: with a channel enabled the engine ignores SIGPIPE (`prepareChannel`/
  `prepareDecisionChannel`), so an agent process that exits no longer kills the game (was exit
  code -13); the next read hits EOF, the channel breaks, the built-in AI finishes the game.
- Known limitations (documented, not fixed): hero battles are not exactly replicable (commander
  stats missing in the replica) — MCTS works until the first desync; sieges are never
  searchable (`searchable:0`); `clang-format` is not installed in this sandbox (style matched
  by hand).

## Unified game agent (committed 2026-09-27)

- `az/game_agent.py` — ONE process serves both channels (`AIDecision` hero targets +
  `BattleAgent`): `GameAgent` subclasses `BattleAgentRunner` and consumes strategic events in
  the `_handle_event` hook; `extra_env` adds `FHEROES2_STRATEGY_SERVER=1`. The battle-only runner
  pops a stray `FHEROES2_STRATEGY_SERVER` from the env (else the engine blocks on decisions
  nobody answers).
- `az/strategy_policies.py` — `greedy|random|builtin|tempo` (shared with `strategy_run.py`);
  context-aware policies implement `observe_turn(turn_context)` (both runners call it).
  `tempo` = gamma^(extra travel turns) x claim penalty for tiles another hero took this kingdom
  turn. Honest status: on 2kings week 1 all candidates are within one turn, `tempo` == `greedy`
  (0/25 deviations) — needs longer games / bigger maps to evaluate.
- C++ fixes found on the way: (1) in both `ai_decision.cpp` and `battle_agent.cpp`,
  `markChannelBroken()` and `isChannelBroken()` used DIFFERENT function-local statics, so the
  "permanent fallback" never engaged (every later query was still written to a dead agent) —
  now one shared `channelBroken` flag; (2) `game_end` was emitted only with the strategic
  channel — now also with the battle agent alone (`sendGameOver` no longer self-gates; the
  caller in `game_auto_playtest.cpp` checks both channels). Also `battleBegins` now checks the
  broken flag (it used to write `battle_start` to a dead agent).
- Verified: 2kings 7 days tempo+random — 87 strategic + 76 battle decisions, ~60 s, outcome on
  all records; battle-only runner now receives `game_end`.

## Strategic benchmark (committed 2026-09-27)

- `az/strategy_bench.py` — paired head-to-head: per seed one control game (all built-in) + one
  game per color where only that color uses the policy (`strategy_policies.ForColor`); verdict
  per (seed, color) by (outcome, castles, army strength) from `game_end`; exact sign test +
  bootstrap CI in the summary. Sanity: `--policy greedy` gives 100% `equal` with 0 overrides.
- Engine support: `FHEROES2_AUTO_PLAYTEST_SEED`; `"p":"<Color>"` in `turn_context`/`decision`;
  `game_end` results carry `k`/`h`/`str`/`g` (castles, heroes, army strength, gold).
  `ai_planner_hero.cpp` accepted agent picks only for `chosen > 0` — tile 0 was silently
  ignored; now `>= 0`.
- Result — `tempo` is NOT better than the built-in AI (kept only as a cheap baseline):
  2kings 14d: 1 better / 13 equal / 2 worse (p=1.0); Battlefi 30d: 16/29/15 (p=1.0, mean
  d_str +273, CI [-32, +638]); Thechaos 30d: 8/31/11 (p=0.65, CI [-145, +2314]). The built-in
  value already accounts for distance; tempo overrides rarely (reports in `az/data/bench_*.json`).
- `StrategyEnv` had the select()+buffered readline() deadlock (turn_context + decision in one
  chunk -> the decision sits in Python's buffer, the engine waits for the reply; shows up under
  load as a 60 s TimeoutError). It now uses `az/line_reader.py` (`LineReader`, shared with
  battle_agent.py). NEVER read engine pipes with select() + readline().

## Next (plan)

0. (NEW, blocks everything net-related) Regenerate expert data (`az/gen_expert.py`) and
   self-play data with the fixed encoding + legal moves, retrain ResNet/transformer, re-measure
   imitation accuracy and the gate win rate. Old checkpoints/data are invalid.
1. (done: see "Strategic benchmark" — tempo is no better than builtin.)
2. Strategic value network: train on `game_agent_*.jsonl` strategy records (candidate features
   + outcome), plug in as a new strategy policy.
3. Hero battles in the replica: send commander stats in `battle_start` so MCTS stays synced
   after the hero acts (needs a battle-server `new` extension).

## Gate runner (az/gate.py)

Plays our engine vs the built-in BattlePlanner: `az/.venv/bin/python az/gate.py --battles 40
--sims 32 --model az/models/<ckpt> --arch transformer --device mps`. Sides alternate between
battles (att/def fairness); the built-in side moves via the `suggest` op (planner action is
reported, the client applies it through the normal `action` op), every battle is
determinism-checked by replay. With `--sims 0` there is no agent — the runner requires at
least 1 sim; without `--model` the search uses uniform priors + the material heuristic
(phase-1 baseline). Baseline at 12 sims: pure MCTS loses to the built-in AI (~17% wins on a
6-battle smoke); meaningful numbers need a trained net and bigger sims counts.

**CRITICAL (2026-09-27): action encoding was wrong since stage 2.** `encoding.action_index` and
`transformer_model.decompose_action` read the wire args in constructor order, but the wire is
in REVERSE order (see the `Command` note in the protocol section). Every MOVE of a unit mapped
to `MOVE_BASE + uid` (a real root state: 46 legal moves -> 2 distinct indexes) and ATTACK
fields were scrambled. Consequences: the "~97% imitation accuracy" below is an artifact (the
targets collapsed), every checkpoint in `az/models/` is invalid, and net-guided MCTS/gate
numbers are meaningless. Fixed via `encoding.ctor_args()` (the single decoding point); the unit
test fixtures had been written in the same wrong order, so they never caught it — the
integration test `test_real_legal_moves_map_to_distinct_action_indexes` now checks real engine
output. Retrain from freshly generated data before any new numbers.

**Known data issue (legal moves, 2026-09-27)**: any `games.jsonl`/`expert.jsonl.gz`/battle-agent
record generated before the legal-move fix may contain phantom legal moves (no-ops in Release)
in battles with wide units or archers; MCTS visit counts over them are meaningless and the
legal-move index space differs. Regenerate data before the next training run.

**Known data issue**: before the replay-op fix (see protocol section), the batched search
replay ignored the search path — every MCTS node materialized the main-line-end state, so
visit counts in any `games.jsonl` produced before that fix are degenerate. Regenerate
self-play data before training on it. Expert data (`expert.jsonl.gz`, `auto` op) is not
affected.

## Battle server protocol (v1, JSON lines)

Ops (stdin): `new` (seed, `att`/`def` as `"monIdx x count,..."`, optional `tile`),
`action` (act = `CommandType` int, args = `Battle::Command` values), `reset`,
`replay` (batched: `acts[]` + `lens[]` + flat `args[]` — resets and applies the whole path
inside the engine in ONE roundtrip; the workhorse of MCTS), `snap`/`restore`/`snap_free`
(battle-state snapshots, see below), `suggest` (the built-in AI's action for the unit to
move, not applied), `quit`.

Illegal commands (2026-09-27): every command of an `action`/`replay`/`restore` path is checked
at its decision point (right unit, a type the enumeration produces, `Arena::isValid*Command`).
An illegal one stops the path and the op answers `{"ev":"error","what":"illegal action"}`
instead of a state; `action` keeps the old main line, `restore` stores no `save_as` snapshot.
Before this, Release silently dropped the command (state unchanged) and Debug hit `assert(0)`.

`new` extras: army slots `"0:13x30,2:21x25"`, formations `sat`/`sdf`, `wseed` (0/absent = the
pinned default 20260926 — never the seed of a previous `new`; it used to stick). Malformed stack
tokens are skipped (parsing via `std::from_chars`; `std::stoi` used to throw and kill the engine).

Replies (stdout): `{"ev":"state","turn":n,"cur":uid|-1,"units":[{u,side,mon,q,hpl,i,ti,sp,
shots,moved}],"obstacles":[...],"legal":[{act,args},...],"result":"att|def|draw"}`.
`legal` is present only when `cur != -1`. `EnumerateLegalMoves` filters every geometric
candidate through `Arena::isValidMoveCommand`/`isValidAttackCommand` — the SAME checks
`ApplyActionMove`/`ApplyActionAttack` run (lifted out of their lambdas in battle_action.cpp),
so the legal list and the engine cannot diverge. Before 2026-09-27 the list over-approximated
(cells that are not the head of a reachable wide-unit position, melee of non-blocked archers,
shots of blocked archers, moat cells). `suggest` adds `"expert":{act,args}`. Unknown
snapshot ids answer `{"ev":"error","what":"unknown snapshot id"}`.

Replay semantics (fixed — see the bug note below): the batched path is applied **on top of
the main line** (main line untouched); the `action` op extends the main line instead. So a
search may only run while the main line ends at the search root (selfplay/gate satisfy this:
every `action` rebuilds the main line).

Snapshots (battle-state search support, `Battle::ArenaSnapshot`):
- `snap` stores the current pause-point state under a client-chosen id; `restore` rewinds to
  it and may apply a path suffix and save the result under another id — ONE roundtrip per
  search-tree node; `snap_free` releases all stored snapshots.
- Snapshots hold plain data (unit states by value, board cells, graveyard/order/rng) and are
  owned by the **BattleServer**, so they survive the arena rebuilds that every main-line op
  performs; a `new` battle invalidates them.
- Restoring resumes mid-round (`Arena::resumeRound` → the interrupted `UnitTurn` continues
  directly). The resumed unit's morale was already drawn before the pause — `UnitTurn` skips
  the `SetRandomMorale` draw when resuming (`resumeMidTurn` flag), otherwise the RNG stream
  diverges (this bit us once).
- Open-field battles only: `captureSnapshot` asserts `castle == nullptr` (siege state is not
  captured). `UnitSnapshotState` (battle_troop.h) must be extended whenever `Unit` gains
  mutable battle-relevant members.

Guarantees and invariants (do not break):
- Battles are deterministic: same (stacks, tile, seed, action sequence) → identical battle.
  Batched replays of identical inputs are bit-identical (verified).
- In battle-server mode the world seed is pinned (`world.SetMapSeed(20260926)` in
  `RunBattleServer`): obstacle placement on the battle tile derives from the world seed,
  which `World::Defaults()` otherwise randomizes per process — without the pin, generated
  datasets are irreproducible across engine restarts (within a process everything is
  deterministic).
- Exactly one state reply per request; the client replies to a decision before sending any
  control op (a control op at a decision point unwinds the battle via `AbortBattle`).
- **Only one `Arena` instance may exist** (static pointer): destroy the old arena BEFORE
  constructing the new one (`_arena.reset()` first — this bit us once).
- `Command` has no public constructor from a runtime type; use `Command::FromRaw(type, values)`.
  Command values are stored in **REVERSE** ctor-param order (the ctor pushes the parameter pack
  right-to-left) and serialized as-is: MOVE is `[dst, uid]`, ATTACK is `[dir, tgt, dst,
  defenderUid, uid]`, SKIP is `[uid]`. `ApplyAction*` reads them via `GetNextValue()` from the
  back, i.e. in ctor order. To decode a command in C++, copy it and call `GetNextValue()` like
  the engine does (see `isAcceptableCommand` in battle_server.cpp) — never index by hand.
  (An earlier version of this file claimed ctor order; that was wrong and cost a debug round.)

## Strategic protocol (AIDecision)

Engine → agent: `turn_context` (per AI turn: day, resources, castles, heroes with army
strength) and `decision` (per hero activation: hero id + all positive-value candidate targets
with values/distances as computed by `Planner::getTargetCandidates()`). Agent → engine:
`pick` (must match a candidate, else ignored) or `skip` (built-in choice). Broken/gone agent
=> permanent fallback to the built-in AI. After each playthrough: `game_end` with per-player
results. Enabled with `FHEROES2_STRATEGY_SERVER=1` together with `FHEROES2_AUTO_PLAYTEST=1`.

## Timing rules and measurement conventions

- **60-second timeouts everywhere**: bridges raise `TimeoutError` instead of blocking forever;
  shell commands run with a 60s cap. Never leave a protocol read unbounded.
- Measured hot path (Release, small armies): ~3 000 full state replies/s (each includes legal
  move enumeration), ~19 000 raw action roundtrips/s; batched replay = one roundtrip for a
  whole path. Battle-state snapshot/restore (C++ `ArenaSnapshot`, see protocol section) is
  implemented: MCTS node materialization is O(1) restore+suffix instead of replay-from-root —
  measured ~3.3x search speedup at main-line depth 30 / 200 sims, with visit counts identical
  between the replay and snapshot descent paths (both verified equivalent).
- In this sandbox, interactive pipes to the game are unreliable; for deterministic runs use
  static stdin files (`cat cmds.txt | ./fheroes2 > out.txt`) or a PTY with ECHO disabled.

## Environment pitfalls (learned the hard way)

- Run the game as `./fheroes2` from the repo root — data paths resolve relative to argv[0].
- `json.dumps` must use `separators=(",", ":")` and the C++ side must be whitespace-tolerant
  (early desync was caused by `"att": "..."` vs `"att":"..."`).
- `pkill -9 -x fheroes2` before protocol runs — killed test runs leave zombies.
- Python 3.14 (brew) — check torch wheel availability before planning training; use a
  dedicated venv if needed.
- `args.def` is a syntax error in Python (`def` is a keyword) — use `dest=` in argparse.
- Map support in autonomous mode: `.fh2m` + `.mp2`; POL artifacts in `.fh2m` maps crash Debug
  builds but are skipped in Release.

## Code conventions (upstream, must follow)

- C++17, 4-space indentation, formatting enforced by `.clang-format` (run `clang-format -i`).
- Every new source file starts with the GPLv2 license header (copy from any existing file).
- Includes: IWYU-friendly, one system block, one project block, alphabetical.
- Game logic lives under `src/fheroes2/<module>/`; engine primitives in `src/engine/`.
- Debug logging via `DEBUG_LOG(DBG_XXX, level, stream-expr)` (needs `WITH_DEBUG`);
  always-on logging via `VERBOSE_LOG`/`ERROR_LOG`.
- Battle state determinism: battles are fully determined by (armies, tile, PCG32 seed, commands).
  `Battle::Command` (see `battle_command.h`) is the action protocol; `Arena::ApplyAction` mutates.
  Any random decision must go through the arena's PCG32 stream so replays stay reproducible.

## Transformer architecture (stage 3.5)

- `az/transformer_model.py`: policy/value transformer on the ready-made HuggingFace **Qwen3**
  body (RMSNorm, SwiGLU, RoPE, grouped-query attention; 4 layers, d_model=128, 4 heads /
  2 KV heads, head_dim=32, ~1.01M params). Tokenization: one token per board cell (continuous
  per-cell features projected to d_model) + [CLS] (value) + [ACTION] (policy query).
  Everything goes through `inputs_embeds` — no vocabulary tokens. RoPE replaced GPT2's
  learned `wpe`, so there is no position table to overflow.
- Actions are decoded autoregressively in two steps: prefill [cells, CLS, ACTION] gives the
  target-cell logits and the value; a one-token cell-identity decode step gives the direction
  distribution. The decode reuses the prefill KV-cache — the reason the cache exists here.
  Batched training: `forward_batch(states, decode_cells)` prefills the batch once and decodes
  directions for the attack rows via `DynamicCache.batch_select_indices`.
- transformers 5.x pitfalls (cost us an afternoon): the attention **mutates** the cache object
  it receives even with `use_cache=False`, so each decode needs its own cache copy
  (`_clone_cache` builds a fresh DynamicCache from cloned per-layer tensors); legacy tuple
  caches are rejected (`no attribute get_seq_length`). Qwen3Config silently defaults
  `head_dim` to 128 (NOT hidden_size/num_heads) — pass it explicitly or the model quietly
  grows to ~1.6M params.
- `az/train.py --arch transformer` trains with teacher forcing (cell CE + direction CE +
  value MSE); sample builders (`build_resnet_samples`/`build_transformer_samples`) are
  module-level functions so they can be unit-tested.
  `az/selfplay.py --arch transformer --model ...` runs network-guided self-play.
- `az/mcts.py` consumes a unified interface: `policy_value.evaluate(state) -> (priors by
  legal move index, value)`. `az/policy_value.py` wraps the ResNet; the transformer
  implements it natively.

## Tests (az/tests, pytest)

- Run: `az/.venv/bin/python -m pytest az/tests -q` (113 tests, ~30 s since the autonomous-mode
  speed fix; was ~2.5 min — the battle-agent
  integration file shares ONE engine session, ~33 s; do NOT go back to one-session-per-test,
  it cost 21 minutes). Coverage:
  `az/.venv/bin/python -m pytest az/tests -q --cov=az --cov-report=term-missing`
  (pytest-cov is installed in the venv; overall ~79%).
- Fully covered: encoding, model (ResNet), policy_value, transformer_model, plus unit tests
  for the train.py sample builders, gen_expert record conversion, selfplay.play_one /
  verify_determinism, strategy_env.run, the engine_bridge EOF/broken-pipe paths and the
  MCTS snapshot descent (SnapshotFakeEnv) — all against fake engines/processes (see the
  FakeEnv pattern in test_mcts.py).
- `test_battle_agent.py` — runner vs a scripted fake engine (`FakeProc`) and a fake replica
  (`FakeReplicaEnv` via `replica_factory`, which records the requested `map_name`);
  `make_runner` bypasses `__init__`, so every new runner attribute must be set there too.
- C++ has NO unit-test framework upstream (no gtest/ctest): the C++ side is tested through
  the integration files in `az/tests` against the real binary. **Run them against BOTH builds**:
  Debug turns engine-rejected commands into `assert(0)` crashes, Release silently drops them —
  the illegal-legal-move bug was only visible in Debug (see "Verification recipes").
- `test_server_protocol.py` also covers: `new` slots/formations/`wseed` reset/malformed tokens,
  every legal move is accepted by the engine (wide units + archers, applied from snapshots —
  a dropped command leaves the state unchanged), illegal commands in `action`/`replay`/`restore`.
- `test_agent_channels_protocol.py`: a dead agent gets exactly one query per channel; the game
  survives the agent process dying (SIGPIPE); one GameAgent serves both channels in a real game
  (Arena.mp2, 5 days: first battles on day 4, world seed is random per process — 4 days once
  produced no battle).
- `test_playtest_protocol.py`: equal seeds replay equal games, `p` colors match `game_end`,
  kingdom stats present. `test_strategy_bench.py`: verdicts, sign test, bootstrap, ForColor.
  `test_strategy_env.py::test_query_in_the_same_chunk_as_the_previous_event_is_answered` is the
  regression for the reader deadlock (an interactive fake engine that blocks for the reply).
- `test_game_agent.py` reuses those fakes (`make_agent` swaps `__class__` to `GameAgent`):
  interleaved hero decision -> battle -> hero decision ordering, outcome on both record kinds.
  `test_strategy_policies.py` covers the policies (pure functions).
- Integration (test_battle_agent_protocol.py + test_server_protocol.py, needs the built `./fheroes2`, skips otherwise):
  snapshot restore == walked states at every prefix, restore+suffix == full replay,
  snapshot lifetime (survive rebuilds, freed by snap_free, invalidated by `new`), and the
  `suggest` op (expert action present in the legal list, state not applied).
- NOT unit-tested (accepted): CLI mains (incl. gate.py), the 60 s watchdog TimeoutError path
  (too slow), real-engine behavior beyond the integration file. Fake engine scripts must
  close stdout (shell `exit 0`), never block holding it open, or `_read` spins until the
  watchdog.

## Licensing constraints

- This repo is **GPLv2**. Do NOT copy code from GPL-3-only projects (Stockfish, lc0) —
  implement from ideas/descriptions only. Python prototype code is separate and private to this fork.
- Map packs from third parties are data, never committed.

## Verification recipes

```sh
# Full build with zero warnings is the gate (msgfmt "Charset ... not portable" lines from the
# upstream .po files are not compiler warnings):
cmake --build build-release -j8 2>&1 | grep -E "warning:|error:" | grep -v Charset
cmake --build build -j8 2>&1 | grep -E "warning:|error:" | grep -v Charset

# Integration tests against the Debug binary (asserts on), then put Release back — the post-build
# step copies whichever build ran last to ./fheroes2:
cp build/fheroes2 ./fheroes2 && az/.venv/bin/python -m pytest az/tests -q
cp build-release/fheroes2 ./fheroes2

# Smoke-test a battle-heavy self-play game:
FHEROES2_AI_LOG=/tmp/fh2_ai.jsonl FHEROES2_AUTO_PLAYTEST=1 \
FHEROES2_AUTO_PLAYTEST_DAYS=7 FHEROES2_AUTO_PLAYTEST_MAP="Arena.mp2" ./fheroes2
python3 - <<'EOF'
import json
lines = [json.loads(l) for l in open('/tmp/fh2_ai.jsonl')]
from collections import Counter
print(Counter(e['ev'] for e in lines))
EOF
```

Expected event types: `session_start`, `turn_start`, `hero_target`, `visit`, `battle_start`,
`battle_action`, `battle_end`.

## AI research (stage 3)

- Battle server `auto` op: plays the current battle with the built-in BattlePlanner and
  streams (state with legal moves, expert action) records; 200-round cap guards against
  pathological matchups. Dataset generator: `az/gen_expert.py` (gzip JSONL).
- Expert-iteration result (INVALID — see the encoding note in "Gate runner"; retrain): the
  network imitated the built-in battle AI per move with ~97%
  accuracy (policy CE 0.027) on 7.5k expert records; checkpoints stay out of git
  (`az/models/`, see .gitignore).
- `az/engine_bridge.py` reads replies with a byte-level line assembler and a hard 60s cap
  per reply: the engine can hang mid-line inside the planner, so a plain readline() is not
  enough. All writes are bytes (`text=False`, `bufsize=0`).
- Tests live in `az/tests/` (pytest, run with `az/.venv/bin/python -m pytest az/tests -q`).
  Unit tests use a fake environment; the protocol integration test requires `./fheroes2`
  and is skipped when the binary is missing.
