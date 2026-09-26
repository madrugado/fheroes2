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
- `FHEROES2_AI_LOG` — JSON-lines event log of the AI (see `AI_LLM_PROTOCOL.md`), written by
  `AILog` (`src/fheroes2/ai/ai_log.*`). Unset = disabled, zero cost.

Notes:
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
- `az/` — Python side of the research:
  - `engine_bridge.py` — battle environment client (`BattleEnv`), with 60s read watchdogs
  - `mcts.py` — PUCT search, heuristic leaf values, batched replay from root
  - `selfplay.py` — battle self-play runner, records `az/data/games.jsonl`
    (state, legal moves, MCTS visit counts, outcome) and verifies determinism per game
  - `strategy_env.py` / `strategy_run.py` — full-game strategic layer: policies
    `greedy|random|builtin`, records `az/data/strategy_<policy>.jsonl`
- C++ side:
  - `src/fheroes2/battle/battle_server.*` — headless battle server
  - `src/fheroes2/ai/ai_log.*` — JSONL event log
  - `src/fheroes2/ai/ai_decision.*` — strategic decision protocol
  - `Arena::Turns()/UnitTurn()` action-provider overloads — the seam for external drivers
- `AI_LLM_PROTOCOL.md` — event-log schema (the observation infra is reused for training data).

## Battle server protocol (v1, JSON lines)

Ops (stdin): `new` (seed, `att`/`def` as `"monIdx x count,..."`, optional `tile`),
`action` (act = `CommandType` int, args = `Battle::Command` values), `reset`,
`replay` (batched: `acts[]` + `lens[]` + flat `args[]` — resets and applies the whole path
inside the engine in ONE roundtrip; the workhorse of MCTS), `quit`.

Replies (stdout): `{"ev":"state","turn":n,"cur":uid|-1,"units":[{u,side,mon,q,hpl,i,ti,sp,
shots,moved}],"obstacles":[...],"legal":[{act,args},...],"result":"att|def|draw"}`.
`legal` is present only when `cur != -1`.

Guarantees and invariants (do not break):
- Battles are deterministic: same (stacks, tile, seed, action sequence) → identical battle.
  Batched replays of identical inputs are bit-identical (verified).
- Exactly one state reply per request; the client replies to a decision before sending any
  control op (a control op at a decision point unwinds the battle via `AbortBattle`).
- **Only one `Arena` instance may exist** (static pointer): destroy the old arena BEFORE
  constructing the new one (`_arena.reset()` first — this bit us once).
- `Command` has no public constructor from a runtime type; use `Command::FromRaw(type, values)`.

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
  whole path. Remaining search cost is replay-from-root (O(depth) per simulation) — the fix is
  a C++ battle-state snapshot/restore (make/unmake analogue), planned.
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

## Licensing constraints

- This repo is **GPLv2**. Do NOT copy code from GPL-3-only projects (Stockfish, lc0) —
  implement from ideas/descriptions only. Python prototype code is separate and private to this fork.
- Map packs from third parties are data, never committed.

## Verification recipes

```sh
# Full build with zero warnings is the gate:
cmake --build build-release -j8 2>&1 | grep -iE "warning|error"

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
