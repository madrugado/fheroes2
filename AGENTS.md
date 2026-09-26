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

- `feature/az-battle-engine` — current research branch. AlphaZero-style battle engine work:
  - `az/` — Python prototype: battle environment (bridge to the engine), MCTS, self-play runner,
    training. Pure-Python first; hot paths move to C++ later.
  - C++ side: headless battle server (JSONL over stdin/stdout) reusing `Battle::Arena`,
    plus the event log and autonomous mode described above.
- `AI_LLM_PROTOCOL.md` — event-log protocol design (historical; the observation infra is reused
  for training data collection and evaluation).

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
