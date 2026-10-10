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
- `FHEROES2_RNG_STREAMS=1` (with `FHEROES2_AUTO_PLAYTEST_SEED`; off by default = upstream behavior):
  separate random streams per turn — the game's generator is re-seeded from (game seed, day, player
  color, `FHEROES2_RESEED` salt from its day on) before every new day and every player's turn
  (`AIDecision::beginRandomStream`, called in `game_startgame.cpp`). A different strategic answer then
  changes the dice only until the next turn boundary, not for the rest of the game. Python:
  `StrategyEnv(rng_streams=True)`, `--rng-streams` in strategy_games/strategy_loop/label_noise/
  oracle_headroom. Battles were already seeded separately (map seed + armies); with the switch the
  battle seed leaves the troop COUNTS out (`computeBattleSeed`, `AIDecision::randomStreamsActive`): one
  peasant more used to give a battle entirely other dice. First noise check (per-turn streams only,
  seeds 301-330, hero label): luck sd -10%, clear differences 10 -> 14 / 90 — small.
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
  - `5479006f` headless battle server + `rl/` prototype skeleton
  - `5558e14d` strategic decision protocol (`AIDecision`)
  - `d27c37d6` batched replay op + 60s protocol watchdogs
  - `d12cfd475` transformer body GPT2 → Qwen3 + test-coverage pass (63 tests)
  - `48f2a26cd` battle-state snapshot/restore + `suggest` op + `rl/gate.py` (see "Gate runner" below)
  - battle-agent channel for real battles + `rl/battle_agent.py` (see "Battle-agent channel")
  - unified game agent + `tempo` strategic policy + channel-fallback fixes (see "Unified game agent")
  - C++ hardening: legal moves == engine validation, illegal commands answered with an error,
    `wseed` reset, robust stack parsing, SIGPIPE-safe agent channels (+ integration tests)
  - hero battles in the MCTS replica: commanders replicated via save-game serialization (see
    "Battle-agent channel"); then full replication (sieges, towns, hero spells)
  - expert data v3 from real battles + wide-unit tail strikes + 1460-slot encoding + transformer
    sizes (see "Expert data v3 and retraining")
- `rl/` — Python side of the research (renamed from `az/` on 2026-09-28; the venv moved with it:
  `rl/.venv/bin/python`, data in `rl/data/`, checkpoints in `rl/models/`):
  - `engine_bridge.py` — battle environment client (`BattleEnv`), with 60s read watchdogs
  - `mcts.py` — PUCT search; node states materialize via battle-server snapshots when the
    engine supports them (fallback: batched replay on top of the main line)
  - `selfplay.py` — battle self-play runner, records `rl/data/games.jsonl`
    (state, legal moves, MCTS visit counts, outcome) and verifies determinism per game
  - `gate.py` — win-rate runner: our MCTS (optional trained net) vs the built-in BattlePlanner
    (via `suggest`), sides alternate, determinism-checked per battle
  - `battle_agent.py` — external agent for real battles (random|planner|policy|mcts)
  - `game_agent.py` — one agent for both channels; `strategy_policies.py` — strategic policies
  - strategic layer = hero targets + building + hiring (see "Strategic protocol");
    `strategy_bench.py` — paired benchmark; `strategy_rollout.py` + `strategy_model.py` —
    counterfactual labels and the learned strategic policy (all four query kinds)
  - `harvest_battles.py` — real battle setups from seeded built-in games (gen_expert/gate input)
  - `strategy_env.py` / `strategy_run.py` — full-game strategic layer: policies
    `greedy|random|builtin`, records `rl/data/strategy_<policy>.jsonl`
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
log events carry `src:"agent"|"planner"`). Runner: `rl/battle_agent.py --policy
random|planner|policy|mcts` (full wire format in `rl/README.md`, "Real-battle integration").

- C++: `battle_agent.{h,cpp}`; `Battle::Loader` sends `battle_start` (seed, tile, wseed,
  `searchable` (always 1 now), per side: stacks as `[slot,mon,count]`, spread formation, color
  `c`, for a hero-led side `hid` + `hero`, `garrison:1` when the defenders are the castle's
  garrison; on a castle/town tile the castle's serialization `castle`) and `battle_end`. Agent gone =>
  permanent built-in fallback; an illegal action falls back for that decision only and reports
  `battle_fallback`. Agent replies: `action` / `planner` / `skip`.
- C++: `battle_server.{h,cpp}` — shared `SerializeArenaState`/`EnumerateLegalMoves`; the `new`
  op accepts army slots (`"0:13x30,2:21x25"`), formation flags `sat`/`sdf`, `wseed`
  (0 = keep the pinned default), colors `acol`/`dcol`, commanders `ahid`/`ahero`,
  `dhid`/`dhero`, the castle `castle` and `dgar` (garrison defends).
- Hero battles (2026-09-27): the commander travels as the hero's **save-game serialization**
  (`Battle::EncodeCommander` = `operator<<(Heroes)`, hex, ~700 chars) instead of a hand-picked
  stat list — primary/secondary skills, artifacts, spell book, spell points, visited objects
  (morale/luck sources) and the army are all exact. The replica restores it into the world's
  hero with the same id (same map loaded) at EVERY arena rebuild (the hero state goes back to the
  battle start) and fights with the hero's army. Two more things had to match: (1) army colors
  (real neutral armies are color 0, the replica used RED/BLUE); (2) player control — the
  battle server now makes every player of the map AI-controlled like the autonomous playtest:
  the bad-morale roll draws an extra random number for AI units, so a human slot in the replica
  shifted the RNG stream (found by diffing the arena RNG state per mirrored move).
  Result: 0 desyncs over ~30k decisions (Battlefi/Thechaos 30 days, 2kings, Arena); before: 269
  of 642 decisions of a 7-day Battlefi game degraded.
- Full replication (2026-09-27): EVERY real battle is searchable — sieges, town battles and hero
  spells included.
  - Castles travel like heroes: `Battle::EncodeCastle` = `operator<<(Castle)` (buildings incl.
    towers/moat/fortifications/captain's quarters, the captain, garrison, owner), restored into
    the replica world's castle on the battle tile before the heroes (their castle modifiers look
    it up); `dgar` makes the castle's army the defenders.
  - Hero lookups by position: `Castle::GetHero()`/`Heroes::inCastle()` search heroes by map
    position and `world.getCastleEntrance()` checks the tile's object type, which for a tile
    with a hero comes from THAT hero's `_objectTypeUnderHero`. The battle server therefore clears
    all heroes from the map tiles once after loading (`clearHeroesFromTiles`) and parks every
    hero at (-1,-1) before each battle (`parkAllHeroes`); only the restored commanders stand on
    the map. Found as a "bad battle setup": a map-placed starting hero on a castle entrance,
    restored by an earlier battle with its real-game state, made the castle vanish.
  - Snapshots capture sieges and commanders (`ArenaSnapshot`): towers (unit state + destroyed
    flag), bridge (destroyed/down), the per-round catapult/tower flags (now `Arena` members
    `_catapultActedThisRound`/`_towersActedThisRound`, reset by `Turns()`, kept by
    `resumeRound()` — they used to be locals of `runRound()`, a resumed round would re-fire the
    catapult), and per side the commander's spell points + `SPELLCASTED`. Walls are board cells.
  - Hero spells are legal moves: `EnumerateSpellCasts` (appended after SKIP) mirrors the battle
    interface's rules — combat spell, `CanCastSpell`, not `isDisableCastSpell` (one spell per
    round, Sphere of Negation, elemental rules), a valid target: no target (mass/summon/
    Armageddon/Earthquake: `[-1, spell]` on the wire), every live unit's head cell that
    `AllowApplySpell` accepts (Mirror Image too), graveyard cells for resurrection, all 99 cells
    for area spells, and unit x passable empty cell for Teleport (`[dst, src, spell]`).
    `isAcceptableCommand` accepts a SPELLCAST iff the enumeration contains it. After a cast the
    same unit gets another decision (the unit's turn is not over), now without spells.
  - State replies carry `"heroes":[{side,sp,cast}]` (commander sides only) and, in sieges,
    `"siege":{"cells":[[idx,obj]...],"towers":[l,c,r],"bridge":0|1|2}`; both are in the
    replica sync check (`STATE_SYNC_FIELDS`).
  Verified: 0 desyncs over ~58k decisions (6 games, 32 castle/town battles incl. sieges, 38
  agent spells; Debug too); the built-in AI's spells (`suggest`) are always in the legal list.
- MCTS mode searches in a headless replica rebuilt from `battle_start`; every agent move is
  mirrored and the state diffed (`turn/cur/units/obstacles/heroes/siege`) — first mismatch
  degrades the rest of the battle to policy/planner.
- Gotchas: decision queries arrive as `"ev":"state"` WITH a `"bid"` field (don't wait for a
  `battle_state` event); the reader must be a byte-level line assembler (states exceed the pipe
  buffer, buffered readline + select() starve); the replica must load the same map
  (`BattleEnv(map_name=...)`) or obstacles mismatch at the root. `BattleEnv` strips
  `FHEROES2_AI_LOG`/`FHEROES2_BATTLE_AGENT`/`FHEROES2_STRATEGY_SERVER` from the child env
  (`CHILD_ENV_BLOCKLIST`): before that the replica appended its own `battle_start`/
  `battle_action` events (`t:0`, colliding battle ids) to the real game's AI log.
- Verified: `random` — full 7-day playtest, 466 decisions; `mcts --sims 4` — ~300 ms per
  searched decision, replica synced in every searchable battle. Release build, zero warnings.
- Agent death: with a channel enabled the engine ignores SIGPIPE (`prepareChannel`/
  `prepareDecisionChannel`), so an agent process that exits no longer kills the game (was exit
  code -13); the next read hits EOF, the channel breaks, the built-in AI finishes the game.
- Round limit: engine battles have no round limit, so `BattleAgentRunner` hands a battle to the
  built-in AI once `state["turn"] > max_battle_turns` (default 30, `DEFAULT_MAX_BATTLE_TURNS`,
  `--max-battle-turns` in both `battle_agent.py` and `game_agent.py`). Without it two MCTS sides
  without a network never engaged (131 165 moves in one battle).
- Known limitations (documented, not fixed): retreat/surrender are not agent actions (the
  built-in AI never gets to decide them while the agent answers); a battle the agent hands to
  the built-in AI (round limit, `planner` replies) is not mirrored, so the replica is not used
  after that; `clang-format` is not installed in this sandbox (style matched by hand).

## Unified game agent (committed 2026-09-27)

- `rl/game_agent.py` — ONE process serves both channels (`AIDecision` hero targets +
  `BattleAgent`): `GameAgent` subclasses `BattleAgentRunner` and consumes strategic events in
  the `_handle_event` hook; `extra_env` adds `FHEROES2_STRATEGY_SERVER=1`. The battle-only runner
  pops a stray `FHEROES2_STRATEGY_SERVER` from the env (else the engine blocks on decisions
  nobody answers).
- `rl/strategy_policies.py` — `greedy|random|builtin|tempo` (shared with `strategy_run.py`);
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

- `rl/strategy_bench.py` — paired head-to-head: per seed one control game (all built-in) + one
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
  value already accounts for distance; tempo overrides rarely (reports in `rl/data/bench_*.json`).
- `StrategyEnv` had the select()+buffered readline() deadlock (turn_context + decision in one
  chunk -> the decision sits in Python's buffer, the engine waits for the reply; shows up under
  load as a 60 s TimeoutError). It now uses `rl/line_reader.py` (`LineReader`, shared with
  battle_agent.py). NEVER read engine pipes with select() + readline().

## Learned strategic policy (all four query kinds, 2026-09-27)

- **Machine load rule (user request, repeated 2026-09-28):** ONE heavy job at a time — a
  training, a generation/benchmark run, a full build or the test suite, never two together; check
  `ps` for leftovers (engines, replicas, trainings) before starting one. Overlapping a 50m
  training with builds and test runs filled the 16 GB laptop and it had to be restarted.
- Earlier form of the rule: the user's laptop must stay usable — at most **2**
  parallel engines (`--jobs 2`, the default now), engines run under `nice -n 10`
  (`StrategyEnv(niceness=10)`), torch limited to 2 threads, and never train while a
  generation/benchmark run is going. 8 jobs + training overloaded it (and a starved engine
  tripped the 60 s read timeout).
- `rl/strategy_rollout.py` — counterfactual labels for EVERY strategic query kind (hero
  `target`, `build`, `hire`, `army` budget): the base seeded game enumerates queries; for a
  sampled query n (day t, color p) the baseline branch replays to day t+H with built-in answers,
  each alternative branch replays identically but answers option j at n. Label = p's stat delta
  at t+H vs the built-in answer (label 0). The built-in answer is known for every kind (target:
  top candidate, build: the base game's `build_result`, hire: `bi`, army: 100%). Valid because
  the day limit only cuts the game (prefix identical, verified) and every branch re-checks query
  n (`Branch.expected`; a branch answers only its expected query). Failed/stuck branches are
  skipped with a log line instead of killing the run. Data:
  `rl/data/strategy_rollouts_all_<map>_h<H>.jsonl`; old target-only files
  (`strategy_rollouts_<map>_h<H>.jsonl`) are converted on load (`convert_legacy`).
- `rl/strategy_model.py` — ONE advantage model per kind (ridge / tiny MLP, saved as JSON,
  model format `version: 2`, fixed feature width per kind) over option + context features;
  label = d_str + 2000*d_castles + 10000*d_outcome; leave-seeds-out CV measured by the realized
  gain of the argmax policy, stored in the model file. Enable rule per kind: `--rule all`
  (default: every kind answered), `--rule ci` (95% bootstrap lower bound of the CV gain > 0 —
  the setting for quality work), `--rule mean` (mean > 0; proved misleading). A disabled kind
  keeps the built-in choice (`skip`).
- `strategy_policies.LearnedPolicy` (`--policy learned --model rl/models/strategy_model.json`,
  `game_agent.py --strategy learned --strategy-model ...`) answers all four kinds and still
  reads the first, target-only model format.
- First round, target-only model — results (Battlefi, 17 seeds 101-117, 425 decisions, H=7): alternatives vs built-in 209 better /
  335 equal / 183 worse; label std 1524 vs mean 130 — dominated by chaos (a different pick
  reshuffles the RNG stream). CV gain +97/decision, 95% CI [-35, +232] (not significant).
  Bench (seeds 1-10, 30 days): army strength **+1066, CI [+352, +1873]** (significant), but
  castles -0.15 and one extra loss -> verdict 23 better / 5 equal / 32 worse (p=0.28). The model
  trades castles for army; not an improvement by the benchmark's (outcome, castles, strength)
  order. Model file is out of git (`rl/models/`); retrain with the command in the module doc.
- Ideas for the next round: shorter horizons (1-3 days) to cut chaos, several horizons per
  decision, a much larger castle weight (or a castle-only classifier as a veto), more seeds
  (generate overnight with `--jobs 2`), a placebo branch to measure the pure-chaos spread.

## Model-driven game, end to end (2026-09-27)

One process plays whole games with every decision from models/search:
`rl/.venv/bin/python rl/game_agent.py --strategy learned --battle mcts --sims 4 --map Battlefi.mp2 --days 7`
(strategy: `rl/models/strategy_model.json` via `--strategy-model`). Verified: 7-day Battlefi game in
~6 s, 568 strategic queries (the model answered all four kinds; 21/21 agent builds applied) +
1216 battle decisions (1076 by MCTS, replica synced for 1110). Quality is NOT the goal yet (user
decision): the default model file is trained with `--rule all` (every kind answered); the
strict `--rule ci` enabled no kind (CV 95% CIs: target [-13, +214], build [-24, +275], hire
[-5, +617], army [-158, +165]); the all-kinds "mean" model was clearly worse in the paired bench
(10 better / 50 worse, -0.5 castles, -1593 army strength, +5703 unspent gold — it hoards: the
7-day label horizon makes not spending look free). Fix the labels before tuning models.

Two things this uncovered (both fixed, both tested):
- **Endless battles.** Engine battles have no round limit; two sides driven by MCTS without a
  network danced forever (131 165 moves in one battle, no attack). `BattleAgentRunner` hands a
  battle to the built-in AI after `max_battle_turns` rounds (default 30, `--max-battle-turns`).
- **Quadratic battle server.** Every main-line op (`action`, and `replay([])` at the start of
  every MCTS search) rebuilt the arena and replayed the whole main line; the illegal-command
  validation made it worse (pathfinder per replayed command). Now the server keeps a snapshot of
  the main-line end: `action`/`replay` restore it and apply only the new commands (validated);
  replay-from-root remains as the fallback (battle over) and as a reference: `replay` with
  `"full":1` / `BattleEnv.replay(path, full=True)`. Test: fast path == full replay at every step
  of long random battles.

## LLM agent via TRL GRPO (2026-09-27, not trained yet)

fheroes2 as a multi-turn tool-calling environment for TRL's `GRPOTrainer(environment_factory=...)`
(trl 1.14 + peft installed in `rl/.venv`; torch/transformers were not touched by the install).
- `rl/grpo_env.py` — `HeroesStrategyEnv`: one seeded game per episode, the LLM plays ONE color on
  the strategic layer (target/build/hire/army; battles and the other players stay built-in).
  `reset(**row)` starts the engine and returns the first question (TRL appends it to the prompt),
  the only tool `choose(option)` answers and returns the next question, `get_reward()` finishes the
  game with built-in answers if the model stopped / ran out of tokens and scores it: (score - score
  of the cached all-built-in control game) / 1000 - 0.1 x invalid calls - 1.0 x unanswered share;
  score = str + 2000 x castles + 10000 x outcome. TRL turns EVERY public method into a tool: keep
  helpers `_private`. Questions without a real choice (0-1 candidates) are auto-skipped; targets
  are capped at `max_options` (8), build/hire lists never (the built-in pick can be anywhere).
  Every question marks the built-in choice ("<- default AI"); building has an explicit "let the
  default AI decide" option (= `skip`): an explicit "build nothing" is NOT equivalent to the
  built-in "nothing" (the game diverged). Names come from the engine headers (`rl/game_names.py`).
- Measured on 2kings, 7 days, one color: 16-22 real choices, ~4-5k tokens per episode (Qwen2.5
  template), 1.5-3 s of engine time. Echoing the advisor gives reward exactly 0 (tested on the
  real engine). No engine-side read timeout on strategic answers (blocking getline), so slow
  generation is fine; one idle engine process per open episode.
- Raw Qwen2.5-0.5B-Instruct answers with plain text ("0", "pick 2"): 1/16 samples was a proper
  tool call. Hence `rl/grpo_sft.py`: `gen` plays the built-in AI's choices into SFT conversations
  (exact replays of the control game, reward 0 checked per episode), `train` = LoRA SFT with
  `assistant_only_loss` + merge (saves the tokenizer with the ORIGINAL chat template, so TRL
  recognizes it for tool-call parsing). Then `rl/grpo_train.py --model rl/models/grpo_sft ...`.
- Tests: `rl/tests/test_grpo_env.py` (fake engine; one-step GRPOTrainer and SFTTrainer smokes with a
  tiny random Qwen2 + the cached Qwen2.5 tokenizer; real-engine expert == control).

## Next (plan)

0. Expert data regenerated (v3, real battles included), ResNet retrained and gated — see
   "Expert data v3 and retraining". Open: the 50m transformer, less overfitting (early stopping,
   more battles), self-play data on top of the expert warm start.
1. (done: see "Strategic benchmark" — tempo is no better than builtin.)
1b. Strategic layer integration — done: targets + building + hiring + army budget; the learned
    policy drives all four kinds end to end together with MCTS battles (see "Model-driven game").
2. Strategic model quality (after the integration): longer / multi-horizon labels (the 7-day
   horizon rewards hoarding), gold in the label, then the strict CV rule and the paired bench.
   Earlier round: target-only model — first round done (see "Learned strategic policy"): more army,
   fewer castles, not a net win yet. Next round: shorter horizons / castle weight / more data.
3. (done: every real battle replicates exactly — heroes, sieges, towns, hero spells; see
   "Battle-agent channel". The nets got spell slots, see "Transformer architecture"/encoding.)

## Gate runner (rl/gate.py)

Plays our engine vs the built-in BattlePlanner: `rl/.venv/bin/python rl/gate.py --battles 40
--sims 32 --model rl/models/<ckpt> --arch transformer --device mps`. Sides alternate between
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
targets collapsed), every checkpoint in `rl/models/` is invalid, and net-guided MCTS/gate
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
pinned default 20260926 — never the seed of a previous `new`; it used to stick), colors
`acol`/`dcol`, commanders `ahid`+`ahero`/`dhid`+`dhero` (a commander that cannot be restored ->
`{"ev":"error","what":"bad battle setup"}`), castle `castle` + `dgar` (see "Full replication").
Malformed stack
tokens are skipped (parsing via `std::from_chars`; `std::stoi` used to throw and kill the engine).

Replies (stdout): `{"ev":"state","turn":n,"cur":uid|-1,"units":[{u,side,mon,q,hpl,i,ti,sp,
shots,moved}],"obstacles":[...],"heroes":[{side,sp,cast}],"siege":{...}(sieges only),
"legal":[{act,args},...],"result":"att|def|draw"}`. `legal` includes hero SPELLCASTs
(`EnumerateSpellCasts`).
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

Main-line fast path (2026-09-27): the server keeps a snapshot of the main-line end. `action`
and `replay` restore it and apply only the new commands (still validated) instead of rebuilding
the arena and replaying the whole main line from the root — that was quadratic in battle length
(worse with per-command validation). Replay-from-root remains the fallback (battle already over)
and the reference: `replay` with `"full":1` / `BattleEnv.replay(path, full=True)`. The fast
path must stay equal to the full replay (tested at every step of long random battles).

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
- Sieges and commanders are captured too (towers, bridge, catapult/tower round flags, spell
  points + SPELLCASTED). `UnitSnapshotState` (battle_troop.h) / `ArenaSnapshot` must be extended
  whenever `Unit`/`Arena` (or a commander) gains mutable battle-relevant state.

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
strength) and four kinds of choice queries — `decision` (hero target: all positive-value
candidates from `Planner::getTargetCandidates()`), `build` (per castle at the end of the turn:
every building allowed by difficulty rules and affordable now or via a marketplace trade;
followed by `build_result`), `hire` (castle x tavern-offer candidates + the built-in choice
`bi`) and `army` (before a castle hires monsters: the affordable offer). Agent → engine: `pick`
/ `build` (`b`=0: nothing) / `hire` (`castle`=-1: none) / `army` (`pct` budget) or `skip`
(built-in choice) for any query; a non-candidate answer = built-in choice. Full wire format:
`rl/README.md` "Strategic layer".

Strategic layer implementation notes (2026-09-27):
- Hooks: `AI::Planner::CastleTurn` (build; defensive castles still `reinforceCastle` first),
  `AI::Planner::purchaseNewHeroes` (hire; `recruitHero(castle, hero, buyArmy)` overload applies
  the choice with the same army-buying rule). Candidate enumeration iterates all 32
  `BuildingType` bits through `Castle::CheckBuyBuilding` (safe for every bit).
- Army: `reinforceCastle( castle, reason )` asks for a budget percent of the kingdom's funds
  (reasons "defense"/"visit"/"hire"); the built-in composition logic runs with
  `getRecruitLimit( monster, funds - reserve )`, reserve = funds - funds*pct/100. pct=100 is
  byte-identical to the built-in path (reserve 0); below 100 troop upgrades are skipped
  (they are paid without a budget check). Monster purchases by heroes at map dwellings stay
  built-in. An agent build answer also skips the built-in boat purchase of `CastleDevelopment`.
- Invariant (tested): answering every build/hire query with the built-in AI's own choice
  replays exactly the all-`skip` game. NOT guaranteed: channel-on vs channel-off games — the
  hire query materializes the tavern offer (`Kingdom::GetRecruits()` may generate heroes and
  consume RNG) when the built-in AI would not look at it. Always compare against channel-on
  controls (strategy_bench does).
- The soft hero limit of the built-in AI does not bind the agent (up to the kingdom maximum). Broken/gone agent
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

- `rl/transformer_model.py`: policy/value transformer on the ready-made HuggingFace **Qwen3**
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
- Hero spells (2026-09-27): the first decoding step is over 99 cells + SKIP + 73 spell tokens
  (`NUM_POLICY_TOKENS`, `SPELL_TOKEN_BASE + spell id`); the ResNet's fixed action space grew to
  1460 (`encoding.SPELL_BASE + spell id`, after the attack block grew, see "Expert data v3"). The target of a spell is not encoded: all legal targets
  of one spell share its slot/token and split its probability evenly (MCTS picks the target).
  `NUM_SCALARS` 3 -> 9 (per side: commander present, spell points/100, cast this round, from the
  state's `heroes`). Old checkpoints do not load (shapes changed) — they were invalid anyway.
- Battle input = what the battle screen shows (user request 2026-09-30): until then the transformer
  saw only the 11 planes per cell — NOT the creature (`mon`), the count only as log2/8, health capped
  at 100 hp, and no commander at all (`NUM_SCALARS` fed only the ResNet). Now `enc.battle_tokens`:
  per cell the planes + the stack on it (head and tail: exact count log+linear, top creature's hp
  uncapped, speed, shots, moved) + the creature id (`mon_embed`), then one token per commander
  (attacker's, defender's: present, spell points, cast this round, turn). Sequence
  [99 cells, att hero, def hero, CLS, ACTION] (`BATTLE_TOKENS`, `CLS_POS`, `ACTION_POS`). New
  layers `unit_proj`/`mon_embed`/`hero_proj`; older checkpoints load with them fresh (and their CLS/
  ACTION positions moved: fine-tune, do not evaluate them as they are). The ResNet planes are unchanged.
- Battle history (user request 2026-09-30): the transformer sees the actions of the battle so far,
  one token per action in front of the board — [history (last `MAX_BATTLE_HISTORY` = 256), 99
  cells, att hero, def hero, CLS, ACTION]. An action (`battle_history_entry`, from the state it was
  taken in + the command) = acting creature (`mon_embed`), own/enemy relative to the side to move,
  kind (move/attack/skip/spell), first-step token (target cell / skip / spell, `hist_action_embed`
  over the policy's token space), strike direction, round, stack size. Rows are `BATTLE_ROW_W` wide
  with a row-kind column (`battle_rows`); batches are LEFT-padded (`batch_rows`) with an attention
  mask and position ids over real tokens only (`_attention`), so CLS/ACTION sit at the end of every
  row; tested: padding does not change a shorter history, evaluate == forward_batch. Training data:
  `train.attach_battle_history` rebuilds the history from the previous expert records of the same
  battle (gen_expert files keep a battle's records together and in order; longest battle 191
  decisions). States without "history" (MCTS, battle_agent) are evaluated with an empty history —
  open: pass main line + search path there.
- Creatures shared by battle and strategy (user request 2026-09-30): strategic tokens used to drop
  the creature id of every army stack (only count, strength, speed, shooter, flyer). Now every token
  ends with `STRAT_MON_SLOTS` (5) creature ids + 1 (`strategy_net._mon_slots`: own heroes, rivals,
  own garrisons, rival castles; 0 elsewhere), embedded by the battle's `mon_embed` through one
  projection per slot (`strat_mon_slots`, `transformer_model._strategic_embeds`); `strat_proj` reads
  the first `STRAT_FEATURE_W` columns as before, so older checkpoints still load. Same id space as the
  battle's `mon` (Monster::GetID); the collected data already had the ids, nothing to regenerate.
  ONE encoding for both: `enc.monster_token`. Checked on a real game (Battlefi 14 days, 43 hero
  battles): battle_start == first battle state 43/43; battle == strategic hero token in 29/29 battles
  whose hero had the same creature types. The other 14 differ in the ARMY, not the encoding:
  `turn_context` is a snapshot at the start of the player's turn, the hero bought/collected troops
  before the battle. So strategic queries in mid-turn see start-of-turn armies (open: send a fresh
  hero snapshot with each query).
- Sizes (`transformer_model.PRESETS`, `train.py --size`): `small` (~1.0M, the prototype),
  `50m` (hidden 512, 12 layers, 8/4 heads of 64, SwiGLU 2048: 47.4M, window 512 since 2026-09-29),
  `100m` (hidden 768, 12 layers, 12/4 heads of 64, SwiGLU 3072: 104.1M, window 512) and `0.5b`
  (Qwen3-0.6B layer shape x 32 layers: 503.7M, window 2048). A battle state is 104 tokens; the
  window is headroom. Checkpoints store the shape (`save_checkpoint`/`load_checkpoint`,
  `{"arch","config","state_dict"}`; a bare state dict = `small`). Measured on the M1 Pro 16 GB
  laptop (MPS): 0.5b — AdamW does not fit (batch 8: 12.6 GB, 47 s/step in swap; batch 32 OOM),
  with SGD-sized memory 1.7 s/step at batch 8 (~4 h/epoch of 67k positions), `evaluate` 0.28 s
  (32 sims ~9 s per decision) — user decision: train `50m` locally instead (batch 32: 0.79 s/step,
  ~28 min/epoch, 3.7 GB; `evaluate` 0.073 s). Transformer training: AdamW + 5% warmup + cosine
  to 10% (`warmup_cosine`), grad clip 1.0, checkpoint every epoch.
- transformers 5.x pitfalls (cost us an afternoon): the attention **mutates** the cache object
  it receives even with `use_cache=False`, so each decode needs its own cache copy
  (`_clone_cache` builds a fresh DynamicCache from cloned per-layer tensors); legacy tuple
  caches are rejected (`no attribute get_seq_length`). Qwen3Config silently defaults
  `head_dim` to 128 (NOT hidden_size/num_heads) — pass it explicitly or the model quietly
  grows to ~1.6M params.
- `rl/train.py --arch transformer` trains with teacher forcing (cell CE + direction CE +
  value MSE); sample builders (`build_resnet_samples`/`build_transformer_samples`) are
  module-level functions so they can be unit-tested.
  `rl/selfplay.py --arch transformer --model ...` runs network-guided self-play.
- `rl/mcts.py` consumes a unified interface: `policy_value.evaluate(state) -> (priors by
  legal move index, value)`. `rl/policy_value.py` wraps the ResNet; the transformer
  implements it natively.

## Tests (rl/tests, pytest)

- Run: `rl/.venv/bin/python -m pytest rl/tests -q` (190 tests, ~1 min since the autonomous-mode
  speed fix; was ~2.5 min — the battle-agent
  integration file shares ONE engine session, ~33 s; do NOT go back to one-session-per-test,
  it cost 21 minutes). Coverage:
  `rl/.venv/bin/python -m pytest rl/tests -q --cov=rl --cov-report=term-missing`
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
  the integration files in `rl/tests` against the real binary. **Run them against BOTH builds**:
  Debug turns engine-rejected commands into `assert(0)` crashes, Release silently drops them —
  the illegal-legal-move bug was only visible in Debug (see "Verification recipes").
- `test_server_protocol.py` also covers: `new` slots/formations/`wseed` reset/malformed tokens,
  every legal move is accepted by the engine (wide units + archers, applied from snapshots —
  a dropped command leaves the state unchanged), illegal commands in `action`/`replay`/`restore`.
- `test_battle_agent_protocol.py::test_mcts_replica_stays_synced_in_hero_battles` — a seeded
  real game with the MCTS runner: every searchable decision keeps the replica in sync, and at
  least one hero battle is replicated (fails with 37/287 desynced decisions when the commanders
  are not sent). `test_server_protocol.py`: bad commanders -> error and the server stays usable,
  army colors (neutral 0) accepted.
- `test_replica_protocol.py`: harvests real `battle_start` setups from a seeded 14-day Battlefi
  playtest (heroes, castles, garrisons, a siege) and checks in the battle server: every setup
  rebuilds; random games preferring spells keep snapshot restore == walked state and fast path
  == full replay (siege: a tower and the bridge get destroyed, walls change); a cast sets
  `cast` and spends spell points; the built-in AI's spells are legal. Run it on Debug too.
- `test_agent_channels_protocol.py`: a dead agent gets exactly one query per channel; the game
  survives the agent process dying (SIGPIPE); one GameAgent serves both channels in a real game
  (Arena.mp2, 5 days: first battles on day 4, world seed is random per process — 4 days once
  produced no battle).
- `test_playtest_protocol.py`: equal seeds replay equal games, `p` colors match `game_end`,
  kingdom stats present. `test_strategy_bench.py`: verdicts, sign test, bootstrap, ForColor.
  `test_strategy_env.py::test_query_in_the_same_chunk_as_the_previous_event_is_answered` is the
  regression for the reader deadlock (an interactive fake engine that blocks for the reply).
- `test_strategic_layer_protocol.py`: all four query kinds arrive, echoing the built-in
  choices replays the built-in game, random agent choices are applied (build_result, hires
  above the built-in limit); `test_learned_policy_drives_every_kind_in_a_real_game` — the
  learned policy answers target/build/hire/army queries in a real game.
- `test_server_protocol.py::test_main_line_snapshot_fast_path_matches_the_full_replay` — the
  snapshot fast path of `action`/`replay` == `"full":1` replay at every step of long random
  battles. `test_battle_agent.py::test_long_battles_are_handed_to_the_builtin_ai` — the
  round limit (`max_battle_turns`).
- `test_strategy_model.py`: options/feature widths per kind, every kind learned and enabled on a
  clear signal, noise labels keep a kind disabled (`--rule ci`), MLP fit, `LearnedPolicy` answers
  every kind and reads the first model format, a rollout branch answers only its expected query,
  built-in option per kind, legacy target-record conversion.
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
cp build/fheroes2 ./fheroes2 && rl/.venv/bin/python -m pytest rl/tests -q
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
  pathological matchups. Dataset generator: `rl/gen_expert.py` (gzip JSONL).
- Expert-iteration result v1 (INVALID — see the encoding note in "Gate runner"): "~97%
  imitation accuracy" was an artifact. Checkpoints stay out of git (`rl/models/`).

## Expert data v3 and retraining (plan item 0, 2026-09-27)

- Real battles: `rl/harvest_battles.py` plays seeded games with the built-in AI (battle-agent
  channel answering `planner`) and stores every `battle_start` (+ `map`, `game_seed`):
  `rl/data/battles_<map>.jsonl` (training: Battlefi/Thechaos/2kings seeds 1-8 30 days, Arena
  1-8 20 days = 2285 battles, all with heroes, ~50 on castle/town tiles) and
  `battles_gate_<map>.jsonl` (held-out games, seeds 101-102: 545 battles). Rebuilt in the battle
  server by `engine_bridge.new_battle_from_setup` (shared with battle_agent.py).
- `auto`/`suggest` fixes (battle_server.cpp): (1) the built-in AI's action is reported as the
  EQUAL enumerated legal move (`builtinChoice`: attacks are compared after
  `Arena::resolveAttackCommand` fills in the target cell/direction the AI leaves at -1);
  (2) `auto` applies one command per decision (the AI planned "spell + attack" in one go and
  only the spell was recorded). Result: 1690/1690 expert actions are literally in the legal list
  (was 993/1003 plus mislabels: an in-place melee with dir -1 was encoded as a shot).
- Legal-move gap found by (1): a wide unit may strike from its TAIL cell (the AI does, the
  engine accepts it); `EnumerateLegalMoves` only tried neighbors of the head. Fixed (dedup by
  command). Encoding consequence: (target cell, dir) no longer identifies an attack (head strike
  from one position == tail strike from another), so the attack block is 99 x 13 sub-slots
  (6 head dirs, 6 tail dirs, ranged; `encoding.attack_parts`, shared with the transformer's
  13-way direction head): ACTION_SPACE 1460. `test_real_legal_moves_map_to_distinct_action_indexes`
  caught it.
- `rl/gen_expert.py --battles N --setups FILES`: random-army battles + harvested real battles;
  records keep `heroes`/`siege`, a `battle` key, exact expert move. v3 dataset
  (`rl/data/expert.jsonl.gz`): 1000 random + 2285 real battles -> 75 490 records in 20 s
  (108 skipped: retreats).
- `rl/train.py`: several `--data` files, `--val` (held-out share of BATTLES, hash of the
  `battle` key) with imitation accuracy (`exact` = argmax legal move == expert move, `slot` =
  same action slot) and value MSE, `--val-max`, `--threads 2`, line-buffered progress.
- `rl/gate.py --setups FILES`: paired gate on real battles — every (setup, side) is played by
  our MCTS vs built-in AND built-in vs built-in; verdict by (outcome, share of own creatures
  alive). Raw win rates on random armies are not paired (the armies differ per battle).
- Results v3 (Release, sims 32):
  - ResNet (`az_battle_expert_v3.pt`, 15 epochs, 67k train positions): held-out imitation
    exact 0.545 / slot 0.547, value MSE 0.207 — overfits (train policy CE 0.17, value 0.016).
  - Gate, random armies (100 battles, same armies): MCTS+ResNet 44% vs pure MCTS 25%.
  - Paired gate, held-out real battles (60 setups x 2 sides): pure MCTS better 3 / equal 73 /
    worse 44 (wins 40% vs built-in 48%); MCTS+ResNet 10 / 83 / 27 (48% vs 48%): matches the
    built-in AI's outcomes, loses more creatures. Not stronger yet.
  - Transformer `50m` (`az_battle_tr50m_expert_v3.pt`, 4 epochs, batch 32, lr 3e-4, ~35 min per
    epoch): held-out imitation exact 0.461 (ResNet 0.545), value MSE 0.208 (ResNet 0.207);
    cell CE 3.00 -> 1.92 -> 1.45 -> 1.04, value 0.27 -> 0.07, but the DIRECTION loss
    stalled (1.78 -> 1.65) near the marginal entropy of the direction labels (1.93) although for
    47% of the expert attacks exactly one direction is legal for the chosen target. Cause: this
    run trained the transformer WITHOUT legality masks (softmax over all 173 first-step tokens and
    all 13 directions), unlike the ResNet (masked policy) and unlike evaluate() (renormalizes over
    legal moves). Fixed afterwards: `build_transformer_samples` stores the legal first-step tokens
    (`cells`) and the legal directions of the target cell (`dirs`), the losses use masked logits
    (`legal_mask`). Retrain the 50m model with the masked loss before comparing it to the ResNet.
- Games vs the built-in AI (`rl/play_vs_builtin.py`, paired by seed: control game all built-in,
  treatment = one color played by our agent; battle units of the other colors move by the
  built-in AI taken from the replica's `suggest`, so the replica stays synced —
  `BattleAgentRunner(battle_color=...)`, `battle_agent.py/game_agent.py --color`). MCTS(32)+ResNet
  battles, built-in strategy: 2kings 30d seeds 1-4: 3 better / 1 equal / 4 worse, d_str -334
  [-1214, +437]; Arena 20d seeds 1-2: 3/0/5, d_str **-407 [-842, -43]**; + learned strategy on
  2kings: 3/1/4, d_str -226. 0 replica desyncs. Not stronger than the built-in AI; loses army.
- **transformers 5.x bug in forward_batch (found 2026-09-28):** `DynamicCache.batch_select_indices`
  filters IN PLACE and returns None; the code assigned its result, so the batched direction decode
  ran with NO cache (no board) in every transformer training so far, while inference decoded with
  it. Fixed + `test_forward_batch_direction_decode_matches_the_full_forward`. Every transformer
  checkpoint (incl. `az_battle_tr50m_expert_v3.pt`) must be retrained. `evaluate()` now uses
  masked softmaxes too (legal tokens, legal directions of the cell) — exactly what the masked
  losses and DPO train (`test_transformer_move_log_probs_match_the_mcts_priors`).
- DPO for the battle policy (`rl/train_dpo.py`, own loss — TRL's DPOTrainer is for token LMs):
  frozen reference copy, -log sigmoid(beta x [(log pi - log ref)(chosen) - (...)(rejected)]),
  log pi over legal moves exactly as MCTS priors (ResNet slots / transformer token x direction),
  optional RPO NLL term. Pairs: `rl/battle_prefs.py` walks real battles with the built-in AI and
  at sampled decisions compares the built-in move, the policy's top-2 and a random move by
  ROLLOUT to the end with the built-in AI on both sides — new battle-server `restore` flag
  `"rollout":1` (~3 ms per rollout, deterministic); score = outcome + own strength left - enemy
  strength left (per-unit `"str"` = monster strength x count, new state field). 2285 training
  battles -> 7597 pairs in 4 min (`rl/data/battle_prefs.jsonl`; expert chosen in 3477, rejected
  in 2017). Not trained yet (waits for the fixed transformer).
- Unified strategic output (user decision: no LLM; the same transformer answers the strategic
  queries with a masked softmax over the options): `rl/strategy_net.py` (query -> context token +
  one token per option from strategy_model's features, width `STRAT_TOKEN_W` = 61; SFT data from
  base games with the built-in answer; DPO pairs from strategy_rollout labels; `NetStrategyPolicy`),
  `transformer_model.strategic_logits` (causal body: [context, options, options], scores from the
  second copy so every option sees all options), `strat_proj`/`strat_head` (older checkpoints load
  with them fresh), `rl/train_strategy_net.py sft|dpo` with a battle-imitation anchor batch per
  step. Not trained yet. Existing labels give 329 pairs (Battlefi).
- Strategic DPO from our own games (2026-09-28, `rl/strategy_games.py`): one color is played by
  the unified net (`--strategy net --strategy-model <ckpt>`), for sampled queries the seeded game
  is replayed with the net's answers and branched (net answer / built-in answer / random option),
  the net continues after the branch. Label (`--label duel`, user design): after a WEEK, the
  score of our player's real hero-vs-hero battle in that week (branch AI log), else a DUEL in the
  battle server of our strongest hero vs the rivals' strongest (game_end now carries `top`:
  hid + save-game serialization + str), attacking and defending, built-in AI on both sides;
  score = outcome + own strength left - enemy strength left. 40 games (2kings 30d) -> 112 pairs
  in ~48 min (2 engines: the game + one duel server).
  - Build queries got a "let the built-in AI decide" option (`strategy_net.BUILTIN`, answered
    `skip`): the built-in building choice depends on inputs the query lacks, imitation of
    explicit buildings topped out at 0.81 even with the race build-order features
    (`build_priority_features`); with BUILTIN the SFT net reproduces the built-in AI on 100% of
    held-out queries and in games (20/20 equal pairs).
  - DPO barely moves the argmax: (1) the SFT net is certain (log p ~ -17 on non-built-in
    options) and DPO's loss vanishes at ~5/beta nats — `--label-smoothing 0.1` cut the gap to
    3.4 nats; (2) `--sft-data/--sft-weight` anchor 1.0 froze every choice (0.1 used); (3) the
    query tokens lack what decides a duel a week later (hero armies, rivals, game history):
    similar queries carry opposite labels, only 6/112 training pairs flipped. Paired games
    (2kings 30d seeds 101-110): SFT 0/20/0; DPO 6 better / 3 equal / 11 worse, d_heroes -0.55,
    d_str -195 [-702, +296]. The duel label favors "one strong army" (32/48 hire pairs prefer
    hiring nobody). Next: richer strategic context (hero armies, rival strength, history),
    a balanced label, more pairs.
- Strategic history (2026-09-28, user request "predict from the previous steps"): the strategic
  input is [previous days of the player (turn_context: day, resources, castles, heroes, army
  strength), previous decisions (kind + the chosen option's features), own heroes (army
  strength, move points, the query's hero), context, options, options] — token types one-hot
  (`STRAT_TOKEN_TYPES`, width 73), capped at 60 days + 400 decisions (window 2048).
  `attach_history` rebuilds it for SFT records (grouped by map/seed/player, ordered by n);
  `NetStrategyPolicy` keeps it during a game (`reset`, `decide`, `record`); `PolicyBranch` records
  the forced answer so a branch's history is exact; DPO pairs store the history they were made in.
  SFT with history: 100% held-out imitation (~6 min/epoch). With history DPO finally fits the
  pairs: loss 0.69 -> 0.29, the best option became the argmax in 90/111 pairs (6/112 without).
  Paired games (2kings 30d, seeds 101-110): 10 better / 1 equal / 9 worse, d_str -498
  [-1241, +178]; final duel (`play_vs_builtin.py --duel`: strongest-hero duel at game end,
  treatment - control) -0.66 [-1.53, +0.13], duel better in 10 pairs, worse in 7. Not yet a win.
- The duel label stays without hero/castle penalties (user: the effective strategy is ONE big army
  passed between heroes, so the objective is the same). Two horizons (user request): the label is
  the mean over `--horizons 7,14`; one replay per branch to the last horizon, the earlier state
  comes from the engine's `day_report` event (`FHEROES2_REPORT_DAYS=d1,d2`: before the first AI
  turn of those days every player's stats + strongest hero, as in game_end;
  `AIDecision::writeKingdomStats` now serves both).
- Round 2 (on-policy from the round-1 DPO model, horizons 7+14, 40 games -> 181 pairs, ~95 s per
  game; a game that ends before a report day falls back to game_end — a crash at seed 28 found it):
  DPO from the round-1 model: train loss 0.76 -> 0.16 but held-out preference accuracy 0.58 ->
  0.55 (chance): the labels do not generalize at this data size. Paired games (seeds 101-110):
  6 better / 5 equal / 9 worse, d_str -452 [-1228, +283], final duel -0.74 [-1.65, +0.10]
  (7 better / 8 worse). No gain over round 1. Candidates: far more pairs per round (overnight),
  several duel seeds per label to cut battle luck, fewer epochs / stronger anchor against
  overfitting, rival information in the input (the label is about the rival's strongest hero,
  the input only sees our side).
- Rival information = what a human player sees (user request 2026-09-28; from the adventure-map
  quick info, `dialog_quickinfo.cpp`): turn_context `rivals` lists enemy heroes on tiles OUTSIDE
  the fog of our color — color, tile, army as monster types with the size word's lower bound (1,
  5, 10, 20, 50, 100, 250, 500, 1000 = few ... legion), `est` = strength estimated from that; with
  full information (Kingdom IDENTIFYHERO / Crystal Ball) exact counts, level, attack/defense/
  power/knowledge, spell and move points, morale, luck. Never the spell book. `w` = map width.
  Fog is maintained for AI kingdoms too (2kings seed 3: rivals visible in 31/46 turn contexts).
  Network: `rival` token type (width 74): estimate, full flag, stacks, Chebyshev distances to the
  query's hero / nearest own hero / castle, skills only when fully known.
- Duels are fought with 3 battle seeds per orientation (user request): the duel score is the mean
  of 6 battles (`strategy_games.DUEL_SEEDS`).
- Result with rivals + 3-seed duels (SFT 100%, 40 games -> 183 pairs, DPO 10 epochs): held-out
  preference accuracy stayed 0.52-0.62 (0.62 before), paired games 3 better / 1 equal / 16 worse,
  d_str -700 [-1228, -151], final duel -0.59 [-1.62, +0.44] (7 better / 12 worse): DPO fits noise.
  Suspected cause: a different answer reshuffles the whole game's randomness, so a week later the
  branch-minus-baseline difference is mostly chance.
- War label (user design, 2026-09-28; `strategy_games.py --label war`, the default): one horizon of
  THREE weeks (21 days) after the query; +2 / -2 if our player won / lost the game by then; else
  the result of our hero battles against the STRONGEST active rival (by total army strength) in
  that window; else duels of our strongest hero against the strongest hero of EVERY active rival
  (3 seeds x attacking/defending each), the mean. Base games are 45 days (queries up to day 24).
  `play_vs_builtin.py --duel` scores the end of the game with the same rule (`war_score`).
- Continuous on-policy DPO (`rl/strategy_loop.py`, user request): collect 100 pairs with the
  current model -> DPO from it (it is the reference; SFT anchor 0.1, 4 epochs) -> paired games vs
  the built-in AI on seeds 101-110 -> `progress.jsonl` -> next round from the new model; `--resume`
  continues. Strictly one step at a time.
  First run (from the rivals SFT model, 100 NEW pairs per round, DPO only on them): round 1 1/19/0,
  d_str +13, duel +0.06; round 2 7/4/9, d_str -274; round 3 4/2/14, d_str -391; held-out
  preference accuracy 0.40 -> 0.24 -> 0.15 (below chance: each round fits its own 100 noisy pairs
  and drifts). Stopped in round 4. Now `--accumulate` (DPO on ALL pairs so far + `--extra-pairs`);
  restarted from round-1 model with the 406 old pairs (`rl/data/strategy_loop_acc`).
- Accumulating loop results (from the round-1 model + 406 old pairs): round 1 2/16/2, duel +0.11,
  held-out preference accuracy 0.55; round 2 7/2/11, d_str -29, 0.62; round 3 6/1/13, d_str -474,
  0.54. Slower drift, same direction; stopped. Suspected cause: the input lacked what decides a war.
- Extended strategic context (user request 2026-09-29, "what a human sees" + all previous steps):
  `turn_context` (AIDecision::sendTurnContext; wire format in rl/README.md) now carries the day of
  the week/week, own heroes in full (stacks with creature traits, level, primary/secondary skills,
  artifacts, spell points, morale, luck), own castles (buildings mask, garrison, creatures per
  dwelling level), rival heroes with stacks, and castles of other owners outside the fog with the
  quick-info rules (Thieves' Guild count / Crystal Ball decide what is seen of the defenders).
  Network (`strategy_net.py`): token types + `castle`, `rcastle` (STRAT_TOKEN_W 76; older
  checkpoints load with a fresh strategic projection), every token carries its day (DAY_SLOT);
  the prefix is the player's game in time order — per previous day [day, heroes, castles, rivals,
  rival castles, that day's answers] — then today's snapshot, context, options x2, trimmed from the
  oldest day to the model window. Window 512 (user decision; `50m` preset, `train_strategy_net
  --window`): a 45-day 2kings game needs <= 489 tokens (median 173).
  SFT (`rl/data/strategy_sft_2kings_v2.jsonl`, 60 seeds x 45 days, 12 934 queries):
  `rl/models/unified_sft_ctx.pt`, 100% held-out imitation after epoch 1; paired games 0/20/0.
- Training memory (the laptop swapped: 16.7 GB, 93 s/step for batch 32 x 490 tokens on the 50m
  model): `train_strategy_net.py` uses gradient checkpointing on the body (3.6 GB, 5.3 s for that
  batch; switched off around the battle anchor, whose decode needs the KV cache) and batches of
  similar history length (`length_batches`): ~11 min per SFT epoch, 5.6 GB process footprint.
- DPO loop on the extended context (`rl/data/strategy_loop_ctx`, fresh pairs only, accumulating):
  round 1 0/20/0 (no argmax moved), 2 7/0/13 d_str -508, 3 9/2/9 d_str -487, 4 9/1/10 d_str -269
  duel -0.72; held-out preference accuracy 0.55 -> 0.48 -> 0.61 -> 0.655. Stopped after round 4.
- **Label noise measured (`rl/label_noise.py`, 2026-09-29):** engine `FHEROES2_RESEED=day:salt`
  re-seeds the game RNG and shifts the world seed (battle luck) before the first AI turn of `day`
  (the game before is byte-identical; tested). 10 games (seeds 301-310) x 3 queries x (built-in
  answer, random answer) x (plain + 4 salts), war label at 21 days: 1/3 of the answers change
  nothing; otherwise the luck spread of ONE answer (sd 0.82) equals the mean difference between
  answers (0.80), only 6/20 queries differ clearly after 4 replays, and the plain one-replay label
  has the sign of the salt-averaged difference in 8/16 — a coin flip. Paired luck does not cancel
  (corr of the two answers over salts -0.2: another answer reshuffles the RNG stream). Army
  strength as the target is no better (12/19). Battles themselves are seeded by map seed + armies
  (`computeBattleSeed`), not by the game RNG. Conclusion: the one-replay DPO labels were noise;
  next: a strategic value network (user decision) instead of single rollouts.
- Unified 100m on the server (2026-09-30/10-01, CPU, 8 threads): 1 epoch without battle history —
  value MSE 2.26 (mean 3.34), corr 0.57, battle imitation 0.076 (battle loss flat at 4.9: one battle
  batch per strategic step = 11k of 67k positions per epoch); 3 epochs WITH battle history
  (`unified_100m_hist.pt`) — battle imitation 0.18 / 0.24 / 0.28, value val MSE 2.43 / 2.56 / 2.79
  while its train loss fell 3.06 -> 1.22 (overfits ~500 games). Strategy 100% incl. the 79
  non-majority queries from epoch 1 (majority baseline 0.960: build/target always option 0, army 2,
  hire 2 in 83%). DPO loop from the 1-epoch model (war label, accumulating): rounds 7/0/13,
  5/4/11, 9/1/10, every CI across 0 — no gain.
- Final duel label (user design 2026-10-01; `strategy_games.final_label`, `--label final`, the
  default of strategy_games/strategy_loop; also play_vs_builtin --duel and the value data): at the END
  of the game (DPO branches are played to the last day) +1 won / -1 lost, else our strongest hero
  vs the strongest active rival's strongest hero, attacking and defending x 3 seeds: all won +1, all
  lost -1; a mixed result is no clear victory — forces are added to the side that won less (battle
  server `new` op `ascl`/`dscl`: every stack of a side at that % of its count, rounded, >= 1; applied
  at every rebuild without accumulating) and the duel is fought again, a 4-step bisection of
  log2(our army / theirs) for the even point; label = -that, in [-1, 1]. The value head predicts
  the same label (`STRAT_VALUE_SCALE` 2 -> 1; value data must be regenerated with it).
- Final-label noise (`label_noise.py --label final`, seeds 301-330, 90 queries x 2 answers x 5
  replays): 41/90 answers never change the label; of the 27 non-zero plain labels the mean of 4
  other-luck replays has the same sign in 12, the opposite in 6, 0 in 9 (war label: 8/16); a single
  other-luck replay repeats the plain sign in only 28%; corr of the two answers over luck 0.55 (war
  -0.2). DPO loop on single-replay final labels (`rl/data/strategy_loop_final`): rounds 3-7 worse
  than the built-in AI (d_str CIs below 0); from round 8 every round collects with / starts DPO from
  model_r6 (`--base-model`) and from round 10 the paired games are against r6
  (`--eval-opponent`; `play_vs_builtin --opponent-model`, `strategy_policies.PerColor`): r8-r11 vs
  r6 10/10, 11/9, 11/9, 11/8 with every duel CI across 0 — no gain.
- Luck-averaged labels (user design 2026-10-03): `--salts K` (strategy_games, strategy_loop) — every
  answer is replayed K times (salt 0 plain, salt k re-seeded from the day after the query), the
  label is the mean of the per-luck differences branch - baseline (`query_scores`; pairs keep
  `salt_scores`). The final duel uses `DUEL_SEEDS` = 5 battle seeds per side (was 3).
  First loop (`rl/data/strategy_loop_avg`, 5 lucks, base r6, vs r6): 7/3/10, 11/0/9, duel CIs across
  0; only 35% of the single lucks agree with their pair's sign, 15/134 pairs have a mean gap > 2 SE.
- Plan of 2026-10-03 (user): (1) DPO only on reliable pairs — `train_strategy_net dpo --min-z`,
  `strategy_loop --min-z` (`reliable_pairs`: mean per-luck gap > z standard errors); (2) 10 lucks per
  answer; (3) a twice larger model, `200m` preset (hidden 1024, 14 layers, 16/4 heads of 64, SwiGLU
  4096: 218.9M), trained from scratch on ALL games: the value data, new seeds, and the DPO games —
  `PolicyBranch` records every player's days + answers (the other players' = the built-in answer),
  `strategy_games.value_trajectories` labels them with `final_label`, the loop appends its base
  games to `value_rN.jsonl`, and `strategy_value.py replay --model M --seeds S` re-plays past DPO
  base games (deterministic: same seed + model = the same game, checked); (4) DPO evaluated against
  the built-in AI again. `train_strategy_net` keeps the best held-out value MSE as `<out>_best.pt`.
- DPO collection profile (2026-10-05, `rl/profile_loop.py`: one label_game with the loop's settings,
  200m model, 2 torch threads): 97% of the wall time is the strategic forward pass (877 ms per query,
  94 queries per 45-day game), the engine 3%, the final-label duels ~0. A round (8 queries x ~25
  replays x 10 lucks per game) took ~14 h. Fix: `NetStrategyPolicy` inference caches — whole answers
  by their exact input (every replay repeats the base game up to the branch point; salt 0's baseline
  IS the base game) and the prefix key/values (`StrategicPrefixCache`, `strategic_logits_cached`):
  the prefix is always computed in fixed 16-token chunks keyed by the hash of all tokens up to the
  chunk's end, so the logits never depend on the cache state (cold == warm, bit-identical; tested).
  Same game: 272 s -> 107 s, output digest identical to the uncached run. What is left is ~0.27 s
  per body call: on CPU a 200m forward of a few tokens is bound by reading the weights.
- Hero label (user design 2026-10-06: "more strength for the final duel", the hero counted with
  skills and magic, the army counted too): `strategy_games --label hero`, `strategy_loop --label
  hero --hero-rule hero|army|mean|agree`, `label_noise --label hero`. Per luck, at the end of the
  game, two parts in log2 units (positive = the branch is stronger), both stored in the pairs
  (`salt_parts`): hero = how much less army our strongest hero needs to win half of the duels against
  a FIXED reference (`hero_equivalent`: bisection of our army scale in [1/8, 8], 6 steps, 5 seeds x
  both sides) — the reference is the strongest rival hero of the BASELINE of the same luck, so the
  rival's luck in the branch does not enter (the two versions of our hero cannot fight each other:
  same hero id = the same world object in the battle server); won game = -3, lost / no hero = +3,
  a rival without heroes = -3 (`equivalent_of`). army = log2 of the strongest heroes'
  `Army::GetStrength` (troops with the hero's attack/defense, morale, luck — not magic). First real
  game: the parts can disagree (no hire: +0.87 army, but the baseline with the hire won the game).
- Hero-label loop round 1 (`rl/data/strategy_loop_hero`, `--hero-rule mean`, 10 lucks, min-z 2):
  12/67 reliable pairs (army part alone 15, hero part alone 8 — the duel-equivalent is the noisier
  part), paired games 8/2/10, d_str -90 [-603, +428], duel +0.09 [-0.31, +0.51]. Then (user choice
  2026-10-06) `--reliable-rule army_hero` (train_strategy_net, strategy_loop): the army part decides
  (mean gap > min-z SE), the hero part must not be against it (mean gap >= 0); every ordered pair of
  the query's scored options is tried, the clearest army gap wins and orients the pair. With the hero
  label strategy_games keeps every query with two scored options (no --margin; the rule filters).
- Hero-label loop rounds 2-4 (army_hero rule): 17 / 12 / 19 reliable pairs per round, paired games
  8/3/9 d_str -369, 9/1/10 d_str -189, 5/1/14 d_str **-1034 [-1711, -344]** duel -0.55 [-1.11, +0.03];
  DPO loss 0.55 -> 0.39, held-out preference accuracy 0.33-0.57: more pairs made the net WORSE, as
  in every earlier loop. Stopped 2026-10-07 (user decision) for the oracle headroom test:
  `rl/oracle_headroom.py` — built-in games, per sampled query the built-in answer + 3 random options
  x 20 lucks; the oracle picks by lucks 1-10, its gain is measured on the fresh lucks 11-20 (the
  selection bias of "best of noisy estimates" stays out); rules mean / army / army_hero.
- Oracle headroom result (120 queries, 40 built-in games, 4 options x 20 lucks): lax rules deviate in
  ~half the queries with an apparent gain of +0.56..0.65 log2 on the selection lucks and ~0 on fresh
  lucks (mean +0.03 [-0.05, +0.12], army 0.00 [-0.09, +0.09]) — pure selection on noise, what the DPO
  loops learned. The strict army_hero rule deviates in 10/120 (8%): fresh army up in 8/10 (~x1.4 per
  deviation), +0.04 [+0.01, +0.08] per query overall — real but rare (no hire, army budget < 100%,
  another hero target). Win/loss rates unchanged.
- Deviations (`rl/deviation_stats.py`, 10 seeds x 2 colors, 30 days): hero-loop model_r4 leaves the
  built-in answer in 29 decisions per game (~40%; hero targets 187/208), median probability margin
  over the built-in answer 0.07; the SFT base 0. Confidence gate (`NetStrategyPolicy(min_margin)`,
  `play_vs_builtin --strategy-margin`): r4 ungated 5/1/14 d_str -1034; margin 0.3 7/2/11 d_str -940
  [-1487, -414]; 0.5 11/1/8 d_str -191 [-740, +288] duel +0.21 [-0.30, +0.71]; 0.8 8/3/9 d_str -189
  [-640, +208] duel +0.21 [-0.24, +0.64]. The gate removes the harm, no significant gain yet.
  40 seeds (101-140, 80 pairs; `rl/data/gate40`): margin 0.5 35/4/41, d_str -335 [-666, -7], duel -0.02
  [-0.26, +0.23]; margin 0.8 32/9/39, d_str -170 [-478, +136], duel +0.06 [-0.15, +0.29]; both ~-0.75
  heroes and ~+1000 unspent gold. The 10-seed duel plus was luck. `strategy_loop --gate-margin` exists
  (collection + paired games) but was not run: nothing to gain at this signal level.
- Group-advantage update (`train_strategy_net pg`, the GRPO idea without selecting pairs; 2026-10-07):
  all 336 hero-loop queries with >= 2 scored options, loss -sum_i adv_i log pi(i) + kl x KL(ref||pi),
  4 epochs from the 200m base. Held-out mean advantage of the argmax 0.053 (base) -> 0.049 (kl 0.1;
  the loss runs away to -3.2 by pushing negative-advantage options to log pi -> -inf) / 0.053 (kl 1.0,
  argmax unchanged). 40 seeds: kl 0.1 24/2/54, d_str ~-640 (both halves' CIs below 0), duel ~-0.51
  (CIs [-0.85, -0.13], [-0.87, -0.22]) — clearly worse; kl 1.0 37/5/38, d_str ~-140, duel ~+0.03 —
  a tie (it barely leaves the base). Not better than DPO: the signal per query is the bottleneck.
- Plans instead of single decisions (user decision 2026-10-08): whole-game rules on top of the
  built-in AI, evaluated by paired games (`play_vs_builtin`, 40 seeds 101-140, both colors, 30 days).
  Python rules (`strategy_policies.RulePolicy`, `--strategy rule --rule ...`): army=0 30/3/47 d_str
  -449 [-757, -126] duel -0.50 [-0.73, -0.26]; army=50 35/5/40 d_str -72, duel -0.03; hire_after=15
  33/3/44 d_str -428 [-778, -74]; hire_after=8 35/2/43 d_str -191, duel +0.01; max_heroes=1 35/2/43
  d_str -74, duel -0.22; max_heroes=2 31/6/43 d_str -31, duel +0.13 [-0.06, +0.31]. Nothing better.
- Engine plans (`src/fheroes2/ai/ai_plan.{h,cpp}`, `FHEROES2_PLAN="color=Blue,champion=1,
  secondary_min=1"`, off by default; `StrategyEnv(plan=...)`, `play_vs_builtin --plan`, applied to our
  color only; the control games never get it). The user's rules (2026-10-08): one main hero that fights
  most battles; secondary heroes collect resources and dwelling troops, carry minimal armies (one
  monster of a FAST BUT WEAK kind: the fastest of the weaker half), hand armies over in chains;
  single-creature stacks to soak retaliation strikes. Upstream assigns a Champion/Courier only with
  more than three heroes, so on 2kings (2-3 heroes) none of it happens. Implemented: `champion=1`
  (the strongest army becomes Champion and stays while it lives; every other hero a Courier, which
  falls back to Hunter when it has nothing to carry), `secondary_min=1` (`AIPlan::handOverArmy` on
  meeting the champion and on visiting an own castle — the secondary leaves its troops in the
  garrison and takes none). Pitfall found: the castle turn of day 1 runs before the first role
  assignment — "secondary" must mean "not the champion the plan already chose" (else the starting
  hero dumped its army into the garrison). Turn contexts carry each hero's AI `role` (0 scout,
  1 courier, 2 hunter, 3 fighter, 4 champion). Seed 7 trace: the champion had 907 strength on day 10
  (136 without the plan), but the emptied castle fell on day 22.
- Plan sweeps (40-80 seeds, both colors, 30 days; d_str / duel / outcome with 95% CIs):
  champion=1 on 120 seeds (240 pairs): 114/2/124, d_str -254 [-445, -65], duel -0.08 [-0.23, +0.07],
  outcome +0.03, castles +0.05 — no gain. Seeds 101-180 (160 pairs): A champion+secondary_min+
  garrison_slowest 69/1/90, duel **-0.57 [-0.75, -0.39]**; B champion+champion_skills 78/1/81, d_str
  -186 [-403, +30], duel +0.08 [-0.10, +0.26], outcome +0.03 (best, not significant); C primary_castle+
  secondary_guild (RulePolicy builds) 68/6/86, d_str -256 [-491, -18], outcome -0.10 [-0.19, -0.01];
  D champion+skills+garrison_slowest+castles 69/3/88, duel -0.18 [-0.35, -0.01]. The minimal-army rule
  hurts the final duel badly even with a garrison kept; the castle rule loses castles. Then
  `secondary_skills=1` (estates first, user rule).
- Long games and a 6-player map (`rl/data/plans6`; A = the OLD secondary_min without the champion's
  return): 2kings 45d A 33/35/92 outcome -0.39 [-0.52, -0.26]; B 70/22/68 duel +0.06, outcome +0.07;
  E 64/25/71 ~0. 2kings 60d A outcome -0.61; B duel +0.04; E -0.08. Battlefi 30d (20 seeds x 6 colors):
  A castles -0.80; **B d_str +599 [-39, +1282], duel +0.15 [+0.00, +0.29]** (first borderline gain),
  castles -0.18; E duel +0.08. Diagnosis of A (seeds 101-103 traced): the champion never came back for
  the troops the secondaries left in the castle (built-in fighters value an own-castle visit at half
  and ignore it below 500) — fixed (`secondary_min` makes the champion value it fully, threshold
  100); secondaries with one monster die to the rival's main hero (feeding it experience) —
  `secondary_min=2` hands over only on meeting, in a castle the secondary takes the garrison like a
  built-in courier and carries it to the champion.
- Battlefi re-run (`rl/data/plans7`, 40 new seeds x 6 colors): B (champion+champion_skills) duel +0.10
  [+0.03, +0.17]; over all 60 seeds (360 pairs) **duel +0.12 [+0.05, +0.19]** — the first significant
  gain over the built-in AI — but castles -0.33 [-0.47, -0.18]. G (champion+skills+secondary_min=1
  with the champion's return+garrison_slowest) duel -0.12 (Battlefi) / -0.43 (2kings); H (secondary_min
  =2) ~0. Then `secondaries=1` (the non-champions keep the built-in fighter/hunter roles — the
  suspected cause of the lost castles: all-courier secondaries capture none).
- Factorial sweep (user remark 2026-10-09: "some rules only work together"): `champion=1` always on
  (the other keys depend on it), five binary keys S champion_skills, R secondaries=1, M
  secondary_min=2, E secondary_skills, G garrison_slowest in all 32 combinations, Battlefi 30d seeds
  101-140 (`rl/data/factorial`, tags `_f<letters>_<seeds>`). `rl/plan_factorial.py` fits main
  effects + pairwise interactions (factors -1/+1, effect = 2 x coefficient) with a seed-cluster
  bootstrap — every pair informs every effect, so far more power than one-rule-at-a-time sweeps.
- Two strategies (user, 2026-10-09): **outcast** — one main hero whose goal is to beat the rival's main
  hero — and **standard** — capture every castle and kill every enemy hero. The current work trains
  OUTCAST: the final duel is the primary metric, castles/outcome secondary. Interim factorial (4980
  pairs, 21/32 cells): R secondaries=1 castles +0.37* but duel -0.07* (a standard-strategy rule);
  E duel -0.05*; S str +140*; RE duel -0.04*, SR str -169*; best cells fS / fG duel ~+0.11.
- Factorial final (7680 pairs, 32 cells, Battlefi 30d seeds 101-140), duel effects: champion (mean
  over cells) +0.047 [+0.013, +0.081]; G garrison_slowest +0.028*; S 0.00 (str +152*); M -0.03;
  E -0.030*; R -0.084*; RE -0.045* (estates hurt only with built-in-role secondaries: with courier
  secondaries E ~ +0.015 — the user's "estates for non-fighting secondaries"); SR str -143*. R
  castles +0.37* (a standard-strategy rule). Best cells: fSEG +0.152, fSMEG +0.120, fEG +0.119, fS
  +0.109, fG +0.108; every R cell is at the bottom. Confirmation on fresh seeds 141-200 (fSEG vs f,
  `rl/data/confirm`) against the winner's curse of picking the best of 32.
- Confirmation on fresh seeds (Battlefi 30d, 141-200, 360 pairs each, `rl/data/confirm`): f (champion
  only) duel **+0.074 [+0.006, +0.143]**, castles -0.40; fSEG duel **+0.079 [+0.004, +0.151]**, castles
  -0.33; fSEG - f on the same games +0.005 [-0.063, +0.073]. The champion rule is a real outcast gain
  (also +0.047 over the factorial); the extras S/E/G are not proven — the best-of-32 cell's +0.15 was
  the winner's curse. Outcast plan core = `champion=1`; test new rules paired against it.
- Plan keys added 2026-10-10: `chains=1` (troops travel to the champion as a relay: a courier with
  cargo goes to the champion if it reaches him this turn, else to the own hero it reaches this turn
  that stands >3 tiles closer to him; AIMeeting between two secondaries gives the army to the one
  closer to the champion); `split_singles=1` (`AIPlan::BattleSplit` in Battle::Loader after the
  pre-battle ordering: the champion's weakest stack is split into single monsters in the free slots
  — upstream's `splitStackOfWeakestUnitsIntoFreeSlots`, which the AI only uses from Hard difficulty;
  the playtests run Normal — and merged back after the battle; `BattlePlanner::decoyTarget`: a single
  stack of the champion attacks the strongest enemy it can reach that has not retaliated yet).
  Trace (Battlefi seed 141): singles in 5/12 champion battles (all 5 slots are often full).
- `az/` was renamed to `rl/` (user request; the venv moved with it).
- Next (user request 2026-09-28): predictions conditioned on the PREVIOUS steps — history tokens
  before the current state (battle: previous actions of this battle; strategy: previous decisions
  of the player), the reason for the 2048 window. MCTS must pass main line + search path as history.
- Test-suite flakiness under load: the Debug integration run failed 3 fixture setups once
  while a Debug build and the 50m training ran at the same time (60 s protocol watchdog); the
  same suite passes on an idle machine (174/174 Debug and Release). Do not run the suite next to
  a build/training when judging failures.
- `rl/engine_bridge.py` reads replies with a byte-level line assembler and a hard 60s cap
  per reply: the engine can hang mid-line inside the planner, so a plain readline() is not
  enough. All writes are bytes (`text=False`, `bufsize=0`).
- Tests live in `rl/tests/` (pytest, run with `rl/.venv/bin/python -m pytest rl/tests -q`).
  Unit tests use a fake environment; the protocol integration test requires `./fheroes2`
  and is skipped when the binary is missing.
