# AlphaZero-style battle engine for fheroes2 (prototype)

Goal: an AlphaZero-like system (self-play + MCTS + policy/value neural network) for
**Heroes II battles**. Battles are chess-like: perfect information, deterministic given
(armies, tile, seed, commands), a compact 11x9 board and a small action space. This is the
foundation; the strategic layer will reuse the trained networks as value/policy priors later.

Architecture (Leela Chess Zero style, Stockfish-inspired engine discipline — **no code reuse**:
Stockfish/lc0 are GPL-3, this repo is GPL-2):

```
+------------------------------------------------------+
| fheroes2 engine (C++, this repo)                     |
|   Battle::Arena — deterministic battle simulation    |
|   battle_server: JSONL on stdin/stdout               |
|     ops: new / action / reset / quit                 |
|     replies: state (+ legal moves) or result         |
+-------------------------+----------------------------+
                          | pipe
+-------------------------v----------------------------+
| rl/ Python prototype                                 |
|   engine_bridge.py — Gym-like environment            |
|   encoding.py      — state/action -> tensors         |
|   mcts.py          — PUCT search (pure MCTS first)   |
|   selfplay.py      — generate training games         |
|   train.py         — PyTorch policy/value network    |
+------------------------------------------------------+
```

## Protocol (v0)

Request line (Python -> engine, JSONL):

```json
{"op":"new","seed":42,"att":"13x10,21x24","def":"22x5,40x8"}
{"op":"action","act":1,"args":[4,25,-1,12,3]}
{"op":"reset"}
{"op":"quit"}
```

- `new`: seed (uint32), stacks as `monsterIdx x count` CSV. The engine picks a deterministic
  land tile from the loaded map (obstacles/terrain come from the tile). Real-battle replication
  extras: `tile`, army slots (`"0:13x30,2:21x25"`), `wseed`, formations `sat`/`sdf`, army
  colors `acol`/`dcol` (`PlayerColor` values, neutral = 0; default attacker RED, defender
  BLUE) and commanders `ahid`+`ahero` / `dhid`+`dhero` (hero id + hex save-game serialization
  from `battle_start`; the hero is restored into the world's hero with that id — the engine
  must load the same map — and that side fights with the hero's own army, its stacks are
  ignored), the castle/town on the tile `castle` (hex serialization from `battle_start`) and
  `dgar:1` (its garrison defends). A commander or castle that cannot be restored answers
  `{"ev":"error","what":"bad battle setup"}`; undecodable commander hex means "no commander".
  The server keeps every hero of the map off the map tiles except the restored commanders. All players of the loaded map are
  AI-controlled, as in the autonomous playtest (control affects the bad-morale roll).
- `action`: `act` is `Battle::CommandType` (0=MOVE,1=ATTACK,8=SKIP, ...), `args` are the
  `Battle::Command` values as stored — in REVERSE constructor order (see `battle_command.h`):
  MOVE `[dst, uid]`, ATTACK `[direction, targetCell|-1, cellToMoveFrom|-1, defenderUID,
  attackerUID]`, SKIP `[uid]`. Copy moves from `legal` rather than building them by hand.
  Only the *current* unit may act; a command the engine would reject (wrong unit, not a valid
  MOVE/ATTACK/SKIP at this point) is answered with `{"ev":"error","what":"illegal action"}`
  and not applied (same for `replay`/`restore` paths).
- `reset`: rebuild the battle from the last `new` setup (replay-based MCTS).
- `replay` (batched path on top of the main line) accepts `"full":1` to force the replay from the
  battle root; by default the engine restores its snapshot of the main-line end (same result).

Reply line (engine -> Python):

```json
{"ev":"state","turn":3,"cur":7,"units":[...],"obstacles":[...],"legal":[...],"result":null}
{"ev":"result","winner":"att"}
```

- `cur`: UID of the unit expected to act, or -1.
- `units`: `{"u":uid,"side":"att"|"def","mon":id,"q":count,"hpl":hpOfTopMonster,"i":headCell,
  "ti":tailCell|-1,"sp":speed,"shots":n,"moved":0|1}`.
- `legal`: list of `{"act":..,"args":[..]}` for the current unit (MOVE to each reachable cell,
  ATTACK targets from reachable cells or as a shooter, SKIP, then the commander's SPELLCASTs:
  one per spell and valid target) — exactly the commands the engine accepts (filtered through
  the engine's own validation).
- `heroes`: `{"side":..,"sp":spellPoints,"cast":0|1}` per side with a commander; `siege` (sieges
  only): `{"cells":[[cell,wallState],..],"towers":[left,center,right] (1/0/-1 not built),
  "bridge":0 up|1 down|2 destroyed}`.
- `result`: `att`/`def`/`draw` once the battle is over; `cur` is -1 and `legal` is absent.

MCTS needs state restore; v0 uses **replay from the root** (`reset` + repeated `action`),
which is exact because the engine is deterministic. A C++ snapshot/restore (make/unmake
analogue) is planned once the loop is proven.

## Phases

1. **Loop proof (pure MCTS, no NN)** — DONE: bridge + PUCT search with a material-strength
   evaluation; ~5 s per battle (sims=16) after the batched replay op.
2. **Encoding + network** — DONE (v0): 11-channel plane stack + 9 scalars (turn, unit counts,
   per side: commander present / spell points / cast this round); fixed 1460-slot action space
   (99 MOVE / 1287 ATTACK = 99 targets x (6 head + 6 tail strike directions + ranged) / 1 SKIP / 73 SPELLCAST by spell id — all targets of a spell share its
   slot and split its probability; the transformer has the same 73 spell tokens in its first
   decoding step); 4-block ResNet (~250k params);
   `rl/train.py` trains on self-play records and saves `rl/models/az_battle_v1.pt`.
   The protocol is strictly stateless now: every op (new/action/replay/reset) answers
   immediately, the engine replays the main line from the root.
3. **AZ training loop** — DONE (v0):
   - Expert warm-start: the dataset is generated from the ready-made algorithms — the battle
     server's "auto" op plays battles with the built-in BattlePlanner and streams (state,
     expert action) pairs (`rl/gen_expert.py`, 400 battles -> 7.5k records in ~2 s; policy CE
     0.77 -> 0.027, i.e. the net imitates the built-in AI per move with ~97% accuracy).
   - The trained net guides the search: policy priors + value head inside MCTS, Dirichlet
     noise at the root (`rl/selfplay.py --model rl/models/az_battle_expert_v1.pt --device mps`).
   - Transformer variant (stage 3.5): `rl/transformer_model.py` — a policy/value transformer
     on the ready-made HuggingFace Qwen3 body (~1.0M params, cell+direction action decoding
     with a KV cache), trained by `rl/train.py --arch transformer`; see AGENTS.md
     "Transformer architecture (stage 3.5)".
   - Search infrastructure: MCTS node states materialize via battle-server snapshot/restore
     (C++ `ArenaSnapshot`, ops `snap`/`restore`; ~3.3x faster than replay-from-root at depth
     30, visit counts verified identical), and `rl/gate.py` measures the win rate of our
     engine vs the built-in BattlePlanner (via the `suggest` op).
   - Known gaps: ~half of the expert records are skipped because the v0 legal-move
     enumeration is narrower than the planner's real options (spells, catapult, some attack
     cells); per-leaf inference is not batched.
4. **Integration**: the trained net + MCTS replaces `AI::BattlePlanner` in real games
   (on-demand battle solving); later, strategic layer value function. First step done:
   the battle-agent channel (see "Real-battle integration" below).

## Strategic layer

The strategic decision protocol delegates the kingdom-level choices — hero targets, castle
building, hero hiring and the army budget — to an external agent while the engine keeps all
mechanics (pathfinding, movement, battles, economy, army composition). Enabled together with the
autonomous playtest mode:

```sh
python3 rl/strategy_run.py --policy greedy --playthroughs 1 --days 10 --map 2kings.mp2
```

Engine -> agent (stdout, JSONL):
- `{"ev":"turn_context","t":..,"p":color,"diff":..,"res":[wood,mercury,ore,sulfur,crystal,gems,gold],
  "castles":[{"n":..,"i":..}],"heroes":[{"id","i","mp","mmp","str"}]}` — at the start of each AI turn;
- `{"ev":"decision","t":..,"p":color,"h":heroId,"from":tile,"cands":[{"i":tile,"obj":type,"v":value,"d":dist}, ...]}`
  — one per hero activation; candidates are all positive-value targets as evaluated by the
  built-in strategic AI (already sorted by value).

- `{"ev":"build","t":..,"p":color,"castle":tile,"race":r,"defensive":0|1,"res":[7],"cands":[{"b":buildingBit,
  "name":..,"trade":0|1,"cost":[7]}, ...]}` — castle development (end of the kingdom turn, per
  castle, only when something can be built): every building the AI may build now, directly or
  after a marketplace trade (`trade:1`). Followed (also without a query) by
  `{"ev":"build_result","t":..,"p":..,"castle":tile,"b":builtBits|0,"src":"agent"|"builtin"}`;
- `{"ev":"hire","t":..,"p":color,"heroes":n,"res":[7],"cands":[{"castle":tile,"slot":1|2,"hero":id,"race":r,
  "lvl":..,"val":recruitValue,"army":castleArmyValue}, ...],"bi":index|-1}` — whenever the AI
  considers hiring and can afford a hero: (castle x tavern offer) candidates and the built-in
  choice `bi` (-1: the built-in AI would not hire, e.g. at its soft hero limit).

- `{"ev":"army","t":..,"p":color,"castle":tile,"reason":"defense"|"visit"|"hire","guest":heroId|-1,"garrison":str,
  "hero":str,"res":[7],"offer":[{"mon":id,"avail":n,"n":affordable,"str":strength}, ...]}` — before the AI
  hires monsters in a castle (castle under threat / a hero visits / right after hiring a hero),
  only when some monster is affordable.

Agent -> engine (stdin):
- `{"op":"pick","h":heroId,"i":tile}` — must be one of the candidates, otherwise ignored
  (the built-in choice is used);
- `{"op":"build","castle":tile,"b":bit}` (must be a candidate) / `{"op":"build","b":0}` (build
  nothing, keep the resources);
- `{"op":"hire","castle":tile,"slot":1|2}` / `{"op":"hire","castle":-1}` (do not hire);
- `{"op":"army","pct":0..100}` — the share of the kingdom's resources this castle may spend on
  monsters; the built-in AI picks what to hire within it (100 = the built-in purchase exactly,
  including troop upgrades; below 100 upgrades are skipped, 0 = save everything);
- `{"op":"skip"}` — keep the built-in choice (answers any query);
- `{"op":"quit"}`.

After each playthrough the engine reports `{"ev":"game_end","playthrough":..,"day":..,
"results":[{"c":color,"s":state,"d":day,"k":castles,"h":heroes,"str":armyStrength,"g":gold}]}`. The channel is self-healing: if the agent dies or
sends garbage, the engine falls back to the built-in AI permanently.

In Python a policy is `policy(decision)` for targets plus optional `build(ev)` / `hire(ev)`
methods returning a candidate, `strategy_policies.NOTHING` or None (built-in), and `army(ev)`
returning a budget percent; `strategy_policies.strategic_reply`
turns that into the reply and a record (`kind` target/build/hire/army).

`rl/strategy_run.py` records every decision with the final game outcome attached
(`rl/data/strategy_<policy>.jsonl`) — the training data format for the strategic value network.
Policies (`rl/strategy_policies.py`): `greedy`, `random`, `builtin` (always skip) and `tempo` —
value/distance-aware: discounts a candidate by `gamma` per extra turn of travel (from the hero's
move points in `turn_context`) and penalizes tiles another hero already claimed this kingdom
turn. The built-in value already folds in the distance, and early-game candidates are usually
all within one turn (2kings, week 1: `tempo` == `greedy`), so the difference shows only with
long-range candidates and several heroes.

### Paired benchmark (`rl/strategy_bench.py`)

```sh
rl/.venv/bin/python rl/strategy_bench.py --policy tempo --map 2kings.mp2 --days 14 --seeds 8 --jobs 4
```

`FHEROES2_AUTO_PLAYTEST_SEED=<n>` re-seeds the engine's random generator before every playthrough
(`seed + playthrough id`), so equal seeds and equal agent choices replay byte-identical games.
The benchmark plays, per seed, one control game (everybody on the built-in AI) and one game per
color where only that color uses the policy (`ForColor`); treatment and control diverge only
through the policy's choices. Per (seed, color) the verdict is better/equal/worse by
(outcome, castles, army strength) from the `game_end` stats; the report goes to
`rl/data/bench_<policy>_<map>_<days>d.json`. Sanity check: `--policy greedy` must give 100%
`equal` with 0 overrides (the built-in choice is the top-value candidate).

Event fields for this: `turn_context`/`decision` carry `"p":"<Color>"` (same names as
`game_end` results); every `game_end` result carries `k` (castles), `h` (heroes), `str` (army
strength of heroes + garrisons) and `g` (gold).

### Learned policy from counterfactual rollouts (experiment)

```sh
rl/.venv/bin/python rl/strategy_rollout.py --map Battlefi.mp2 --seeds 101-120 --per-seed 25 --top 4 --horizon 7 --jobs 2
rl/.venv/bin/python rl/strategy_model.py --data rl/data/strategy_rollouts_Battlefi_h7.jsonl --out rl/models/strategy_model.json
rl/.venv/bin/python rl/strategy_bench.py --policy learned --model rl/models/strategy_model.json --map Battlefi.mp2 --days 30 --seeds 10
```

Seeded games are deterministic, so the value of picking candidate j at decision n is measured
exactly: replay the game, pick j at n, play H more days, diff the player's stats against the
built-in branch. A small advantage model (ridge/MLP, JSON) is trained on these labels. First
result: significantly more army strength but slightly fewer castles — not a net win yet (see
AGENTS.md). Keep `--jobs` low (2) on a laptop; engines run under `nice`.

### Everything from models: `game_agent.py --strategy learned`

```sh
rl/.venv/bin/python rl/strategy_model.py --data rl/data/strategy_rollouts_all_Battlefi_h7.jsonl --out rl/models/strategy_model.json
rl/.venv/bin/python rl/game_agent.py --strategy learned --battle mcts --sims 4 --map Battlefi.mp2 --days 7
```

The learned strategic policy answers hero targets, building, hiring and army budgets; battles go
through MCTS in the headless replica (after 30 rounds the built-in AI finishes a battle — engine
battles have no round limit). `strategy_model.py --rule all` (default) answers every kind;
`--rule ci` keeps only kinds with a convincingly positive cross-validated gain.

### One agent for both channels (`rl/game_agent.py`)

```sh
rl/.venv/bin/python rl/game_agent.py --strategy tempo --battle mcts --sims 16 --map 2kings.mp2 --days 7
```

Spawns the engine with `FHEROES2_STRATEGY_SERVER=1` and `FHEROES2_BATTLE_AGENT=1`. Both channels
share stdin/stdout and the engine blocks on exactly one query at a time (a hero decision, or a
unit decision of the battle that a hero move started), so one reader loop dispatches every event
(`GameAgent._handle_event` on top of `BattleAgentRunner`). Records of both kinds go to
`rl/data/game_agent_<strategy>_<battle>.jsonl` with `"kind":"strategy"|"battle"` and the
`game_end` outcome attached. Smoke (2kings, 7 days, tempo + random): 87 strategic + 76 battle
decisions, ~60 s.

## Real-battle integration (battle agent)

With `FHEROES2_BATTLE_AGENT=1` (plus the autonomous playtest mode) every AI unit activation
in a real game asks an external agent for its action. `rl/battle_agent.py` spawns the engine
and serves the channel:

```sh
rl/.venv/bin/python rl/battle_agent.py --policy mcts --sims 32 --map Arena.mp2 --days 7
# policies: random | planner (always delegate) | policy (net, needs --model) | mcts
# records every decision to rl/data/battle_agent_<policy>.jsonl
```

Engine -> agent (stdout, JSONL):
- `{"ev":"battle_start","bid":..,"seed":..,"tile":..,"wseed":..,"searchable":0|1,
  "att":{"spread":0|1,"c":color,"hid":heroId,"hero":"<hex>","stacks":[[slot,mon,count],...]},
  "def":{...}}` — right after the arena is built: everything needed to rebuild the battle in a
  headless replica. `hid`/`hero` only for a side led by a hero: the hero's save-game
  serialization (`Battle::EncodeCommander`, ~700 hex chars) — skills, artifacts, spell book,
  visited objects (morale/luck), army. `"garrison":1` on `def` when the castle's garrison
  defends; `"castle":"<hex>"` for battles on a castle/town tile (`Battle::EncodeCastle`).
  `searchable` is always 1 (kept for compatibility);
- `{"ev":"state",...,"bid":..,"searchable":..}` — a decision query; the battle-server state
  format (units, obstacles, `legal`) extended with the battle id. NOTE: it is `"ev":"state"`,
  not a separate event name;
- `{"ev":"battle_fallback","bid":..,"what":"invalid action"}` — the last reply was not a legal
  move; the built-in AI decided this one turn, the channel stays alive;
- `{"ev":"battle_end","bid":..,"result":"att|def|draw"}`; `game_end` as in the strategic layer.

Agent -> engine (stdin): `{"op":"action","act":..,"args":[..]}` (must match a legal move) or
`{"op":"planner"}` / `{"op":"skip"}` (built-in AI decides). Agent gone (EOF) => permanent
built-in fallback. `battle_action` log events carry `"src":"agent"|"planner"`.

MCTS mode: at the first decision of a `searchable` battle the runner starts a headless battle
server replica (`BattleEnv(map_name=...)` — it must load the SAME map, obstacles derive from
the tile), rebuilds the battle with `new_battle(seed, stacks, tile, world_seed, spread_*,
color_*, hero_*)`,
mirrors every real action into it and diffs the states (`turn/cur/units/obstacles`). The first
mismatch degrades the rest of the battle to policy/planner mode (`replica_synced:false` in
the records).

Every battle replicates exactly — heroes, sieges, towns and hero spells (verified: 0 desyncs
over ~58k decisions on Battlefi, Thechaos, 2kings, Arena). Hero spells are legal moves
(`act` 2, wire `[target, spell]`, Teleport `[dst, src, spell]`, no-target spells `[-1, spell]`);
after a cast the same unit gets another decision. States carry `"heroes":[{side,sp,cast}]` and, in
sieges, `"siege":{cells,towers,bridge}`. Not covered: retreat/surrender. The
reader is a byte-level line assembler — states exceed the pipe buffer, buffered readline +
select() starve.

## Running

```sh
# engine (repo root, Release build):
echo '{"op":"quit"}' | FHEROES2_BATTLE_SERVER=1 ./fheroes2

# python side:
python3 rl/selfplay.py --battles 16 --sims 32 --out rl/data
```
