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
| az/ Python prototype                                 |
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
  land tile from the loaded map (obstacles/terrain come from the tile).
- `action`: `act` is `Battle::CommandType` (0=MOVE,1=ATTACK,8=SKIP, ...), `args` are the
  `Battle::Command` values as stored — in REVERSE constructor order (see `battle_command.h`):
  MOVE `[dst, uid]`, ATTACK `[direction, targetCell|-1, cellToMoveFrom|-1, defenderUID,
  attackerUID]`, SKIP `[uid]`. Copy moves from `legal` rather than building them by hand.
  Only the *current* unit may act; a command the engine would reject (wrong unit, not a valid
  MOVE/ATTACK/SKIP at this point) is answered with `{"ev":"error","what":"illegal action"}`
  and not applied (same for `replay`/`restore` paths).
- `reset`: rebuild the battle from the last `new` setup (replay-based MCTS).

Reply line (engine -> Python):

```json
{"ev":"state","turn":3,"cur":7,"units":[...],"obstacles":[...],"legal":[...],"result":null}
{"ev":"result","winner":"att"}
```

- `cur`: UID of the unit expected to act, or -1.
- `units`: `{"u":uid,"side":"att"|"def","mon":id,"q":count,"hpl":hpOfTopMonster,"i":headCell,
  "ti":tailCell|-1,"sp":speed,"shots":n,"moved":0|1}`.
- `legal`: list of `{"act":..,"args":[..]}` for the current unit (MOVE to each reachable cell,
  ATTACK targets from reachable cells or as a shooter, SKIP) — exactly the commands the engine
  accepts (filtered through the engine's own validation).
- `result`: `att`/`def`/`draw` once the battle is over; `cur` is -1 and `legal` is absent.

MCTS needs state restore; v0 uses **replay from the root** (`reset` + repeated `action`),
which is exact because the engine is deterministic. A C++ snapshot/restore (make/unmake
analogue) is planned once the loop is proven.

## Phases

1. **Loop proof (pure MCTS, no NN)** — DONE: bridge + PUCT search with a material-strength
   evaluation; ~5 s per battle (sims=16) after the batched replay op.
2. **Encoding + network** — DONE (v0): 11-channel plane stack + 3 scalars; fixed 793-slot
   action space (99 MOVE / 693 ATTACK / 1 SKIP); 4-block ResNet (~250k params);
   `az/train.py` trains on self-play records and saves `az/models/az_battle_v1.pt`.
   The protocol is strictly stateless now: every op (new/action/replay/reset) answers
   immediately, the engine replays the main line from the root.
3. **AZ training loop** — DONE (v0):
   - Expert warm-start: the dataset is generated from the ready-made algorithms — the battle
     server's "auto" op plays battles with the built-in BattlePlanner and streams (state,
     expert action) pairs (`az/gen_expert.py`, 400 battles -> 7.5k records in ~2 s; policy CE
     0.77 -> 0.027, i.e. the net imitates the built-in AI per move with ~97% accuracy).
   - The trained net guides the search: policy priors + value head inside MCTS, Dirichlet
     noise at the root (`az/selfplay.py --model az/models/az_battle_expert_v1.pt --device mps`).
   - Transformer variant (stage 3.5): `az/transformer_model.py` — a policy/value transformer
     on the ready-made HuggingFace Qwen3 body (~1.0M params, cell+direction action decoding
     with a KV cache), trained by `az/train.py --arch transformer`; see AGENTS.md
     "Transformer architecture (stage 3.5)".
   - Search infrastructure: MCTS node states materialize via battle-server snapshot/restore
     (C++ `ArenaSnapshot`, ops `snap`/`restore`; ~3.3x faster than replay-from-root at depth
     30, visit counts verified identical), and `az/gate.py` measures the win rate of our
     engine vs the built-in BattlePlanner (via the `suggest` op).
   - Known gaps: ~half of the expert records are skipped because the v0 legal-move
     enumeration is narrower than the planner's real options (spells, catapult, some attack
     cells); per-leaf inference is not batched.
4. **Integration**: the trained net + MCTS replaces `AI::BattlePlanner` in real games
   (on-demand battle solving); later, strategic layer value function. First step done:
   the battle-agent channel (see "Real-battle integration" below).

## Strategic layer (phase 4 preview)

The strategic decision protocol delegates hero target choices to an external agent while the
engine keeps all mechanics (pathfinding, movement, battles, economy). Enabled together with the
autonomous playtest mode:

```sh
python3 az/strategy_run.py --policy greedy --playthroughs 1 --days 10 --map 2kings.mp2
```

Engine -> agent (stdout, JSONL):
- `{"ev":"turn_context","t":..,"diff":..,"res":[wood,mercury,ore,sulfur,crystal,gems,gold],
  "castles":[{"n":..,"i":..}],"heroes":[{"id","i","mp","mmp","str"}]}` — at the start of each AI turn;
- `{"ev":"decision","t":..,"h":heroId,"from":tile,"cands":[{"i":tile,"obj":type,"v":value,"d":dist}, ...]}`
  — one per hero activation; candidates are all positive-value targets as evaluated by the
  built-in strategic AI (already sorted by value).

Agent -> engine (stdin):
- `{"op":"pick","h":heroId,"i":tile}` — must be one of the candidates, otherwise ignored
  (the built-in choice is used);
- `{"op":"skip"}` — keep the built-in choice;
- `{"op":"quit"}`.

After each playthrough the engine reports `{"ev":"game_end","playthrough":..,"day":..,
"results":[{"c":color,"s":state,"d":day}]}`. The channel is self-healing: if the agent dies or
sends garbage, the engine falls back to the built-in AI permanently.

`az/strategy_run.py` records every decision with the final game outcome attached
(`az/data/strategy_<policy>.jsonl`) — the training data format for the strategic value network.
Policies (`az/strategy_policies.py`): `greedy`, `random`, `builtin` (always skip) and `tempo` —
value/distance-aware: discounts a candidate by `gamma` per extra turn of travel (from the hero's
move points in `turn_context`) and penalizes tiles another hero already claimed this kingdom
turn. The built-in value already folds in the distance, and early-game candidates are usually
all within one turn (2kings, week 1: `tempo` == `greedy`), so the difference shows only with
long-range candidates and several heroes.

### One agent for both channels (`az/game_agent.py`)

```sh
az/.venv/bin/python az/game_agent.py --strategy tempo --battle mcts --sims 16 --map 2kings.mp2 --days 7
```

Spawns the engine with `FHEROES2_STRATEGY_SERVER=1` and `FHEROES2_BATTLE_AGENT=1`. Both channels
share stdin/stdout and the engine blocks on exactly one query at a time (a hero decision, or a
unit decision of the battle that a hero move started), so one reader loop dispatches every event
(`GameAgent._handle_event` on top of `BattleAgentRunner`). Records of both kinds go to
`az/data/game_agent_<strategy>_<battle>.jsonl` with `"kind":"strategy"|"battle"` and the
`game_end` outcome attached. Smoke (2kings, 7 days, tempo + random): 87 strategic + 76 battle
decisions, ~60 s.

## Real-battle integration (battle agent)

With `FHEROES2_BATTLE_AGENT=1` (plus the autonomous playtest mode) every AI unit activation
in a real game asks an external agent for its action. `az/battle_agent.py` spawns the engine
and serves the channel:

```sh
az/.venv/bin/python az/battle_agent.py --policy mcts --sims 32 --map Arena.mp2 --days 7
# policies: random | planner (always delegate) | policy (net, needs --model) | mcts
# records every decision to az/data/battle_agent_<policy>.jsonl
```

Engine -> agent (stdout, JSONL):
- `{"ev":"battle_start","bid":..,"seed":..,"tile":..,"wseed":..,"searchable":0|1,
  "att":{"spread":0|1,"stacks":[[slot,mon,count],...]},"def":{...}}` — right after the arena
  is built: everything needed to rebuild the battle in a headless replica;
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
the tile), rebuilds the battle with `new_battle(seed, stacks, tile, world_seed, spread_*)`,
mirrors every real action into it and diffs the states (`turn/cur/units/obstacles`). The first
mismatch degrades the rest of the battle to policy/planner mode (`replica_synced:false` in
the records).

Known limitations: hero battles desync after the hero acts (the replica has no commander
stats), so MCTS only covers monster-only battles fully; sieges are never searchable. The
reader is a byte-level line assembler — states exceed the pipe buffer, buffered readline +
select() starve.

## Running

```sh
# engine (repo root, Release build):
echo '{"op":"quit"}' | FHEROES2_BATTLE_SERVER=1 ./fheroes2

# python side:
python3 az/selfplay.py --battles 16 --sims 32 --out az/data
```
