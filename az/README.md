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
{"op":"action","act":1,"args":[3,12,-1,25,4]}
{"op":"reset"}
{"op":"quit"}
```

- `new`: seed (uint32), stacks as `monsterIdx x count` CSV. The engine picks a deterministic
  land tile from the loaded map (obstacles/terrain come from the tile).
- `action`: `act` is `Battle::CommandType` (0=MOVE,1=ATTACK,8=SKIP, ...), `args` follow the
  `Battle::Command` layout (see `battle_command.h`). ATTACK: (attackerUID, defenderUID,
  cellToMoveFrom or -1, targetCell or -1, direction). Only the *current* unit may act.
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
  ATTACK targets from reachable cells or as a shooter, SKIP).
- `result`: `att`/`def`/`draw` once the battle is over; `cur` is -1 and `legal` is absent.

MCTS needs state restore; v0 uses **replay from the root** (`reset` + repeated `action`),
which is exact because the engine is deterministic. A C++ snapshot/restore (make/unmake
analogue) is planned once the loop is proven.

## Phases

1. **Loop proof (pure MCTS, no NN)**: bridge + PUCT search with a simple material-strength
   evaluation; self-play runner generates games; measure games/hour.
2. **Encoding + network**: 11x9 plane stack (sides, stacks, hp, speeds, obstacles, active unit)
   + scalar features; small ResNet; policy = move layout over cells/targets, value = outcome.
3. **AZ training loop**: batched NN inference inside self-play, Dirichlet noise, temperature
   sampling, replay buffer, iterative train/gate (gate: new net vs built-in `BattlePlanner`
   via the autonomous playtest harness).
4. **Integration**: the trained net + MCTS replaces `AI::BattlePlanner` in real games
   (on-demand battle solving); later, strategic layer value function.

## Running

```sh
# engine (repo root, Release build):
echo '{"op":"quit"}' | FHEROES2_BATTLE_SERVER=1 ./fheroes2

# python side:
python3 az/selfplay.py --battles 16 --sims 32 --out az/data
```
