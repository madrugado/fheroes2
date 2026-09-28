"""Unit tests for rl/mcts.py with a scripted fake environment (no engine required)."""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from mcts import Mcts, evaluate_state  # noqa: E402


class FakeEnv:
    """Deterministic two-step 'battle': both sides have one unit, one move decides everything.

    Legal moves at every decision point: MOVE to cell 1, ATTACK (dir flag 4), SKIP.
    After any ATTACK the battle ends with the mover's side winning; MOVE/SKIP lead to a
    terminal loss for the mover (simulated by the fake engine's script).
    """

    def __init__(self):
        self.script = []

    def replay(self, path):
        self.script.append(list(path))
        attacked = any(act == 1 for act, _ in path)
        return self.state(attacked)

    @staticmethod
    def state(attacked):
        units = [
            {"u": 1, "side": "att", "mon": 1, "q": 10, "hpl": 5, "i": 0, "ti": -1, "sp": 3, "shots": 0, "moved": 0},
            {"u": 2, "side": "def", "mon": 2, "q": 10, "hpl": 5, "i": 6, "ti": -1, "sp": 3, "shots": 0, "moved": 0},
        ]
        legal = [
            {"act": 0, "args": [1, 1]},
            {"act": 1, "args": [1, 2, -1, 6, 4]},
            {"act": 8, "args": [1]},
        ]
        state = {"turn": 1, "units": units, "obstacles": [], "cur": 1, "legal": legal}
        if attacked:
            state = dict(state)
            state["legal"] = []
            state["result"] = "att"
        return state


def test_heuristic_value_symmetry():
    state = FakeEnv.state(attacked=False)
    assert evaluate_state(state, "att") == -evaluate_state(state, "def")


def test_mcts_prefers_winning_attack():
    env = FakeEnv()
    rng = random.Random(1)
    mcts = Mcts(env, rng=rng, root_noise=0.0)

    root_state = env.replay(())
    legal, counts = mcts.run(root_state, num_simulations=12)

    assert len(legal) == 3
    # The first simulation expands the root without choosing a move, so the children share
    # num_simulations - 1 visits.
    assert sum(counts) == 12 - 1

    attack_index = next(i for i, (act, _) in enumerate(legal) if act == 1)
    assert counts[attack_index] == max(counts), "the winning attack must collect the most visits"


def test_mcts_visit_counts_are_conserved():
    env = FakeEnv()
    mcts = Mcts(env, rng=random.Random(7), root_noise=0.0)
    root_state = env.replay(())

    _, counts = mcts.run(root_state, num_simulations=9)
    assert sum(counts) == 9 - 1


def test_mcts_with_root_noise_still_terminates():
    env = FakeEnv()
    mcts = Mcts(env, rng=random.Random(3), root_noise=0.5, dirichlet_alpha=1.0)
    root_state = env.replay(())

    legal, counts = mcts.run(root_state, num_simulations=8)
    assert sum(counts) == 8 - 1
    assert all(c >= 0 for c in counts)


def test_mcts_terminal_root_returns_empty():
    env = FakeEnv()
    mcts = Mcts(env, rng=random.Random(1))

    terminal = env.replay(())
    terminal = dict(terminal)
    terminal["legal"] = []
    terminal["result"] = "att"

    legal, counts = mcts.run(terminal, num_simulations=4)
    assert legal == [] and counts == []


class SnapshotFakeEnv(FakeEnv):
    """FakeEnv with battle-server-style snapshot support, exercising the snapshot code path
    of MCTS without a real engine. A snapshot remembers whether the battle was already won;
    restore + path suffix applies the same transition rule as replay()."""

    def __init__(self):
        super().__init__()
        self.snapshots: dict[int, bool] = {}  # id -> 'attacked' flag of the stored state
        self.restore_count = 0
        self.free_count = 0
        self.replay_calls = 0

    def snapshot_save(self, snap_id):
        # run() snapshots the battle root, which in FakeEnv terms is the un-attacked state.
        self.snapshots[snap_id] = False
        return {"ev": "state"}

    def snapshot_restore(self, snap_id, path=(), save_as: int = 0):
        if snap_id not in self.snapshots:
            return {"ev": "error", "what": "unknown snapshot id"}
        attacked = self.snapshots[snap_id] or any(act == 1 for act, _ in path)
        if save_as:
            self.snapshots[save_as] = attacked
        self.restore_count += 1
        return self.state(attacked)

    def snapshots_free(self):
        self.snapshots.clear()
        self.free_count += 1
        return {"ev": "state"}

    def replay(self, path):
        self.replay_calls += 1
        return super().replay(path)


def test_mcts_snapshot_path_matches_replay_path():
    """With a snapshot-capable engine the search must produce the same visit counts as the
    replay-based path (same rng seed, no root noise). replay() may only be used for the
    run() position canonicalization, never for node materialization."""
    replay_env = FakeEnv()
    snap_env = SnapshotFakeEnv()

    root_state = snap_env.replay(())
    snap_env.replay_calls = 0

    legal_r, counts_r = Mcts(replay_env, rng=random.Random(11), root_noise=0.0).run(
        replay_env.replay(()), num_simulations=15)
    legal_s, counts_s = Mcts(snap_env, rng=random.Random(11), root_noise=0.0).run(
        root_state, num_simulations=15)

    assert legal_r == legal_s
    assert counts_r == counts_s
    assert snap_env.restore_count > 0, "nodes must be materialized via snapshot restore"
    assert snap_env.replay_calls == 1, "replay() is only allowed for the run() canonicalization"
    assert snap_env.free_count >= 1, "snapshots must be freed at the end of run()"


def test_mcts_uses_snapshots_only_when_supported():
    """An engine without snapshot methods must keep working through replay()."""
    env = FakeEnv()
    mcts = Mcts(env, rng=random.Random(5), root_noise=0.0)

    assert not mcts._use_snapshots

    legal, counts = mcts.run(env.replay(()), num_simulations=8)
    assert len(legal) == 3
    assert sum(counts) == 8 - 1
