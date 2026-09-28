"""Unit tests for rl/grpo_env.py (the TRL GRPO environment) against a scripted fake engine, plus a
one-step GRPOTrainer smoke test with a tiny random model (skipped without trl or a cached Qwen2.5
tokenizer)."""

import io
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from grpo_env import ControlCache, HeroesStrategyEnv, make_rows, player_score  # noqa: E402
from line_reader import LineReader  # noqa: E402


class FakeEngine:
    """Replays a fixed event script (independent of the replies), records the replies."""

    def __init__(self, script):
        read_fd, write_fd = os.pipe()
        os.write(write_fd, "".join(json.dumps(ev) + "\n" for ev in script).encode())
        os.close(write_fd)
        self._stdout = os.fdopen(read_fd, "rb", buffering=0)
        self._reader = LineReader(self._stdout.fileno())
        self.read_timeout = 5.0
        self.replies = []
        self.closed = False

    def _send(self, obj):
        self.replies.append(obj)

    def run(self, policy):
        summaries = []
        while (line := self._reader.read_line(self.read_timeout)) is not None:
            ev = json.loads(line)
            if ev["ev"] in ("decision", "build", "hire", "army"):
                self._send({"op": "skip"})
            elif ev["ev"] == "game_end":
                summaries.append(ev)
        return summaries

    def close(self):
        self.closed = True


def game_end(blue_str, blue_castles=1, blue_state="2"):
    return {"ev": "game_end", "day": 3, "results": [
        {"c": "Blue", "s": blue_state, "k": blue_castles, "h": 1, "str": blue_str, "g": 0},
        {"c": "Green", "s": "2", "k": 1, "h": 1, "str": 100, "g": 0}]}


RES = [10, 0, 10, 0, 0, 0, 5000]

SCRIPT = [
    {"ev": "turn_context", "t": 1, "p": "Blue", "res": RES, "castles": [{"i": 7}],
     "heroes": [{"id": 3, "i": 7, "mp": 1000, "mmp": 1500, "str": 120.0}]},
    {"ev": "decision", "t": 1, "p": "Blue", "h": 3, "from": 7, "cands": []},  # no choice -> skipped
    {"ev": "decision", "t": 1, "p": "Blue", "h": 3, "from": 7, "cands": [{"i": 50, "obj": 0, "v": 9.0, "d": 1}]},  # one
    {"ev": "decision", "t": 1, "p": "Blue", "h": 3, "from": 7,
     "cands": [{"i": 40, "obj": 0, "v": 900.0, "d": 300}, {"i": 41, "obj": 0, "v": 500.0, "d": 100}]},
    {"ev": "hire", "t": 1, "p": "Blue", "heroes": 1, "res": RES, "bi": 1,
     "cands": [{"castle": 7, "slot": 1, "race": 1, "lvl": 1, "val": 800}, {"castle": 7, "slot": 2, "race": 2, "lvl": 1, "val": 900}]},
    {"ev": "decision", "t": 1, "p": "Green", "h": 9, "from": 8,
     "cands": [{"i": 60, "obj": 0, "v": 1.0, "d": 1}, {"i": 61, "obj": 0, "v": 0.5, "d": 1}]},  # other color
    {"ev": "build", "t": 1, "p": "Blue", "castle": 7, "race": 1, "defensive": 0, "res": RES,
     "cands": [{"b": 16, "name": "Statue", "trade": 0, "cost": [0, 0, 5, 0, 0, 0, 1250]}]},
    {"ev": "build_result", "t": 1, "p": "Blue", "castle": 7, "b": 16, "src": "agent"},
    {"ev": "army", "t": 2, "p": "Blue", "castle": 7, "reason": "visit", "guest": 3, "garrison": 0, "hero": 120.0,
     "res": RES, "offer": [{"mon": 1, "avail": 10, "n": 10, "str": 20.0}]},
    game_end(300),
]

CONTROL = [game_end(200)]


def make_env(script=SCRIPT, control=CONTROL, **kwargs):
    engines = []

    def factory(map_name, days, seed):
        # The first engine of a (map, days, seed) is the episode, later ones are control games.
        engine = FakeEngine(script if not engines else control)
        engines.append(engine)
        return engine

    env = HeroesStrategyEnv(engine_factory=factory, control_cache=ControlCache(), **kwargs)
    return env, engines


ROW = make_rows([5], days=3)[0]


def test_only_tool_is_choose():
    import inspect

    env, _ = make_env()
    tools = [name for name, _ in inspect.getmembers(env, predicate=inspect.ismethod)
             if name not in ("reset", "get_reward") and not name.startswith("_")]
    assert tools == ["choose"]


def test_reset_skips_queries_without_a_choice_and_shows_the_first_real_one():
    env, engines = make_env()
    obs = env.reset(**ROW)

    assert "You play Blue" in obs and "where should the hero go?" in obs
    assert "0: " in obs and "1: " in obs and "<- default AI" in obs.splitlines()[-2]
    assert engines[0].replies == [{"op": "skip"}, {"op": "skip"}]  # empty + single candidate


def test_choices_map_to_engine_replies_and_other_colors_are_builtin():
    env, engines = make_env()
    env.reset(**ROW)

    obs = env.choose(1)  # second target
    assert "Hire a new hero" in obs
    assert "1: Knight hero" in obs and "2: Barbarian hero level 1, value 900, in castle at tile 7  <- default AI" in obs
    obs = env.choose(0)  # hire nobody
    assert "What to build?" in obs and "0: let the default AI decide  <- default AI" in obs and "1: build nothing" in obs
    obs = env.choose(2)  # Statue
    assert "Troop budget?" in obs and "10 Peasant (of 10)" in obs and "100% of the treasury  <- default AI" in obs
    obs = env.choose(1)  # 50%
    assert "The game is over" in obs and "army strength 300" in obs

    assert engines[0].replies[2:] == [
        {"op": "pick", "h": 3, "i": 41},
        {"op": "hire", "castle": -1},
        {"op": "skip"},  # Green's decision
        {"op": "build", "castle": 7, "b": 16},
        {"op": "army", "castle": 7, "pct": 50},
    ]
    assert engines[0].closed
    assert env.get_reward() == pytest.approx((300 - 200) / 1000)
    assert env._info["answered"] == 4 and env._info["unanswered"] == 0


def test_invalid_options_keep_the_question_and_are_penalized():
    env, _ = make_env(invalid_penalty=0.1)
    env.reset(**ROW)

    obs = env.choose(7)
    assert obs.startswith("Invalid option 7") and "where should the hero go?" in obs
    obs = env.choose("x")
    assert obs.startswith("Invalid option 'x'")
    assert "Hire a new hero" in env.choose(0)

    for _ in range(3):
        env.choose(0)
    assert env.choose(0) == "The game is over; there is nothing to answer."
    assert env.get_reward() == pytest.approx(0.1 - 3 * 0.1)


def test_early_stop_finishes_with_builtin_answers_and_penalizes_the_rest():
    env, engines = make_env(unanswered_penalty=1.0)
    env.reset(**ROW)
    env.choose(0)

    reward = env.get_reward()  # the model stopped after one answer: 3 questions left

    assert engines[0].replies[-1] == {"op": "skip"} and engines[0].closed
    assert env._info["answered"] == 1 and env._info["unanswered"] == 3
    assert reward == pytest.approx((300 - 200) / 1000 - 3 / 4)
    assert env.get_reward() == reward  # cached


def test_agent_days_and_kinds_limit_the_questions():
    env, _ = make_env()
    obs = env.reset(**dict(ROW, agent_days=1, kinds="hire,army"))
    assert "Hire a new hero" in obs
    obs = env.choose(2)
    assert "The game is over" in obs  # build is not a controlled kind, the army query is on day 2


def test_score_and_control_cache():
    assert player_score({"str": 100, "k": 2, "s": "0"}) == 100 + 4000 + 10000
    assert player_score({"str": 100, "k": 0, "s": "1"}) == 100 - 10000

    cache = ControlCache()
    calls = []

    def factory(map_name, days, seed):
        calls.append(seed)
        return FakeEngine(CONTROL)

    assert cache.get(factory, "m", 3, 1)["Blue"]["str"] == 200
    assert cache.get(factory, "m", 3, 1)["Blue"]["str"] == 200
    assert calls == [1]


def test_episode_log(tmp_path):
    log = tmp_path / "episodes.jsonl"
    env, _ = make_env(episode_log=str(log))
    env.reset(**ROW)
    env.choose(0)
    env.get_reward()

    record = json.loads(log.read_text())
    assert record["seed"] == 5 and record["color"] == "Blue" and record["unanswered"] == 3
    assert record["choices"] == [{"kind": "target", "t": 1, "option": 0, "advisor": 0}]


def _cached_tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", padding_side="left", local_files_only=True)
    except Exception:
        return None


def test_grpo_trainer_runs_one_step_with_the_environment(tmp_path, monkeypatch):
    pytest.importorskip("trl")
    tokenizer = _cached_tokenizer()
    if tokenizer is None:
        pytest.skip("Qwen2.5 tokenizer not cached")

    import torch
    from datasets import Dataset
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from trl import GRPOConfig, GRPOTrainer

    monkeypatch.setenv("TRL_EXPERIMENTAL_SILENCE", "1")
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=4096))
    envs = []

    def environment_factory():
        env, _ = make_env()
        envs.append(env)
        return env

    config = GRPOConfig(output_dir=str(tmp_path), num_generations=2, per_device_train_batch_size=2, max_steps=1,
                        max_completion_length=8, max_tool_calling_iterations=2, report_to="none", use_cpu=True,
                        logging_steps=1, save_strategy="no")
    trainer = GRPOTrainer(model=model, args=config, train_dataset=Dataset.from_list(make_rows([1, 2], days=3)),
                          processing_class=tokenizer, environment_factory=environment_factory)
    assert [tool.__name__ for tool in trainer.tools] == ["choose"]
    trainer.train()

    # Random tokens make no tool calls: every episode is finished with built-in answers and scored.
    scored = [env for env in envs if env._reward is not None]
    assert scored and all(env._info["unanswered"] == 4 for env in scored)


def test_expert_episode_answers_with_the_advisor_and_builds_an_sft_example():
    from grpo_sft import FINAL_ANSWER, expert_episode

    engines = []

    def factory(map_name, days, seed):
        engine = FakeEngine(SCRIPT if not engines else CONTROL)
        engines.append(engine)
        return engine

    import grpo_env

    saved = grpo_env.CONTROL_CACHE
    grpo_env.CONTROL_CACHE = ControlCache()  # expert_episode builds the env with the default cache
    try:
        example = expert_episode(ROW, engine_factory=factory)
    finally:
        grpo_env.CONTROL_CACHE = saved

    calls = [m["tool_calls"][0]["function"]["arguments"]["option"] for m in example["messages"] if m.get("tool_calls")]
    assert calls == [0, 2, 0, 2]  # top target, the built-in hire, delegate the build, 100% budget
    assert engines[0].replies[2:] == [{"op": "pick", "h": 3, "i": 40}, {"op": "hire", "castle": 7, "slot": 2},
                                      {"op": "skip"}, {"op": "skip"}, {"op": "army", "castle": 7, "pct": 100}]
    roles = [m["role"] for m in example["messages"]]
    assert roles[:2] == ["system", "user"] and roles[2:-1] == ["assistant", "tool"] * 4 and roles[-1] == "assistant"
    assert example["messages"][-1]["content"] == FINAL_ANSWER
    assert "where should the hero go?" in example["messages"][1]["content"]
    assert json.loads(example["tools"])[0]["function"]["name"] == "choose"


def test_sft_warm_start_runs_one_step(tmp_path):
    pytest.importorskip("trl")
    tokenizer = _cached_tokenizer()
    if tokenizer is None:
        pytest.skip("Qwen2.5 tokenizer not cached")

    import torch
    from datasets import Dataset
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from trl import SFTConfig, SFTTrainer

    from grpo_sft import tool_schemas

    torch.manual_seed(0)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=4096))
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "Day 1. 0: a  1: b"},
                {"role": "assistant", "content": "",
                 "tool_calls": [{"type": "function", "function": {"name": "choose", "arguments": {"option": 1}}}]},
                {"role": "tool", "name": "choose", "content": "The game is over."},
                {"role": "assistant", "content": "The game is over."}]
    dataset = Dataset.from_list([{"messages": messages, "tools": json.dumps(tool_schemas())}] * 2)
    config = SFTConfig(output_dir=str(tmp_path), max_steps=1, per_device_train_batch_size=2, assistant_only_loss=True,
                       report_to="none", use_cpu=True, save_strategy="no")
    trainer = SFTTrainer(model=model, args=config, train_dataset=dataset, processing_class=tokenizer)
    trainer.train()

    sample = trainer.train_dataset[0]
    trained = [token for token, label in zip(sample["input_ids"], sample["labels"]) if label != -100]
    assert 0 < len(trained) < len(sample["input_ids"])  # loss on the assistant turns only
    text = tokenizer.decode(trained)
    assert '"option": 1' in text and "Day 1" not in text


GAME_BINARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "fheroes2")
MAP_2KINGS = os.path.join(os.path.dirname(GAME_BINARY), "maps", "2kings.mp2")


@pytest.mark.skipif(not (os.path.exists(GAME_BINARY) and os.path.exists(MAP_2KINGS)), reason="needs ./fheroes2 and maps/2kings.mp2")
def test_real_engine_expert_replays_the_control_game_and_other_choices_count():
    from grpo_sft import expert_episode

    os.chdir(os.path.dirname(GAME_BINARY))
    row = make_rows([11], days=5)[0]
    example = expert_episode(row)
    assert example["choices"] > 5
    assert example["reward"] == 0.0  # the built-in choices replay the control game exactly

    env = HeroesStrategyEnv()
    env.reset(**row)
    while env._pending is not None:
        options = env._pending["options"]
        env.choose(len(options) - 1)  # always the last option: a different game
    env.get_reward()
    assert env._info["agent"] != env._info["control"]
