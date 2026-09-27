"""SFT warm start for the GRPO agent (grpo_env.py / grpo_train.py): teaches a small instruct model
the tool-calling format on trajectories of the built-in AI.

Why: small models (Qwen2.5-0.5B-Instruct measured) mostly answer the questions with plain text
("0", "pick 2") instead of a `choose` tool call — ~1 of 16 samples is a proper call, and an episode
needs 15+ calls in a row. GRPO alone would spend most of its budget learning the format.

Expert trajectories are exact: every query has the built-in AI's choice among its options (marked
"<- default AI"; for building it is "let the default AI decide"), and answering every query with it
replays the all-built-in game (tested invariant) — the episode reward is then exactly 0.

    # 1. trajectories -> az/data/grpo_sft_<map>.jsonl (one engine at a time)
    az/.venv/bin/python az/grpo_sft.py gen --map 2kings.mp2 --days 7 --seeds 1-200
    # 2. LoRA SFT, merged model -> az/models/grpo_sft (the --model of grpo_train.py)
    az/.venv/bin/python az/grpo_sft.py train --data az/data/grpo_sft_2kings.mp2.jsonl --out az/models/grpo_sft
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grpo_env import KINDS, HeroesStrategyEnv, default_engine_factory, make_rows  # noqa: E402

FINAL_ANSWER = "The game is over."


def tool_schemas() -> list[dict]:
    from transformers.utils import get_json_schema

    return [get_json_schema(HeroesStrategyEnv.choose)]


def expert_episode(row: dict, engine_factory=None, **env_kwargs) -> dict:
    """Plays one dataset row with the built-in AI's choices; returns an SFT example
    {"messages", "tools", "reward", ...}. The reward is 0 when the replay matched the control game."""
    engine_factory = engine_factory or default_engine_factory()
    env = HeroesStrategyEnv(engine_factory=engine_factory, **env_kwargs)

    messages = [dict(message) for message in row["prompt"]]
    # Same prompt construction as GRPOTrainer: the reset observation is appended to the last message.
    messages[-1]["content"] = messages[-1]["content"] + env.reset(**row)
    while env._pending is not None:
        option = env._pending["advisor"]
        messages.append({"role": "assistant", "content": "",
                         "tool_calls": [{"type": "function", "function": {"name": "choose", "arguments": {"option": option}}}]})
        messages.append({"role": "tool", "name": "choose", "content": env.choose(option)})
    messages.append({"role": "assistant", "content": FINAL_ANSWER})

    reward = env.get_reward()
    return {"messages": messages, "tools": json.dumps(tool_schemas()), "reward": reward, "seed": row["seed"],
            "map": row["map"], "color": env._color, "choices": env._info.get("answered", 0)}


def generate(args) -> None:
    from grpo_train import parse_seeds

    kinds = tuple(k for k in args.kinds.split(",") if k)
    out = args.out or f"az/data/grpo_sft_{args.map}.jsonl"
    engine_factory = default_engine_factory(args.binary)
    mismatched = 0
    with open(out, "w") as f:
        for seed in parse_seeds(args.seeds):
            row = make_rows([seed], maps=(args.map,), days=args.days, kinds=kinds)[0]
            example = expert_episode(row, engine_factory, max_options=args.max_options)
            mismatched += abs(example["reward"]) > 1e-9  # must stay 0: the replay IS the control game
            f.write(json.dumps(example, separators=(",", ":")) + "\n")
            print(f"seed {seed}: {example['choices']} choices, reward {example['reward']:.3f}", flush=True)
    print(f"wrote {out}; {mismatched} episode(s) diverged from the control game")


def train(args) -> None:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    torch.set_num_threads(args.threads)
    dataset = load_dataset("json", data_files=args.data, split="train").select_columns(["messages", "tools"])
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    peft_config = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0, task_type="CAUSAL_LM",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    config = SFTConfig(output_dir=args.out + "_ckpt", num_train_epochs=args.epochs, per_device_train_batch_size=args.batch,
                       gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, assistant_only_loss=True,
                       max_length=args.max_length, logging_steps=5, save_strategy="no", report_to="none",
                       model_init_kwargs={"dtype": args.dtype})
    trainer = SFTTrainer(model=args.model, args=config, train_dataset=dataset, processing_class=tokenizer,
                         peft_config=peft_config)
    trainer.train()
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(args.out)
    # Save the tokenizer with its original chat template, as the base model ships it: GRPOTrainer
    # recognizes known templates to parse tool calls (the SFT training template is not one of them).
    AutoTokenizer.from_pretrained(args.model).save_pretrained(args.out)
    print(f"merged model -> {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("gen", help="expert trajectories")
    gen.add_argument("--binary", default="./fheroes2")
    gen.add_argument("--map", default="2kings.mp2")
    gen.add_argument("--days", type=int, default=7)
    gen.add_argument("--seeds", default="1-200")
    gen.add_argument("--kinds", default=",".join(KINDS))
    gen.add_argument("--max-options", type=int, default=8)
    gen.add_argument("--out", default=None)

    tr = sub.add_parser("train", help="LoRA SFT + merge")
    tr.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    tr.add_argument("--data", required=True)
    tr.add_argument("--out", default="az/models/grpo_sft")
    tr.add_argument("--epochs", type=float, default=1.0)
    tr.add_argument("--batch", type=int, default=1)
    tr.add_argument("--grad-accum", type=int, default=8)
    tr.add_argument("--lr", type=float, default=2e-4)
    tr.add_argument("--lora-r", type=int, default=16)
    tr.add_argument("--max-length", type=int, default=6144)
    tr.add_argument("--dtype", default="float32", choices=("float32", "bfloat16", "float16"))
    tr.add_argument("--threads", type=int, default=2)

    args = parser.parse_args()
    generate(args) if args.cmd == "gen" else train(args)


if __name__ == "__main__":
    main()
