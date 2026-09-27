"""GRPO (TRL) training of an LLM that plays fheroes2 on the strategic layer (see grpo_env.py).

Each prompt is a seeded game; the `num_generations` completions of a prompt play the same game
with different choices, and GRPO pushes the model towards the choices that ended better than the
group. Start from the SFT warm start (grpo_sft.py): a raw 0.5B instruct model rarely emits a tool
call. Example (Mac, LoRA, 2 threads, engines under nice):

    az/.venv/bin/python az/grpo_train.py --model az/models/grpo_sft --map 2kings.mp2 \\
        --days 7 --seeds 1-64 --num-generations 4 --max-steps 100 --out az/models/grpo_heroes

Machine load rule (AGENTS.md): do not run this while another training / generation / benchmark
run is going; the default keeps torch at 2 threads. One engine process per generation stays alive
while its episode is open (idle while the model generates).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grpo_env import KINDS, HeroesStrategyEnv, make_dataset  # noqa: E402


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, high = part.split("-", 1)
            seeds.extend(range(int(low), int(high) + 1))
        else:
            seeds.append(int(part))
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--binary", default="./fheroes2")
    parser.add_argument("--map", action="append", dest="maps", help="map file (repeatable), default 2kings.mp2")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--agent-days", type=int, default=None, help="the model decides until this day (default: all)")
    parser.add_argument("--kinds", default=",".join(KINDS), help="strategic query kinds the model answers")
    parser.add_argument("--color", action="append", dest="colors", help="color to play (repeatable); default: first player")
    parser.add_argument("--seeds", default="1-64")
    parser.add_argument("--no-advisor", action="store_true", help="do not mark the built-in AI's choice")
    parser.add_argument("--max-options", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--batch", type=int, default=4, help="per-device batch (a multiple of --num-generations)")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.0, help="KL coefficient")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-completion-length", type=int, default=5120,
                        help="token budget of the whole multi-turn episode (model text + tool results); a 7-day "
                             "2kings episode of the built-in AI is ~4.6k tokens with Qwen2.5 (an overlong episode is "
                             "finished by the built-in AI and penalized as unanswered)")
    parser.add_argument("--max-tool-calls", type=int, default=64)
    parser.add_argument("--lora-r", type=int, default=16, help="0 = full fine-tuning")
    parser.add_argument("--dtype", default="float32", choices=("float32", "bfloat16", "float16"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", default="az/models/grpo_heroes")
    parser.add_argument("--episodes-log", default=None, help="JSONL with every episode (default: <out>/episodes.jsonl)")
    parser.add_argument("--save-steps", type=int, default=50)
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    torch.set_num_threads(args.threads)
    os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
    os.makedirs(args.out, exist_ok=True)
    episodes_log = args.episodes_log or os.path.join(args.out, "episodes.jsonl")

    kinds = tuple(k for k in args.kinds.split(",") if k)
    dataset = make_dataset(parse_seeds(args.seeds), maps=tuple(args.maps or ["2kings.mp2"]), days=args.days,
                           agent_days=args.agent_days, colors=tuple(args.colors or [None]), kinds=kinds)

    def environment_factory():
        return HeroesStrategyEnv(binary=args.binary, days=args.days, kinds=kinds, max_options=args.max_options,
                                 show_advisor=not args.no_advisor, episode_log=episodes_log)

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = None
    if args.lora_r > 0:
        from peft import LoraConfig

        peft_config = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0, task_type="CAUSAL_LM",
                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    config = GRPOConfig(
        output_dir=args.out,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        beta=args.beta,
        temperature=args.temperature,
        max_completion_length=args.max_completion_length,
        max_tool_calling_iterations=args.max_tool_calls,
        # Qwen3-style templates: short answers, no thinking blocks (ignored by other templates).
        chat_template_kwargs={"enable_thinking": False},
        model_init_kwargs={"dtype": args.dtype},
        logging_steps=1,
        save_steps=args.save_steps,
        report_to="none",
        log_completions=True,
        dataloader_num_workers=0,
    )

    trainer = GRPOTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        environment_factory=environment_factory,
    )
    trainer.train()
    trainer.save_model(args.out)


if __name__ == "__main__":
    main()
