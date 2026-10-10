"""Task 5: TRL DPOTrainer + PEFT/LoRA persona tuning of SmolLM2-360M-Instruct.

    CUDA_VISIBLE_DEVICES=0 python scripts/train_lora_dpo.py \
        --data_path data/persona_lora/ramsay_style_preferences.json \
        --output_dir outputs/lora_dpo --configs r16_all_lr5e-5

The 220 provided pairs are split (seed 42) into 200 training and 20 held-out
pairs. Each row becomes a TRL conversational preference record::

    {"prompt":   [{"role": "user", "content": ...}],
     "chosen":   [{"role": "assistant", "content": ...}],
     "rejected": [{"role": "assistant", "content": ...}]}

so TRL applies the SmolLM2 chat template. The reference model is the frozen
base model (``ref_model=None`` with a PEFT model: TRL disables the adapter to
compute reference log-probabilities).

Several LoRA configurations (see ``CONFIGS``) can be trained in one call; each
writes ``<output_dir>/<name>/adapter`` and ``<output_dir>/<name>/metrics.jsonl``.
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
from pathlib import Path

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
ATTENTION = ["q_proj", "k_proj", "v_proj", "o_proj"]
ALL_LINEAR = ATTENTION + ["gate_proj", "up_proj", "down_proj"]

CONFIGS = {
    # rank 8 on attention only, moderate learning rate
    "r8_attn_lr5e-5": {"r": 8, "alpha": 16, "targets": ATTENTION, "lr": 5e-5},
    # main configuration: rank 16 on every linear projection
    "r16_all_lr5e-5": {"r": 16, "alpha": 32, "targets": ALL_LINEAR, "lr": 5e-5},
    # same adapter with a 4x larger learning rate
    "r16_all_lr2e-4": {"r": 16, "alpha": 32, "targets": ALL_LINEAR, "lr": 2e-4},
}


def load_split(path, n_eval, seed):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    eval_idx = set(order[:n_eval])

    def convert(row):
        return {
            "prompt": [{"role": "user", "content": row["prompt"]}],
            "chosen": [{"role": "assistant", "content": row["chosen"]}],
            "rejected": [{"role": "assistant", "content": row["rejected"]}],
        }

    train = [convert(r) for i, r in enumerate(rows) if i not in eval_idx]
    held_out = [convert(r) for i, r in enumerate(rows) if i in eval_idx]
    return train, held_out


def filter_kwargs(callable_, kwargs):
    params = inspect.signature(callable_).parameters
    accepted = {k: v for k, v in kwargs.items() if k in params}
    dropped = sorted(set(kwargs) - set(accepted))
    if dropped:
        print(f"note: {callable_.__qualname__} does not accept {dropped}; skipped")
    return accepted


def train_one(name, cfg, train_rows, eval_rows, args):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    out_dir = Path(args.output_dir) / name
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)

    peft_config = LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["alpha"],
        lora_dropout=args.lora_dropout,
        target_modules=cfg["targets"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    config_kwargs = dict(
        output_dir=str(out_dir / "trainer"),
        beta=args.beta,
        learning_rate=cfg["lr"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        bf16=False,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
        max_grad_norm=1.0,
        weight_decay=0.0,
    )
    training_args = DPOConfig(**filter_kwargs(DPOConfig, config_kwargs))
    trainer_kwargs = dict(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(eval_rows),
        peft_config=peft_config,
    )
    if "processing_class" in inspect.signature(DPOTrainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = DPOTrainer(**trainer_kwargs)
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"[{name}] trainable params {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    initial_eval = trainer.evaluate()
    trainer.train()
    final_eval = trainer.evaluate()
    trainer.model.save_pretrained(out_dir / "adapter")
    tokenizer.save_pretrained(out_dir / "adapter")

    with (out_dir / "metrics.jsonl").open("w") as handle:
        handle.write(json.dumps({"step": 0, **initial_eval}) + "\n")
        for record in trainer.state.log_history:
            handle.write(json.dumps(record) + "\n")
    summary = {
        "name": name,
        "model": args.model_name,
        "lora": {"r": cfg["r"], "alpha": cfg["alpha"], "dropout": args.lora_dropout, "target_modules": cfg["targets"]},
        "learning_rate": cfg["lr"],
        "beta": args.beta,
        "epochs": args.epochs,
        "per_device_batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "optimizer_steps": trainer.state.global_step,
        "trainable_params": trainable,
        "total_params": total,
        "n_train": len(train_rows),
        "n_eval": len(eval_rows),
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "trainer_args": {k: v for k, v in config_kwargs.items() if k != "output_dir"},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    del trainer, model
    torch.cuda.empty_cache()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_path", default="data/persona_lora/ramsay_style_preferences.json")
    parser.add_argument("--output_dir", default="outputs/lora_dpo")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--n_eval", type=int, default=20)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_rows, eval_rows = load_split(args.data_path, args.n_eval, args.seed)
    print(f"{len(train_rows)} train / {len(eval_rows)} held-out pairs")
    summaries = [train_one(name, CONFIGS[name], train_rows, eval_rows, args) for name in args.configs]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "sweep_summary.json").write_text(json.dumps(summaries, indent=2, default=str))
    for s in summaries:
        fe = s["final_eval"]
        print(
            f"{s['name']}: eval_loss {fe.get('eval_loss'):.4f} | "
            f"acc {fe.get('eval_rewards/accuracies')} | margin {fe.get('eval_rewards/margins')}"
        )


if __name__ == "__main__":
    main()
