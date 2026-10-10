"""From-scratch supervised fine-tuning loop (no HF Trainer).

* response-only loss (prompt and padding labels are -100);
* token-weighted gradient accumulation: every micro-batch's summed NLL is
  divided by the number of response tokens in the whole accumulation window,
  so each optimizer step minimises the mean NLL over all response tokens it
  saw, regardless of how tokens are split across micro-batches;
* fp32 master weights, fp16 autocast, dynamic loss scaling (GradScaler);
* gradient clipping, linear warmup + cosine decay, AdamW;
* periodic validation loss; ``best`` (lowest val loss) and ``final``
  checkpoints only.

Usage::

    python -m alignment.sft --train_path outputs/data/sft_train.jsonl \
        --val_path outputs/data/sft_val.jsonl --output_dir outputs/sft
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from alignment.batching import collate, pad_token_id_for, tokenize_pair
from alignment.data import read_jsonl
from alignment.logprobs import batch_logprobs
from alignment.prompting import format_prompt, prepare_tokenizer
from alignment.train_utils import (
    JsonlLogger,
    decay_parameter_groups,
    environment_info,
    save_checkpoint,
    set_seed,
    warmup_cosine,
    write_json,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--val_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--min_lr_ratio", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=100, help="optimizer steps between validations")
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_val_examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no_fp16", action="store_true", help="disable fp16 autocast")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--no_save", action="store_true", help="skip checkpoint writing (smoke tests)")
    return parser.parse_args(argv)


def load_examples(path, tokenizer, max_length, limit=None):
    rows = read_jsonl(path)[:limit]
    return [tokenize_pair(tokenizer, format_prompt(r["instruction"]), r["response"], max_length) for r in rows]


@torch.no_grad()
def evaluate(model, loader, use_fp16) -> dict:
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
            logps, counts = batch_logprobs(model, batch)
        total_nll -= logps.sum().item()
        total_tokens += counts.sum().item()
    model.train()
    loss = total_nll / max(total_tokens, 1)
    return {"val_loss": loss, "val_ppl": math.exp(min(loss, 50)), "val_tokens": total_tokens}


def main(argv=None):
    args = parse_args(argv)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    use_fp16 = device.type == "cuda" and not args.no_fp16

    tokenizer = prepare_tokenizer(AutoTokenizer.from_pretrained(args.model_name))
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32, attn_implementation="sdpa")
    model.to(device)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()

    train_examples = load_examples(args.train_path, tokenizer, args.max_length, args.max_train_examples)
    val_examples = load_examples(args.val_path, tokenizer, args.max_length, args.max_val_examples)
    pad_id = pad_token_id_for(tokenizer)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_examples,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=lambda rows: collate(rows, pad_id),
    )
    val_loader = DataLoader(val_examples, batch_size=args.eval_batch_size, collate_fn=lambda rows: collate(rows, pad_id))

    micro_per_epoch = len(train_loader)
    steps_per_epoch = math.ceil(micro_per_epoch / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, round(args.warmup_ratio * total_steps)) if args.warmup_ratio > 0 else 0

    optimizer = torch.optim.AdamW(
        decay_parameter_groups(model, args.weight_decay),
        lr=args.lr,
        betas=(0.9, args.adam_beta2),
        eps=1e-8,
        fused=device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: warmup_cosine(step, total_steps, warmup_steps, args.min_lr_ratio)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    run_config = {
        **vars(args),
        "n_train": len(train_examples),
        "n_val": len(val_examples),
        "effective_batch_size": args.per_device_batch_size * args.grad_accum,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "loss_reduction": "token-level mean over response tokens in each accumulation window",
        "precision": "fp32 weights + fp16 autocast + GradScaler" if use_fp16 else "fp32",
        "environment": environment_info(),
    }
    write_json(out_dir / "run_config.json", run_config)
    print({k: run_config[k] for k in ("n_train", "n_val", "total_steps", "warmup_steps", "effective_batch_size")})

    logger = JsonlLogger(out_dir / "metrics.jsonl")
    best_val = float("inf")
    best_step = None
    step = 0
    start = time.time()

    initial = evaluate(model, val_loader, use_fp16)
    logger.log({"type": "eval", "step": 0, **initial})
    print(f"step 0 | val_loss {initial['val_loss']:.4f}")

    for epoch in range(args.epochs):
        iterator = iter(train_loader)
        for window_start in range(0, micro_per_epoch, args.grad_accum):
            window = [next(iterator) for _ in range(min(args.grad_accum, micro_per_epoch - window_start))]
            window_tokens = sum(int(b["labels"][:, 1:].ne(-100).sum()) for b in window)
            window_nll = 0.0
            for batch in window:
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                    logps, _ = batch_logprobs(model, batch)
                loss = -logps.sum() / max(window_tokens, 1)
                scaler.scale(loss).backward()
                window_nll += loss.item()

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = use_fp16 and scaler.get_scale() < scale_before
            optimizer.zero_grad(set_to_none=True)
            lr = scheduler.get_last_lr()[0]
            scheduler.step()
            step += 1

            logger.log(
                {
                    "type": "train",
                    "step": step,
                    "epoch": epoch + (window_start + len(window)) / micro_per_epoch,
                    "loss": window_nll,
                    "lr": lr,
                    "grad_norm": float(grad_norm),
                    "tokens": window_tokens,
                    "skipped_step": bool(skipped),
                    "loss_scale": scaler.get_scale(),
                }
            )
            if step % 10 == 0:
                rate = step / (time.time() - start)
                print(f"step {step}/{total_steps} | loss {window_nll:.4f} | lr {lr:.2e} | gn {float(grad_norm):.2f} | {rate:.3f} it/s")

            if step % args.eval_every == 0 or step == total_steps:
                metrics = evaluate(model, val_loader, use_fp16)
                is_best = metrics["val_loss"] < best_val
                logger.log({"type": "eval", "step": step, **metrics, "is_best": is_best})
                print(f"step {step} | val_loss {metrics['val_loss']:.4f} | best={is_best}")
                if is_best:
                    best_val, best_step = metrics["val_loss"], step
                    if not args.no_save:
                        save_checkpoint(model, tokenizer, out_dir / "best", {"step": step, **metrics})

    if not args.no_save:
        save_checkpoint(model, tokenizer, out_dir / "final", {"step": step})
    summary = {"best_step": best_step, "best_val_loss": best_val, "final_step": step, "train_time_sec": time.time() - start}
    write_json(out_dir / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
