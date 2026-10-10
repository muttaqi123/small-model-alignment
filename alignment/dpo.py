"""From-scratch Direct Preference Optimization loop (no TRL / HF Trainer).

* trainable policy initialised from the SFT checkpoint (fp32 weights, fp16
  autocast, GradScaler) on ``--policy_device``;
* frozen reference = a second copy of the same SFT checkpoint on
  ``--reference_device``, ``requires_grad=False``, always evaluated under
  ``torch.no_grad()``;
* response-only summed log-probabilities for chosen and rejected responses
  (identical prompt formatting on both sides);
* loss ``-log sigmoid(beta * ((pi_w - pi_l) - (ref_w - ref_l)))``;
* validation DPO loss, reward accuracy (strict ``r_w > r_l``) and reward
  margin; ``best`` checkpoint by validation reward accuracy (ties broken by
  margin) plus ``final``.

By default the reference is loaded in the same precision and run under the
same autocast as the policy, so at step 0 the two models produce identical
scores and every implicit reward is exactly zero.

Usage::

    python -m alignment.dpo --sft_path outputs/sft/best \
        --train_path outputs/data/dpo_hh_train.jsonl \
        --val_path outputs/data/dpo_hh_val.jsonl --output_dir outputs/dpo_hh
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
from alignment.losses import dpo_loss_from_logps, dpo_metrics, policy_and_reference_logps
from alignment.prompting import format_prompt
from alignment.train_utils import (
    JsonlLogger,
    environment_info,
    save_checkpoint,
    set_seed,
    warmup_constant,
    warmup_cosine,
    write_json,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sft_path", required=True, help="SFT checkpoint used for both policy and reference")
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--val_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--schedule", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--optimizer", choices=["adamw", "rmsprop"], default="adamw")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=50, help="optimizer steps between validations")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="pairs per validation forward pass")
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_val_examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy_device", default="cuda:0")
    parser.add_argument("--reference_device", default="cuda:1")
    parser.add_argument("--reference_dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--no_save", action="store_true")
    return parser.parse_args(argv)


def load_pairs(path, tokenizer, max_length, limit=None):
    rows = read_jsonl(path)[:limit]
    pairs = []
    for row in rows:
        prompt = format_prompt(row["instruction"])
        pairs.append(
            (
                tokenize_pair(tokenizer, prompt, row["chosen"], max_length),
                tokenize_pair(tokenizer, prompt, row["rejected"], max_length),
            )
        )
    return pairs


def make_collate(pad_id):
    def collate_pairs(rows):
        chosen = [c for c, _ in rows]
        rejected = [r for _, r in rows]
        width = max(len(ids) for ids, _ in chosen + rejected)
        return {"chosen": collate(chosen, pad_id, width), "rejected": collate(rejected, pad_id, width)}

    return collate_pairs


def autocast(device, enabled):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)


@torch.no_grad()
def evaluate(policy, reference, loader, beta, policy_device, use_fp16) -> dict:
    policy.eval()
    losses, correct, margins, chosen_r, rejected_r, pol_c, pol_r = [], [], [], [], [], [], []
    for batch in loader:
        with autocast(policy_device, use_fp16):
            pc, pr, rc, rr = policy_and_reference_logps(policy, reference, batch)
        loss = dpo_loss_from_logps(pc, pr, rc, rr, beta)
        stats = dpo_metrics(pc, pr, rc, rr, beta)
        losses.append(loss.item() * pc.shape[0])
        correct.append(stats["correct"].cpu())
        margins.append(stats["margins"].cpu())
        chosen_r.append(stats["chosen_rewards"].cpu())
        rejected_r.append(stats["rejected_rewards"].cpu())
        pol_c.append(pc.float().cpu())
        pol_r.append(pr.float().cpu())
    policy.train()
    correct = torch.cat(correct)
    n = correct.numel()
    return {
        "val_loss": sum(losses) / n,
        "val_reward_accuracy": correct.mean().item(),
        "val_reward_margin": torch.cat(margins).mean().item(),
        "val_chosen_reward": torch.cat(chosen_r).mean().item(),
        "val_rejected_reward": torch.cat(rejected_r).mean().item(),
        "val_policy_chosen_logp": torch.cat(pol_c).mean().item(),
        "val_policy_rejected_logp": torch.cat(pol_r).mean().item(),
        "val_pairs": n,
    }


def main(argv=None):
    args = parse_args(argv)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    policy_device = torch.device(args.policy_device)
    reference_device = torch.device(args.reference_device)
    use_fp16 = policy_device.type == "cuda" and not args.no_fp16

    tokenizer = AutoTokenizer.from_pretrained(args.sft_path)
    policy = AutoModelForCausalLM.from_pretrained(args.sft_path, torch_dtype=torch.float32, attn_implementation="sdpa")
    policy.to(policy_device)
    policy.config.use_cache = False
    if args.gradient_checkpointing:
        policy.gradient_checkpointing_enable()
    policy.train()

    ref_dtype = torch.float16 if args.reference_dtype == "fp16" else torch.float32
    reference = AutoModelForCausalLM.from_pretrained(args.sft_path, torch_dtype=ref_dtype, attn_implementation="sdpa")
    reference.to(reference_device)
    reference.config.use_cache = False
    reference.eval()
    reference.requires_grad_(False)

    pad_id = pad_token_id_for(tokenizer)
    train_pairs = load_pairs(args.train_path, tokenizer, args.max_length, args.max_train_examples)
    val_pairs = load_pairs(args.val_path, tokenizer, args.max_length, args.max_val_examples)
    collate_pairs = make_collate(pad_id)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_pairs, batch_size=args.per_device_batch_size, shuffle=True, generator=generator, collate_fn=collate_pairs
    )
    val_loader = DataLoader(val_pairs, batch_size=args.eval_batch_size, collate_fn=collate_pairs)

    micro_per_epoch = len(train_loader)
    steps_per_epoch = math.ceil(micro_per_epoch / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs

    params = [p for p in policy.parameters() if p.requires_grad]
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, fused=policy_device.type == "cuda")
    else:
        optimizer = torch.optim.RMSprop(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.schedule == "constant":
        lr_lambda = lambda step: warmup_constant(step, args.warmup_steps)  # noqa: E731
    else:
        lr_lambda = lambda step: warmup_cosine(step, total_steps, args.warmup_steps)  # noqa: E731
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    run_config = {
        **vars(args),
        "n_train_pairs": len(train_pairs),
        "n_val_pairs": len(val_pairs),
        "effective_batch_size": args.per_device_batch_size * args.grad_accum,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "reference_checkpoint": str(Path(args.sft_path).resolve()),
        "precision": "policy fp32 weights + fp16 autocast + GradScaler" if use_fp16 else "fp32",
        "environment": environment_info(),
    }
    write_json(out_dir / "run_config.json", run_config)
    print({k: run_config[k] for k in ("n_train_pairs", "n_val_pairs", "total_steps", "effective_batch_size")})

    logger = JsonlLogger(out_dir / "metrics.jsonl")
    best_key = (-1.0, -float("inf"))
    best_step = None
    step = 0
    start = time.time()

    initial = evaluate(policy, reference, val_loader, args.beta, policy_device, use_fp16)
    logger.log({"type": "eval", "step": 0, **initial})
    print(f"step 0 | {initial}")

    for epoch in range(args.epochs):
        iterator = iter(train_loader)
        for window_start in range(0, micro_per_epoch, args.grad_accum):
            n_micro = min(args.grad_accum, micro_per_epoch - window_start)
            window_loss, w_correct, w_margin, w_chosen, w_rejected, w_n = 0.0, 0.0, 0.0, 0.0, 0.0, 0
            for _ in range(n_micro):
                batch = next(iterator)
                with autocast(policy_device, use_fp16):
                    pc, pr, rc, rr = policy_and_reference_logps(policy, reference, batch)
                loss = dpo_loss_from_logps(pc, pr, rc, rr, args.beta)
                scaler.scale(loss / n_micro).backward()
                stats = dpo_metrics(pc, pr, rc, rr, args.beta)
                window_loss += loss.item() / n_micro
                w_correct += stats["correct"].sum().item()
                w_margin += stats["margins"].sum().item()
                w_chosen += stats["chosen_rewards"].sum().item()
                w_rejected += stats["rejected_rewards"].sum().item()
                w_n += pc.shape[0]

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
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
                    "epoch": epoch + (window_start + n_micro) / micro_per_epoch,
                    "loss": window_loss,
                    "reward_accuracy": w_correct / w_n,
                    "reward_margin": w_margin / w_n,
                    "chosen_reward": w_chosen / w_n,
                    "rejected_reward": w_rejected / w_n,
                    "lr": lr,
                    "grad_norm": float(grad_norm),
                    "skipped_step": bool(skipped),
                }
            )
            if step % 10 == 0:
                rate = step / (time.time() - start)
                print(
                    f"step {step}/{total_steps} | loss {window_loss:.4f} | acc {w_correct / w_n:.3f} | "
                    f"margin {w_margin / w_n:.4f} | gn {float(grad_norm):.2f} | {rate:.3f} it/s"
                )

            if step % args.eval_every == 0 or step == total_steps:
                metrics = evaluate(policy, reference, val_loader, args.beta, policy_device, use_fp16)
                key = (metrics["val_reward_accuracy"], metrics["val_reward_margin"])
                is_best = key > best_key
                logger.log({"type": "eval", "step": step, **metrics, "is_best": is_best})
                print(f"step {step} | {metrics} | best={is_best}")
                if is_best:
                    best_key, best_step = key, step
                    if not args.no_save:
                        save_checkpoint(policy, tokenizer, out_dir / "best", {"step": step, **metrics})

    if not args.no_save:
        save_checkpoint(policy, tokenizer, out_dir / "final", {"step": step})
    summary = {
        "best_step": best_step,
        "best_val_reward_accuracy": best_key[0],
        "best_val_reward_margin": best_key[1],
        "final_step": step,
        "train_time_sec": time.time() - start,
    }
    write_json(out_dir / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
