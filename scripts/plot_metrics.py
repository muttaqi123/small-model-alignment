"""Training plots for SFT, both from-scratch DPO runs, and the LoRA-DPO sweep.

    python scripts/plot_metrics.py --sft outputs/sft --dpo outputs/dpo_hh outputs/dpo_uf \
        --lora outputs/lora_dpo --out_dir outputs/plots

Train curves are shown raw (faint) and with an exponential moving average
(solid, ``--ema`` weight); validation points are unsmoothed. The selected
``best`` checkpoint is marked with a star.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"train": "#4C72B0", "val": "#DD8452", "hh": "#55A868", "uf": "#8172B3"}


def read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ema(values, weight):
    out, last = [], None
    for v in values:
        last = v if last is None else weight * last + (1 - weight) * v
        out.append(last)
    return out


def split(records):
    return [r for r in records if r["type"] == "train"], [r for r in records if r["type"] == "eval"]


def best_step(run_dir):
    summary = json.loads((Path(run_dir) / "summary.json").read_text())
    return summary.get("best_step")


def plot_sft(run_dir, out_dir, smooth):
    train, evals = split(read_jsonl(Path(run_dir) / "metrics.jsonl"))
    best = best_step(run_dir)
    fig, ax = plt.subplots(figsize=(7, 4))
    steps = [r["step"] for r in train]
    losses = [r["loss"] for r in train]
    ax.plot(steps, losses, color=COLORS["train"], alpha=0.25, lw=0.8)
    ax.plot(steps, ema(losses, smooth), color=COLORS["train"], lw=1.8, label=f"train loss (EMA {smooth})")
    ax.plot([r["step"] for r in evals], [r["val_loss"] for r in evals], "o-", color=COLORS["val"], label="validation loss")
    for r in evals:
        if r["step"] == best:
            ax.plot(r["step"], r["val_loss"], "*", ms=16, color="crimson", label=f"best checkpoint (step {best})")
    ax.set_xlabel("optimizer step (effective batch 32)")
    ax.set_ylabel("response-token NLL (nats/token)")
    ax.set_title("SFT: Qwen2.5-0.5B on 30k UltraChat examples")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "sft_loss.png", dpi=160)
    plt.close(fig)


def plot_dpo(run_dirs, out_dir, smooth):
    names = {str(d): Path(d).name.replace("dpo_", "").upper() for d in run_dirs}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for run_dir in run_dirs:
        key = Path(run_dir).name.replace("dpo_", "")
        color = COLORS.get(key, None)
        label = names[str(run_dir)]
        train, evals = split(read_jsonl(Path(run_dir) / "metrics.jsonl"))
        best = best_step(run_dir)
        steps = [r["step"] for r in train]
        for ax, train_key, val_key, ylabel in (
            (axes[0], "loss", "val_loss", "DPO loss"),
            (axes[1], "reward_accuracy", "val_reward_accuracy", "reward accuracy (r_w > r_l)"),
            (axes[2], "reward_margin", "val_reward_margin", "reward margin (r_w - r_l)"),
        ):
            values = [r[train_key] for r in train]
            ax.plot(steps, ema(values, smooth), color=color, lw=1.2, alpha=0.6, label=f"{label} train (EMA {smooth})")
            ax.plot([r["step"] for r in evals], [r[val_key] for r in evals], "o-", color=color, lw=2, label=f"{label} validation")
            for r in evals:
                if r["step"] == best:
                    ax.plot(r["step"], r[val_key], "*", ms=15, color="crimson")
            ax.set_xlabel("optimizer step (effective batch 32)")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
    axes[0].axhline(0.6931, ls="--", color="gray", lw=1, label="log 2 (initial loss)")
    axes[0].legend(fontsize=8)
    fig.suptitle("From-scratch DPO (beta = 0.1, lr = 1e-6); stars mark the best-accuracy checkpoint")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "dpo_curves.png", dpi=160)
    plt.close(fig)


def plot_lora(lora_dir, out_dir):
    runs = sorted(p for p in Path(lora_dir).iterdir() if (p / "metrics.jsonl").exists())
    if not runs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for run in runs:
        records = read_jsonl(run / "metrics.jsonl")
        evals = [r for r in records if "eval_loss" in r]
        train = [r for r in records if "loss" in r and "eval_loss" not in r]
        axes[0].plot([r["step"] for r in train], [r["loss"] for r in train], lw=1, alpha=0.8, label=f"{run.name} train")
        axes[0].plot([r["step"] for r in evals], [r["eval_loss"] for r in evals], "o--", label=f"{run.name} held-out")
        axes[1].plot([r["step"] for r in evals], [r.get("eval_rewards/accuracies") for r in evals], "o-", label=run.name)
        axes[2].plot([r["step"] for r in evals], [r.get("eval_rewards/margins") for r in evals], "o-", label=run.name)
    for ax, ylabel in zip(axes, ("DPO loss", "held-out reward accuracy", "held-out reward margin")):
        ax.set_xlabel("optimizer step (effective batch 8)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.suptitle("TRL DPOTrainer + LoRA on SmolLM2-360M-Instruct (200 train / 20 held-out persona pairs)")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "lora_dpo_curves.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", default=None)
    parser.add_argument("--dpo", nargs="*", default=[])
    parser.add_argument("--lora", default=None)
    parser.add_argument("--out_dir", default="outputs/plots")
    parser.add_argument("--ema", type=float, default=0.9)
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.sft:
        plot_sft(args.sft, args.out_dir, args.ema)
    if args.dpo:
        plot_dpo(args.dpo, args.out_dir, args.ema)
    if args.lora:
        plot_lora(args.lora, args.out_dir)
    print(f"plots written to {args.out_dir}")


if __name__ == "__main__":
    main()
