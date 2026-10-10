"""Batched deterministic generation on a fixed evaluation set.

Writes a JSON list of ``{"id", "instruction", "output", "generator",
"dataset", ...}`` records plus a ``<output>.meta.json`` sidecar recording the
checkpoint and decoding settings.

Prompt formats:

* ``chatml`` - the template used for SFT/DPO training of Qwen2.5-0.5B
  (``alignment.prompting.format_prompt``), used for base, SFT and DPO models;
* ``chat_template`` - the tokenizer's own chat template (SmolLM2-Instruct,
  matching how TRL formats conversational DPO data).

Usage::

    python -m alignment.generate --model_path outputs/sft/best \
        --eval_path eval_sets/alpaca_eval_300_seed42.json \
        --output_path outputs/generations/alpaca_sft.json --generator qwen2.5-0.5b-sft
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from alignment.prompting import format_prompt, prepare_tokenizer, stop_token_ids
from alignment.train_utils import write_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", default=None, help="optional PEFT/LoRA adapter to merge")
    parser.add_argument("--eval_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--generator", required=True, help="model label stored in every record")
    parser.add_argument("--dataset", default=None, help="override the dataset field")
    parser.add_argument("--prompt_format", choices=["chatml", "chat_template"], default="chatml")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None, help="only the first N prompts (sanity checks)")
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args(argv)


def load_eval_set(path):
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    for row in rows:
        if "instruction" not in row:
            row["instruction"] = row["prompt"]
    return rows


def build_prompts(tokenizer, rows, prompt_format):
    if prompt_format == "chatml":
        return [format_prompt(r["instruction"]) for r in rows]
    return [
        tokenizer.apply_chat_template([{"role": "user", "content": r["instruction"]}], tokenize=False, add_generation_prompt=True)
        for r in rows
    ]


def load_model_and_tokenizer(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if args.prompt_format == "chatml":
        tokenizer = prepare_tokenizer(tokenizer)
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype, attn_implementation="sdpa")
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path).merge_and_unload()
    model.to(args.device).eval()
    model.config.use_cache = True
    return model, tokenizer


@torch.no_grad()
def generate_all(model, tokenizer, prompts, args):
    tokenizer.padding_side = "left"
    stop_ids = stop_token_ids(tokenizer) if args.prompt_format == "chatml" else [tokenizer.eos_token_id]
    order = sorted(range(len(prompts)), key=lambda i: -len(tokenizer.encode(prompts[i], add_special_tokens=False)))
    outputs, finished = [None] * len(prompts), [None] * len(prompts)
    for start in range(0, len(order), args.batch_size):
        idx = order[start : start + args.batch_size]
        enc = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(args.device) for k, v in enc.items()}
        kwargs = dict(
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=stop_ids,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=args.repetition_penalty,
        )
        if args.no_repeat_ngram_size:
            kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size
        generated = model.generate(**enc, **kwargs)
        new_tokens = generated[:, enc["input_ids"].shape[1] :]
        for row, i in enumerate(idx):
            tokens = new_tokens[row].tolist()
            hit_stop = any(t in stop_ids for t in tokens)
            if hit_stop:
                tokens = tokens[: min(tokens.index(t) for t in stop_ids if t in tokens)]
            outputs[i] = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            finished[i] = hit_stop
        print(f"generated {min(start + args.batch_size, len(order))}/{len(order)}", flush=True)
    tokenizer.padding_side = "right"
    return outputs, finished


def main(argv=None):
    args = parse_args(argv)
    out_path = Path(args.output_path)
    if args.skip_existing and out_path.exists():
        print(f"skip existing {out_path}")
        return
    rows = load_eval_set(args.eval_path)[: args.limit]
    model, tokenizer = load_model_and_tokenizer(args)
    prompts = build_prompts(tokenizer, rows, args.prompt_format)
    start = time.time()
    outputs, finished = generate_all(model, tokenizer, prompts, args)
    records = []
    for row, output, done in zip(rows, outputs, finished):
        record = {
            "id": row["id"],
            "instruction": row["instruction"],
            "output": output,
            "generator": args.generator,
            "dataset": args.dataset or row.get("dataset", Path(args.eval_path).stem),
            "finished": done,
        }
        for key in ("category", "expected_behavior", "source_dataset"):
            if key in row:
                record[key] = row[key]
        records.append(record)
    write_json(out_path, records)
    meta = {
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "eval_path": args.eval_path,
        "generator": args.generator,
        "n": len(records),
        "prompt_format": args.prompt_format,
        "decoding": {
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "dtype": args.dtype,
        },
        "unfinished": sum(1 for f in finished if not f),
        "seconds": round(time.time() - start, 1),
    }
    write_json(out_path.with_suffix(".meta.json"), meta)
    print(meta)


if __name__ == "__main__":
    main()
