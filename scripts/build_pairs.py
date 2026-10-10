"""Build blinded pairwise-comparison JSONL files from two generation files.

    python scripts/build_pairs.py \
        --model_path outputs/generations/alpaca_dpo_uf.json --model_name dpo_uf \
        --baseline_path outputs/generations/alpaca_sft.json --baseline_name sft \
        --rubric general --output_path outputs/eval/dpo_uf_vs_sft_pairs.jsonl

For every prompt present in both files a seeded coin flip decides which model
is shown as "Response A". The judge only ever sees ``instruction``,
``response_a``, ``response_b`` (and, for the safety set, the prompt
category's expected behaviour). The A/B -> model mapping is stored under
``hidden`` and is used only after judging. ``--swapped_output_path``
additionally writes the same pairs with A and B exchanged, for a
position-bias check.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load(path):
    with open(path, encoding="utf-8") as handle:
        return {row["id"]: row for row in json.load(handle)}


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--baseline_path", required=True)
    parser.add_argument("--baseline_name", required=True)
    parser.add_argument("--rubric", choices=["general", "safety", "persona_style", "persona_quality"], required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--swapped_output_path", default=None)
    parser.add_argument("--comparison", default=None, help="comparison label (default: <model>_vs_<baseline>)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_rows, baseline_rows = load(args.model_path), load(args.baseline_path)
    ids = [i for i in baseline_rows if i in model_rows]
    missing = sorted(set(baseline_rows) ^ set(model_rows))
    if missing:
        print(f"warning: {len(missing)} ids are not in both files and are skipped")
    rng = random.Random(args.seed)
    comparison = args.comparison or f"{args.model_name}_vs_{args.baseline_name}"
    pairs, swapped = [], []
    for item_id in ids:
        model_row, base_row = model_rows[item_id], baseline_rows[item_id]
        if model_row["instruction"] != base_row["instruction"]:
            raise ValueError(f"instruction mismatch for {item_id}")
        model_first = rng.random() < 0.5
        a_row, b_row = (model_row, base_row) if model_first else (base_row, model_row)
        a_name, b_name = (args.model_name, args.baseline_name) if model_first else (args.baseline_name, args.model_name)
        pair = {
            "pair_id": f"{comparison}::{item_id}",
            "id": item_id,
            "rubric": args.rubric,
            "instruction": model_row["instruction"],
            "category": model_row.get("category"),
            "expected_behavior": model_row.get("expected_behavior"),
            "response_a": a_row["output"],
            "response_b": b_row["output"],
            "hidden": {
                "comparison": comparison,
                "model_a": a_name,
                "model_b": b_name,
                "target_model": args.model_name,
                "baseline_model": args.baseline_name,
            },
        }
        pairs.append(pair)
        swapped.append(
            {
                **pair,
                "pair_id": pair["pair_id"] + "::swapped",
                "response_a": pair["response_b"],
                "response_b": pair["response_a"],
                "hidden": {**pair["hidden"], "model_a": b_name, "model_b": a_name, "swapped": True},
            }
        )
    write_jsonl(args.output_path, pairs)
    print(f"wrote {len(pairs)} pairs to {args.output_path} ({sum(p['hidden']['model_a'] == args.model_name for p in pairs)} with {args.model_name} as A)")
    if args.swapped_output_path:
        write_jsonl(args.swapped_output_path, swapped)
        print(f"wrote swapped copy to {args.swapped_output_path}")


if __name__ == "__main__":
    main()
