"""Create the fixed 300-prompt AlpacaEval subset (seed 42).

    python scripts/make_alpaca_subset.py --out eval_sets/alpaca_eval_300_seed42.json

Source: ``alpaca_eval.json`` from the ``tatsu-lab/alpaca_eval`` dataset repo
(805 prompts). Selection: ``random.Random(42).sample(range(805), 300)``; the
order of the sample is kept.
"""

from __future__ import annotations

import argparse
import json

from alignment.data import sample_alpaca_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None, help="local alpaca_eval.json (downloaded from the Hub if omitted)")
    parser.add_argument("--out", default="eval_sets/alpaca_eval_300_seed42.json")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = args.source
    if source is None:
        from huggingface_hub import hf_hub_download

        source = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")
    with open(source, encoding="utf-8") as handle:
        rows = json.load(handle)
    subset = sample_alpaca_eval(rows, args.n, args.seed)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(subset, handle, indent=2, ensure_ascii=False)
    print(f"wrote {len(subset)} prompts (of {len(rows)}) to {args.out}")


if __name__ == "__main__":
    main()
