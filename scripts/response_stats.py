"""Descriptive statistics of saved generations (length, truncation, refusal markers).

    python scripts/response_stats.py --out results/eval/response_stats

The refusal detector is a keyword heuristic (phrases such as "I cannot",
"I'm sorry, but", "not appropriate"); it is only a coarse signal used to
describe behaviour and is never used to score models.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

REFUSAL = re.compile(
    r"\b(i can(?:no|')t|i cannot|i am unable|i'm unable|i won't|i will not|i'm sorry, but|i am sorry, but|"
    r"as an ai|not appropriate|i must decline|i do not condone|i don't condone|is illegal and unethical|"
    r"cannot (?:help|assist|provide)|can't (?:help|assist|provide))\b",
    re.I,
)

FILES = {
    "hh": ["hh_base", "hh_sft", "hh_dpo_hh", "hh_dpo_uf"],
    "alpaca": ["alpaca_base", "alpaca_sft", "alpaca_dpo_hh", "alpaca_dpo_uf"],
    "persona": ["persona_base", "persona_lora_r8_attn_lr5e-5", "persona_lora_r16_all_lr5e-5", "persona_lora_r16_all_lr2e-4"],
}


def stats(rows):
    words = [len(r["output"].split()) for r in rows]
    return {
        "n": len(rows),
        "mean_words": sum(words) / len(words),
        "median_words": sorted(words)[len(words) // 2],
        "unfinished_rate": sum(not r["finished"] for r in rows) / len(rows),
        "refusal_marker_rate": sum(bool(REFUSAL.search(r["output"][:400])) for r in rows) / len(rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", default="results/generations")
    parser.add_argument("--out", default="results/eval/response_stats")
    args = parser.parse_args()
    report = {}
    for group, names in FILES.items():
        for name in names:
            path = Path(args.generations) / f"{name}.json"
            if not path.exists():
                continue
            rows = json.loads(path.read_text())
            entry = stats(rows)
            by_cat = defaultdict(list)
            for r in rows:
                if r.get("category"):
                    by_cat[r["category"]].append(r)
            if by_cat:
                entry["by_category"] = {c: stats(v) for c, v in sorted(by_cat.items())}
            report[name] = entry
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    lines = ["| generations | n | mean words | median words | hit length limit | refusal-marker rate |", "|---|---|---|---|---|---|"]
    for name, e in report.items():
        lines.append(
            f"| {name} | {e['n']} | {e['mean_words']:.0f} | {e['median_words']} | {100 * e['unfinished_rate']:.1f}% | {100 * e['refusal_marker_rate']:.1f}% |"
        )
    cats = sorted({c for n in FILES["hh"] if n in report for c in report[n].get("by_category", {})})
    if cats:
        header = [n for n in FILES["hh"] if n in report]
        lines += ["", "Refusal-marker rate on the HH evaluation set by category", "", "| category | " + " | ".join(header) + " |", "|---" * (len(header) + 1) + "|"]
        for c in cats:
            lines.append(f"| {c} | " + " | ".join(f"{100 * report[n]['by_category'][c]['refusal_marker_rate']:.0f}%" for n in header) + " |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
