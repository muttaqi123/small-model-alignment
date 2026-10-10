"""Aggregate judged pair files into tables (Markdown + JSON).

    python scripts/summarize_results.py --judged_dir results/eval/judge_gemini-3.6-flash \
        --second_judge_dir results/eval/judge_gemini-3.7-flash --out results/eval/summary

Reports, per comparison: n, wins / losses / ties / unknown / failures, win rate
excluding ties, win rate with ties counted as half, and per-category results.
If swapped-order files (``*::swapped`` pair ids) are present, it also reports
how often the judge's verdict survives swapping A and B; with a second judge
directory it reports judge-judge agreement on the same pairs.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scripts.judge_pairwise_kaggle import read_jsonl, summarize


def outcome(row):
    """Verdict from the target model's perspective: win / loss / tie / None."""

    if row["decision"] in ("unknown", "failure"):
        return None
    if row["decision"] == "tie":
        return "tie"
    return "win" if row["winner_model"] == row["target_model"] else "loss"


def load_dir(directory):
    files = {}
    for path in sorted(Path(directory).glob("*_judged.jsonl")):
        files[path.name.replace("_judged.jsonl", "")] = read_jsonl(path)
    return files


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judged_dir", required=True)
    parser.add_argument("--second_judge_dir", default=None)
    parser.add_argument("--out", default="results/eval/summary")
    args = parser.parse_args()

    primary = load_dir(args.judged_dir)
    secondary = load_dir(args.second_judge_dir) if args.second_judge_dir else {}
    report = {}
    lines = [
        "| comparison | n | wins | losses | ties | unknown | failures | win rate (excl. ties) | win rate (ties = 1/2) | judge picked A |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, rows in primary.items():
        if name.endswith("_swapped"):
            continue
        s = summarize(rows)
        entry = {"summary": s}
        lines.append(
            f"| {name} | {s['n']} | {s['wins']} | {s['losses']} | {s['ties']} | {s['unknown']} | {s['failures']} | "
            f"{pct(s['win_rate_excluding_ties'])} | {pct(s['win_rate_ties_as_half'])} | {pct(s['judge_picked_A_fraction'])} |"
        )
        swapped = primary.get(name + "_swapped")
        if swapped:
            by_id = {r["id"]: outcome(r) for r in swapped}
            same = [outcome(r) == by_id.get(r["id"]) for r in rows if outcome(r) and by_id.get(r["id"])]
            merged = []
            for r in rows:
                a, b = outcome(r), by_id.get(r["id"])
                if a and b:
                    merged.append(a if a == b else "tie")
            entry["swap_consistency"] = sum(same) / len(same) if same else None
            entry["order_debiased"] = {
                "n": len(merged),
                "wins": merged.count("win"),
                "losses": merged.count("loss"),
                "ties": merged.count("tie"),
                "win_rate_ties_as_half": (merged.count("win") + 0.5 * merged.count("tie")) / len(merged) if merged else None,
            }
        if name in secondary:
            other = {r["id"]: outcome(r) for r in secondary[name]}
            agree = [outcome(r) == other.get(r["id"]) for r in rows if outcome(r) and other.get(r["id"])]
            entry["second_judge"] = {
                "judge": summarize(secondary[name]).get("judge_models"),
                "summary": summarize(secondary[name]),
                "agreement": sum(agree) / len(agree) if agree else None,
            }
        report[name] = entry

    extra = ["", "| comparison | swap consistency | order-debiased W/L/T | order-debiased win rate | second-judge win rate | judge agreement |", "|---|---|---|---|---|---|"]
    for name, entry in report.items():
        d = entry.get("order_debiased")
        sj = entry.get("second_judge")
        extra.append(
            f"| {name} | {pct(entry.get('swap_consistency'))} | "
            f"{'-' if not d else f'{d['wins']}/{d['losses']}/{d['ties']}'} | {pct(d['win_rate_ties_as_half']) if d else '-'} | "
            f"{pct(sj['summary']['win_rate_ties_as_half']) if sj else '-'} | {pct(sj['agreement']) if sj else '-'} |"
        )

    category_lines = []
    for name, entry in report.items():
        cats = entry["summary"].get("by_category")
        if not cats:
            continue
        category_lines += ["", f"**{name}** by category", "", "| category | n | W | L | T | win rate (ties = 1/2) |", "|---|---|---|---|---|---|"]
        for cat, c in cats.items():
            category_lines.append(f"| {cat} | {c['n']} | {c['wins']} | {c['losses']} | {c['ties']} | {pct(c['win_rate_ties_as_half'])} |")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    out.with_suffix(".md").write_text("\n".join(lines + extra + category_lines) + "\n")
    print("\n".join(lines + extra))


if __name__ == "__main__":
    main()
