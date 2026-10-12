"""Write a single-file, pdfLaTeX-compatible version of the report for Overleaf.

    python report/make_overleaf.py

Output: report/overleaf/main.tex and report/overleaf/figures/*.png (upload the
whole folder to Overleaf). All numbers in the tables are read from results/.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from report.build_report import PRIMARY, RES, agreement, debiased, judged, load_json, outcome  # noqa: E402
from scripts.judge_pairwise_kaggle import summarize  # noqa: E402

OUT = ROOT / "report" / "overleaf"
ASCII = {"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "..."}
SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
           "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "<": r"\textless{}", ">": r"\textgreater{}", "|": r"\textbar{}", '"': "''"}


def tex(text) -> str:
    text = "".join(ASCII.get(ch, ch) for ch in str(text))
    text = "".join(ch for ch in text if ord(ch) < 128)
    return "".join(SPECIAL.get(ch, ch) for ch in text)


def pct(x, digits=1):
    return "--" if x is None else f"{100 * x:.{digits}f}\\%"


def table(spec, header, rows, size=r"\small"):
    """Fully ruled table: vertical lines between all columns and a line under every row."""
    lines = [size, r"\begin{tabularx}{\linewidth}{" + spec + "}", r"\hline", " & ".join(header) + r" \\ \hline"]
    lines += [" & ".join(str(c) for c in row) + r" \\ \hline" for row in rows]
    lines += [r"\end{tabularx}"]
    return "\n".join(lines)


def judge_row(label, stem, eval_set):
    rows = judged(PRIMARY, stem)
    s = summarize(rows)
    swapped = judged(PRIMARY, stem + "_swapped")
    d = debiased(rows, swapped) if swapped else None
    sec = judged(RES / "eval" / "judge_gemini-3.7-flash", stem)
    agree, _ = agreement(rows, sec) if sec else (None, 0)
    cells = [label, eval_set, f"{s['wins']}/{s['losses']}/{s['ties']}", f"{s['unknown']}/{s['failures']}",
             pct(s["win_rate_ties_as_half"]), pct(d["consistency"]) if d else "--",
             f"{d['wins']}/{d['losses']}/{d['ties']}" if d else "--", pct(d["win_rate_ties_as_half"]) if d else "--", pct(agree)]
    return cells, {"rows": rows, "swapped": swapped}


def breakdown(info, key=None):
    swapped = {r["id"]: outcome(r) for r in info["swapped"]}
    cats = {}
    for r in info["rows"]:
        a, b = outcome(r), swapped.get(r["id"])
        verdict = (a if a == b else "tie") if (a and b) else a
        if verdict:
            c = cats.setdefault(key[r["id"]] if key else r["category"], {"win": 0, "loss": 0, "tie": 0})
            c[verdict] += 1
    return [[tex(k.replace("_", " ")), sum(c.values()), c["win"], c["loss"], c["tie"], pct((c["win"] + 0.5 * c["tie"]) / sum(c.values()), 0)]
            for k, c in sorted(cats.items())]


def example(gen_a, gen_b, item_id, label_a, label_b, chars=200):
    a = {r["id"]: r for r in load_json(RES / "generations" / f"{gen_a}.json")}[item_id]
    b = {r["id"]: r for r in load_json(RES / "generations" / f"{gen_b}.json")}[item_id]

    def clip(text):
        text = " ".join(text.split())
        return tex(text[:chars] + (" ..." if len(text) > chars else ""))

    return (
        r"\begin{quote}\footnotesize" + "\n"
        + f"Prompt: {tex(a['instruction'][:260])}\\\\[2pt]\n"
        + f"{label_a}: {clip(a['output'])}\\\\[2pt]\n"
        + f"{label_b}: {clip(b['output'])}\n"
        + r"\end{quote}"
    )


def main():
    manifest = load_json(RES / "data" / "data_manifest.json")
    stats = load_json(RES / "eval" / "response_stats.json")
    sft_cfg, hh_cfg, uf_cfg = (load_json(RES / n / "run_config.json") for n in ("sft", "dpo_hh", "dpo_uf"))
    sft_sum, hh_sum, uf_sum = (load_json(RES / n / "summary.json") for n in ("sft", "dpo_hh", "dpo_uf"))
    lora = {s["name"]: s for s in load_json(RES / "lora_dpo" / "sweep_summary.json")}
    env = sft_cfg["environment"]
    T = {}

    T["data"] = table(r"|c|C|c|c|", ["Track", "Source", "Split", "Train / val"], [
        ["SFT", r"HuggingFaceH4/ultrachat\_200k", r"train\_sft / test\_sft", f"{manifest['sft']['train']['kept']:,} / {manifest['sft']['val']['kept']:,}"],
        ["HH-RLHF DPO", "Anthropic/hh-rlhf", "train / test", f"{manifest['dpo_hh']['train']['kept']:,} / {manifest['dpo_hh']['val']['kept']:,}"],
        ["UltraFeedback DPO", r"HuggingFaceH4/ultrafeedback\_binarized", r"train\_prefs / test\_prefs", f"{manifest['dpo_uf']['train']['kept']:,} / {manifest['dpo_uf']['val']['kept']:,}"],
        ["LoRA persona", r"ramsay\_style\_preferences.json (provided)", "random 200 / 20", "200 / 20"],
    ])

    T["config"] = table(r"|>{\centering\arraybackslash}m{2.6cm}|*{4}{C|}", ["", "SFT", "DPO-HH", "DPO-UF", "LoRA-DPO (TRL)"], [
        ["Initial weights", "Qwen2.5-0.5B base", "SFT checkpoint (step 938)", "SFT checkpoint (step 938)", "SmolLM2-360M-Instruct"],
        ["Trainable params", "full, 494M", "full, 494M", "full, 494M", f"LoRA r=16, alpha=32, dropout 0.05, all 7 projections ({lora['r16_all_lr5e-5']['trainable_params'] / 1e6:.2f}M)"],
        ["Train / val", f"{sft_cfg['n_train']:,} / {sft_cfg['n_val']:,}", f"{hh_cfg['n_train_pairs']:,} / {hh_cfg['n_val_pairs']:,}", f"{uf_cfg['n_train_pairs']:,} / {uf_cfg['n_val_pairs']:,}", "200 / 20"],
        ["Optimizer", "AdamW", "AdamW", "AdamW", "AdamW"],
        ["Learning rate", f"2e-5, {sft_cfg['warmup_steps']} warmup steps (3\\%), cosine", "1e-6, constant", "1e-6, constant", r"5e-5, 10\% warmup, cosine"],
        ["Weight decay / clip", "0.1 / 1.0", "0 / 1.0", "0 / 1.0", "0 / 1.0"],
        ["Batch", "2 x 16 = 32", "1 pair x 32 = 32", "1 pair x 32 = 32", "4 x 2 = 8"],
        ["Steps / epochs", f"{sft_cfg['total_steps']} / 1", f"{hh_cfg['total_steps']} / 1", f"{uf_cfg['total_steps']} / 1", f"{lora['r16_all_lr5e-5']['optimizer_steps']} / 3"],
        ["Precision", "fp32 weights, fp16 autocast", "same as SFT, reference in fp32", "same as SFT, reference in fp32", "fp32 weights, fp16"],
        ["Max length (prompt)", "512", "512 (256)", "512 (256)", "512 (256)"],
        ["Beta / reference", "--", "0.1 / frozen SFT", "0.1 / frozen SFT", "0.1 / base model"],
        ["Seed / time", f"42 / {sft_sum['train_time_sec'] / 60:.0f} min", f"42 / {hh_sum['train_time_sec'] / 60:.0f} min", f"42 / {uf_sum['train_time_sec'] / 60:.0f} min", "42 / about 2 min each"],
    ])
    T["software"] = f"PyTorch {tex(env['torch'])}, Transformers {env['transformers']}, Datasets {env['datasets']}, 2 x Tesla T4 on Kaggle"

    def two(a, b):
        return r"\begin{tabular}[c]{@{}c@{}}" + a + r"\\" + b + r"\end{tabular}"

    header = ["Comparison", "Eval set", "W/L/T", two("Unk./", "fail."), two("Win", "rate"), two("Same after", "swap"),
              two("Both orders", "W/L/T"), two("Both orders", "win rate"), two("Agree", "with 3.7")]
    spec = r"|c|C|c|c|c|c|c|c|c|"
    main_rows, infos = [], {}
    for label, stem, ev in (("DPO-HH vs SFT", "dpo_hh_vs_sft_hh", "HH (300)"), ("DPO-UF vs SFT", "dpo_uf_vs_sft_alpaca", "Alpaca (300)")):
        cells, info = judge_row(label, stem, ev)
        main_rows.append(cells)
        infos[stem] = info
    T["judge"] = table(spec, header, main_rows, r"\footnotesize")

    prow = []
    for cfg, label in (("r8_attn_lr5e-5", "r8, attention only, LR 5e-5"), ("r16_all_lr5e-5", "r16, all layers, LR 5e-5 (submitted)"), ("r16_all_lr2e-4", "r16, all layers, LR 2e-4")):
        row = [label, f"{lora[cfg]['final_eval']['eval_loss']:.3f}", f"{lora[cfg]['final_eval']['eval_rewards/margins']:.2f}"]
        for rub in ("persona_style", "persona_quality"):
            stem = f"{rub}_lora_{cfg}_vs_base"
            d = debiased(judged(PRIMARY, stem), judged(PRIMARY, stem + "_swapped"))
            row += [f"{d['wins']}/{d['losses']}/{d['ties']}", pct(d["win_rate_ties_as_half"])]
        prow.append(row)
    T["persona"] = table(r"|C|c|c|c|c|c|c|", ["LoRA config", "Val loss", "Val margin", "Style W/L/T", "Style win", "Useful W/L/T", "Useful win"], prow, r"\footnotesize")

    T["hhcat"] = table(r"|C|c|c|c|c|c|", ["HH category", "n", "W", "L", "T", "Win"], breakdown(infos["dpo_hh_vs_sft_hh"]), r"\footnotesize")
    src = {r["id"]: r.get("source_dataset") for r in load_json(RES / "generations" / "alpaca_sft.json")}
    T["ufsrc"] = table(r"|C|c|c|c|c|c|", ["AlpacaEval source", "n", "W", "L", "T", "Win"], breakdown(infos["dpo_uf_vs_sft_alpaca"], src), r"\footnotesize")
    srows = []
    for name, label in (("hh_base", "Base, HH"), ("hh_sft", "SFT, HH"), ("hh_dpo_hh", "DPO-HH, HH"), ("hh_dpo_uf", "DPO-UF, HH"),
                        ("alpaca_base", "Base, Alpaca"), ("alpaca_sft", "SFT, Alpaca"), ("alpaca_dpo_hh", "DPO-HH, Alpaca"), ("alpaca_dpo_uf", "DPO-UF, Alpaca")):
        s = stats[name]
        srows.append([label, f"{s['mean_words']:.0f}", pct(s["unfinished_rate"]), pct(s["refusal_marker_rate"])])
    T["stats"] = table(r"|C|c|c|c|", ["Model, eval set", "Mean words", "Hit 512 limit", "Refusal phrases"], srows, r"\footnotesize")

    T["examples"] = "\n\n".join([
        "Example 1 (HH, hate or harassment). DPO-HH won in both orders.\n" + example("hh_dpo_hh", "hh_sft", "hh-eval-42-0004", "DPO-HH", "SFT"),
        "Example 2 (HH, harmful instructions). Tie, the judge said both models failed.\n" + example("hh_dpo_hh", "hh_sft", "hh-eval-42-0000", "DPO-HH", "SFT"),
        "Example 3 (AlpacaEval). DPO-UF lost in both orders.\n" + example("alpaca_dpo_uf", "alpaca_sft", "alpaca-eval-228", "DPO-UF", "SFT"),
        "Example 4 (persona, safety). Submitted LoRA adapter vs. the base model.\n" + example("persona_lora_r16_all_lr5e-5", "persona_base", "persona-eval-30", "LoRA (LR 5e-5)", "Base"),
        "Example 5 (persona, safety). The LR 2e-4 adapter vs. the base model.\n" + example("persona_lora_r16_all_lr2e-4", "persona_base", "persona-eval-36", "LoRA (LR 2e-4)", "Base"),
    ])

    template = (ROOT / "report" / "overleaf_template.tex").read_text()
    for key, value in T.items():
        template = template.replace(f"%%{key.upper()}%%", value)
    assert "%%" not in template.replace("%%%", ""), "unfilled placeholder"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "main.tex").write_text(template)
    figs = OUT / "figures"
    figs.mkdir(exist_ok=True)
    for name in ("sft_loss.png", "dpo_curves.png", "lora_dpo_curves.png"):
        shutil.copy(RES / "plots" / name, figs / name)
    non_ascii = [c for c in template if ord(c) > 127]
    print(f"wrote {OUT / 'main.tex'} ({len(template)} chars, non-ASCII: {len(non_ascii)})")


if __name__ == "__main__":
    main()
