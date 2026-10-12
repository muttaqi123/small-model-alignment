"""Generate the LaTeX tables and example boxes used by report/latex/main.tex.

    python report/make_tex_tables.py && tectonic report/latex/main.tex

All numbers come from the files under results/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from report.build_report import (  # noqa: E402
    PRIMARY,
    RES,
    agreement,
    debiased,
    judged,
    load_json,
    outcome,
)
from scripts.judge_pairwise_kaggle import summarize  # noqa: E402

OUT = ROOT / "report" / "latex" / "generated"

SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
           "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def tex(text) -> str:
    return "".join(SPECIAL.get(ch, ch) for ch in str(text))


def pct(x, digits=1):
    return "--" if x is None else f"{100 * x:.{digits}f}\\%"


def write(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text(body.strip() + "\n")


def tabular(spec, header, rows, size=r"\footnotesize"):
    lines = [size, r"\begin{tabularx}{\linewidth}{" + spec + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(str(c) for c in row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabularx}"]
    return "\n".join(lines)


def data_table(manifest):
    rows = [
        ["SFT", r"HuggingFaceH4/ultrachat\_200k", r"train\_sft / test\_sft",
         r"first user turn + first reply; full sequence $\le$ 512 tokens; dedup by prompt",
         f"{manifest['sft']['train']['kept']:,} / {manifest['sft']['val']['kept']:,}"],
        ["HH-RLHF DPO", r"Anthropic/hh-rlhf (all subsets)", r"train / test",
         r"exactly one Human + Assistant turn on both sides, same prompt, different replies; prompt $\le$ 256, each side $\le$ 512",
         f"{manifest['dpo_hh']['train']['kept']:,} / {manifest['dpo_hh']['val']['kept']:,}"],
        ["UltraFeedback DPO", r"HuggingFaceH4/ ultrafeedback\_binarized", r"train\_prefs / test\_prefs",
         r"2-message conversations, score\_chosen $>$ score\_rejected, different replies; same length limits",
         f"{manifest['dpo_uf']['train']['kept']:,} / {manifest['dpo_uf']['val']['kept']:,}"],
        ["LoRA persona", r"ramsay\_style\_ preferences.json (provided)", "220 rows", "seeded 200 / 20 split", "200 / 20"],
    ]
    return tabular(r"@{}>{\raggedright\arraybackslash}p{1.9cm}>{\raggedright\arraybackslash}p{3.5cm}>{\raggedright\arraybackslash}p{2.4cm}>{\raggedright\arraybackslash}X@{\hspace{4pt}}r@{}", ["Track", "Source", "Split", "Filtering", "Train / val"], rows)


def config_table():
    sft_cfg, hh_cfg, uf_cfg = (load_json(RES / n / "run_config.json") for n in ("sft", "dpo_hh", "dpo_uf"))
    sft_sum, hh_sum, uf_sum = (load_json(RES / n / "summary.json") for n in ("sft", "dpo_hh", "dpo_uf"))
    lora = {s["name"]: s for s in load_json(RES / "lora_dpo" / "sweep_summary.json")}["r16_all_lr5e-5"]
    rows = [
        ["Initial weights", "Qwen2.5-0.5B (base)", "SFT best (step 938)", "SFT best (step 938)", "SmolLM2-360M-Instruct"],
        ["Trainable params", "full FT, 494M", "full FT, 494M", "full FT, 494M",
         rf"LoRA $r{{=}}16$, $\alpha{{=}}32$, dropout 0.05, q/k/v/o/gate/up/down ({lora['trainable_params'] / 1e6:.2f}M, 2.3\%)"],
        ["Data (train / val)", f"UltraChat {sft_cfg['n_train']:,} / {sft_cfg['n_val']:,}", f"HH-RLHF {hh_cfg['n_train_pairs']:,} / {hh_cfg['n_val_pairs']:,}",
         f"UltraFeedback {uf_cfg['n_train_pairs']:,} / {uf_cfg['n_val_pairs']:,}", "persona 200 / 20"],
        ["Optimizer", r"AdamW ($\beta$ = 0.9, 0.999), fused", r"AdamW ($\beta$ = 0.9, 0.999)", r"AdamW ($\beta$ = 0.9, 0.999)", "AdamW (TRL default)"],
        ["Peak LR / schedule", f"2e-5, warmup {sft_cfg['warmup_steps']} steps (3\\%) + cosine to 0", "1e-6, constant, no warmup",
         "1e-6, constant, no warmup", r"5e-5, 10\% warmup + cosine"],
        ["Weight decay / clip", "0.1 (not on norms/biases) / 1.0", "0 / 1.0", "0 / 1.0", "0 / 1.0"],
        [r"Batch (device $\times$ accum)", r"$2 \times 16 = 32$", r"1 pair $\times$ 32 = 32", r"1 pair $\times$ 32 = 32", r"$4 \times 2 = 8$"],
        ["Steps / epochs", f"{sft_cfg['total_steps']} / 1", f"{hh_cfg['total_steps']} / 1", f"{uf_cfg['total_steps']} / 1", f"{lora['optimizer_steps']} / 3"],
        ["Precision", "fp32 weights, fp16 autocast, GradScaler", "policy as SFT; reference fp32, no grad, cuda:1", "same as DPO-HH", "fp32 weights, fp16 AMP"],
        ["Max length (prompt)", "512", "512 (256)", "512 (256)", "512 (256)"],
        [r"DPO $\beta$ / reference", "--", "0.1 / frozen SFT best", "0.1 / frozen SFT best", "0.1 / base model, adapter disabled"],
        ["Seed / wall-clock", f"42 / {sft_sum['train_time_sec'] / 60:.0f} min", f"42 / {hh_sum['train_time_sec'] / 60:.0f} min",
         f"42 / {uf_sum['train_time_sec'] / 60:.0f} min", r"42 / $\approx$1.5--2 min per config"],
    ]
    rows = [[rf"\textbf{{{r[0]}}}"] + r[1:] for r in rows]
    return tabular(r"@{}>{\raggedright\arraybackslash}p{2.45cm}*{4}{>{\raggedright\arraybackslash}X}@{}", ["", "SFT", "DPO-HH", "DPO-UF", "LoRA-DPO (TRL)"], rows), sft_cfg["environment"]


def judge_cells(label, stem, eval_set):
    rows = judged(PRIMARY, stem)
    s = summarize(rows)
    swapped = judged(PRIMARY, stem + "_swapped")
    d = debiased(rows, swapped) if swapped else None
    sec = judged(RES / "eval" / "judge_gemini-3.7-flash", stem)
    agree, n_agree = agreement(rows, sec) if sec else (None, 0)
    retries = sum(max(0, r.get("attempts", 1) - 1) for r in rows + swapped)
    return [
        label,
        eval_set,
        s["n"],
        f"{s['wins']}/{s['losses']}/{s['ties']}",
        f"{s['unknown']}/{s['failures']}",
        pct(s["win_rate_excluding_ties"]),
        pct(s["win_rate_ties_as_half"]),
        pct(d["consistency"]) if d else "--",
        f"{d['wins']}/{d['losses']}/{d['ties']}" if d else "--",
        pct(d["win_rate_ties_as_half"]) if d else "--",
        pct(agree) if sec else "--",
        retries,
    ], {"rows": rows, "swapped": swapped}


JUDGE_HEADER = ["Comparison", "Eval set", "$n$", "W/L/T", r"\shortstack{Unk./\\fail.}", r"\shortstack{Win\\(excl.\ ties)}",
                r"\shortstack{Win\\(ties \textonehalf)}", r"\shortstack{Swap-\\consistent}", r"\shortstack{Debiased\\W/L/T}",
                r"\shortstack{Debiased\\win}", r"\shortstack{Agree\\w/ 3.7}", "Retries"]
JUDGE_SPEC = r"@{}lXrrrrrrrrrr@{}"


def category_rows(info, key=None):
    swapped = {r["id"]: outcome(r) for r in info["swapped"]}
    cats = {}
    for r in info["rows"]:
        a, b = outcome(r), swapped.get(r["id"])
        verdict = (a if a == b else "tie") if (a and b) else a
        if verdict is None:
            continue
        name = key[r["id"]] if key else r["category"]
        c = cats.setdefault(name, {"win": 0, "loss": 0, "tie": 0})
        c[verdict] += 1
    rows = []
    for cat, c in sorted(cats.items()):
        n = sum(c.values())
        rows.append([tex(cat.replace("_", " ")), n, c["win"], c["loss"], c["tie"], pct((c["win"] + 0.5 * c["tie"]) / n, 0)])
    return rows


def persona_table(stats):
    lora = {s["name"]: s for s in load_json(RES / "lora_dpo" / "sweep_summary.json")}
    rows = []
    for cfg, label in (("r8_attn_lr5e-5", "r8, attention, LR 5e-5"), ("r16_all_lr5e-5", r"\textbf{r16, all-linear, LR 5e-5 (submitted)}"),
                       ("r16_all_lr2e-4", "r16, all-linear, LR 2e-4")):
        row = [label, f"{lora[cfg]['final_eval']['eval_loss']:.3f}", f"{lora[cfg]['final_eval']['eval_rewards/margins']:.2f}"]
        for rub in ("persona_style", "persona_quality"):
            stem = f"{rub}_lora_{cfg}_vs_base"
            d = debiased(judged(PRIMARY, stem), judged(PRIMARY, stem + "_swapped"))
            row += [f"{d['wins']}/{d['losses']}/{d['ties']}", pct(d["win_rate_ties_as_half"])]
        row.append(f"{stats[f'persona_lora_{cfg}']['mean_words']:.0f}")
        rows.append(row)
    return tabular(r"@{}Xrrrrrrr@{}", ["LoRA config", r"\shortstack{Held-out\\loss}", r"\shortstack{Held-out\\margin}", r"\shortstack{Style\\W/L/T}",
                                       r"\shortstack{Style\\win}", r"\shortstack{Usefulness\\W/L/T}", r"\shortstack{Usefulness\\win}", r"\shortstack{Mean\\words}"], rows)


def stats_table(stats):
    rows = []
    for name, label in (("hh_base", "base, HH"), ("hh_sft", "SFT, HH"), ("hh_dpo_hh", "DPO-HH, HH"), ("hh_dpo_uf", "DPO-UF, HH"),
                        ("alpaca_base", "base, Alpaca"), ("alpaca_sft", "SFT, Alpaca"), ("alpaca_dpo_hh", "DPO-HH, Alpaca"),
                        ("alpaca_dpo_uf", "DPO-UF, Alpaca")):
        s = stats[name]
        rows.append([label, f"{s['mean_words']:.0f}", pct(s["unfinished_rate"]), pct(s["refusal_marker_rate"])])
    return tabular(r"@{}Xrrr@{}", ["Model, eval set", r"\shortstack{Mean\\words}", r"\shortstack{Hit 512\\limit}", r"\shortstack{Refusal\\phrases}"], rows)


def example_box(gen_a, gen_b, item_id, label_a, label_b, chars=380):
    a = {r["id"]: r for r in load_json(RES / "generations" / f"{gen_a}.json")}[item_id]
    b = {r["id"]: r for r in load_json(RES / "generations" / f"{gen_b}.json")}[item_id]

    def clip(text):
        text = " ".join(text.split())
        return tex(text[:chars] + (" ..." if len(text) > chars else ""))

    tag = a.get("category") or a.get("source_dataset") or ""
    return (
        r"\begin{exbox}" + "\n"
        + rf"\textbf{{Prompt}} ({tex(tag.replace('_', ' '))}): {tex(a['instruction'][:300])}\par\smallskip" + "\n"
        + rf"\textbf{{{label_a}:}} {clip(a['output'])}\par\smallskip" + "\n"
        + rf"\textbf{{{label_b}:}} {clip(b['output'])}" + "\n"
        + r"\end{exbox}"
    )


EXAMPLES = [
    ("hh_dpo_hh", "hh_sft", "hh-eval-42-0004", "DPO-HH (win in both orders)", "SFT"),
    ("hh_dpo_hh", "hh_sft", "hh-eval-42-0000", "DPO-HH (tie: judge says both fail)", "SFT"),
    ("hh_dpo_hh", "hh_sft", "hh-eval-42-0006", "DPO-HH (loss)", "SFT"),
    ("alpaca_dpo_uf", "alpaca_sft", "alpaca-eval-281", "DPO-UF (win in both orders)", "SFT"),
    ("alpaca_dpo_uf", "alpaca_sft", "alpaca-eval-228", "DPO-UF (loss in both orders)", "SFT"),
    ("persona_lora_r16_all_lr5e-5", "persona_base", "persona-eval-30", "LoRA r16, LR 5e-5 (submitted)", "SmolLM2 base"),
    ("persona_lora_r16_all_lr2e-4", "persona_base", "persona-eval-36", "LoRA r16, LR 2e-4 (over-optimized)", "SmolLM2 base"),
]


def main():
    manifest = load_json(RES / "data" / "data_manifest.json")
    stats = load_json(RES / "eval" / "response_stats.json")
    write("data_table", data_table(manifest))
    cfg, env = config_table()
    write("config_table", cfg)
    write("software", "\\newcommand{\\softwareinfo}{" + f"PyTorch {tex(env['torch'])}, Transformers {env['transformers']}, Datasets {env['datasets']}, 2$\\times$ Tesla T4 (Kaggle)" + "}")

    main_rows, infos = [], {}
    for label, stem, eval_set in (("DPO-HH vs SFT", "dpo_hh_vs_sft_hh", "HH (300)"), ("DPO-UF vs SFT", "dpo_uf_vs_sft_alpaca", "Alpaca (300)")):
        cells, info = judge_cells(label, stem, eval_set)
        main_rows.append(cells)
        infos[stem] = info
    write("judge_main", tabular(JUDGE_SPEC, JUDGE_HEADER, main_rows, r"\scriptsize"))
    cross_rows = [judge_cells(label, stem, ev)[0] for label, stem, ev in (
        ("DPO-HH vs SFT", "dpo_hh_vs_sft_alpaca", "Alpaca (300)"), ("DPO-UF vs SFT", "dpo_uf_vs_sft_hh", "HH (300)"))]
    write("judge_cross", tabular(JUDGE_SPEC, JUDGE_HEADER, cross_rows, r"\scriptsize"))
    write("persona_table", persona_table(stats))

    write("hh_categories", tabular(r"@{}Xrrrrr@{}", ["HH category", "$n$", "W", "L", "T", "Win"], category_rows(infos["dpo_hh_vs_sft_hh"]), r"\scriptsize"))
    src = {r["id"]: r.get("source_dataset") for r in load_json(RES / "generations" / "alpaca_sft.json")}
    write("uf_sources", tabular(r"@{}Xrrrrr@{}", ["AlpacaEval source", "$n$", "W", "L", "T", "Win"], category_rows(infos["dpo_uf_vs_sft_alpaca"], src), r"\scriptsize"))
    write("response_stats", stats_table(stats).replace(r"\footnotesize", r"\scriptsize", 1))
    write("examples", "\n".join(example_box(*e) for e in EXAMPLES))
    print(f"wrote tables to {OUT}")


if __name__ == "__main__":
    main()
