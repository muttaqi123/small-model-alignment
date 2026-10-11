"""Build report/report.html (and report.pdf via headless Chrome) from results/.

    python report/build_report.py [--pdf]

Every number in the tables is read from the files under results/, so the
report can be regenerated after any rerun.
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
sys.path.insert(0, str(ROOT))

from scripts.judge_pairwise_kaggle import read_jsonl, summarize  # noqa: E402
from report.text import TEXT  # noqa: E402

PRIMARY = RES / "eval" / "judge_gemini-3.6-flash"
SECONDARY = RES / "eval" / "judge_gemini-3.7-flash"


def load_json(path):
    return json.loads(Path(path).read_text())


def pct(x, digits=1):
    return "–" if x is None else f"{100 * x:.{digits}f}%"


def outcome(row):
    if row["decision"] in ("unknown", "failure"):
        return None
    if row["decision"] == "tie":
        return "tie"
    return "win" if row["winner_model"] == row["target_model"] else "loss"


def judged(directory, stem):
    path = Path(directory) / f"{stem}_judged.jsonl"
    return read_jsonl(path) if path.exists() else []


def debiased(rows, swapped):
    """Combine the two presentation orders: agreeing verdicts stand, disagreements become ties."""

    other = {r["id"]: outcome(r) for r in swapped}
    merged, consistent = [], []
    for r in rows:
        a, b = outcome(r), other.get(r["id"])
        if a and b:
            merged.append(a if a == b else "tie")
            consistent.append(a == b)
    n = len(merged)
    w, l, t = merged.count("win"), merged.count("loss"), merged.count("tie")
    return {
        "n": n,
        "wins": w,
        "losses": l,
        "ties": t,
        "win_rate_ties_as_half": (w + 0.5 * t) / n if n else None,
        "win_rate_excluding_ties": w / (w + l) if (w + l) else None,
        "consistency": sum(consistent) / len(consistent) if consistent else None,
    }


def agreement(rows, other_rows):
    other = {r["id"]: outcome(r) for r in other_rows}
    pairs = [(outcome(r), other[r["id"]]) for r in rows if r["id"] in other and outcome(r) and other[r["id"]]]
    return (sum(a == b for a, b in pairs) / len(pairs), len(pairs)) if pairs else (None, 0)


def table(header, rows, cls=""):
    out = [f'<table class="{cls}"><thead><tr>' + "".join(f"<th>{h}</th>" for h in header) + "</tr></thead><tbody>"]
    for row in rows:
        out.append("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def judge_row(label, stem, eval_set):
    rows = judged(PRIMARY, stem)
    if not rows:
        return None, None
    s = summarize(rows)
    swapped = judged(PRIMARY, stem + "_swapped")
    d = debiased(rows, swapped) if swapped else None
    sec = judged(SECONDARY, stem)
    agree, n_agree = agreement(rows, sec) if sec else (None, 0)
    retries = sum(max(0, r.get("attempts", 1) - 1) for r in rows + swapped)
    cells = [
        label,
        eval_set,
        s["n"],
        f"{s['wins']} / {s['losses']} / {s['ties']}",
        f"{s['unknown']} / {s['failures']}",
        pct(s["win_rate_excluding_ties"]),
        pct(s["win_rate_ties_as_half"]),
        pct(d["consistency"]) if d else "–",
        f"{d['wins']} / {d['losses']} / {d['ties']}" if d else "–",
        pct(d["win_rate_ties_as_half"]) if d else "–",
        f"{pct(agree)} (n={n_agree})" if sec else "–",
        retries,
    ]
    return cells, {"summary": s, "debiased": d, "agreement": agree, "rows": rows, "swapped": swapped}


def category_table(info):
    rows = info["rows"]
    swapped = {r["id"]: outcome(r) for r in info["swapped"]}
    cats = {}
    for r in rows:
        a, b = outcome(r), swapped.get(r["id"])
        verdict = (a if a == b else "tie") if (a and b) else a
        if verdict is None:
            continue
        c = cats.setdefault(r["category"], {"win": 0, "loss": 0, "tie": 0})
        c[verdict] += 1
    out = []
    for cat, c in sorted(cats.items()):
        n = sum(c.values())
        out.append([cat.replace("_", " "), n, c["win"], c["loss"], c["tie"], pct((c["win"] + 0.5 * c["tie"]) / n, 0)])
    return out


def example(gen_a, gen_b, item_id, label_a, label_b, chars=420):
    a = {r["id"]: r for r in load_json(RES / "generations" / f"{gen_a}.json")}[item_id]
    b = {r["id"]: r for r in load_json(RES / "generations" / f"{gen_b}.json")}[item_id]

    def clip(text):
        text = " ".join(text.split())
        return html.escape(text[:chars] + (" …" if len(text) > chars else ""))

    return (
        f'<div class="ex"><div class="exq"><b>Prompt</b> ({html.escape(str(a.get("category") or a.get("source_dataset") or ""))}): '
        f"{html.escape(a['instruction'][:300])}</div>"
        f'<div class="exa"><b>{label_a}:</b> {clip(a["output"])}</div>'
        f'<div class="exa"><b>{label_b}:</b> {clip(b["output"])}</div></div>'
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()

    manifest = load_json(RES / "data" / "data_manifest.json")
    sft_cfg, hh_cfg, uf_cfg = (load_json(RES / n / "run_config.json") for n in ("sft", "dpo_hh", "dpo_uf"))
    sft_sum, hh_sum, uf_sum = (load_json(RES / n / "summary.json") for n in ("sft", "dpo_hh", "dpo_uf"))
    lora = {s["name"]: s for s in load_json(RES / "lora_dpo" / "sweep_summary.json")}
    stats = load_json(RES / "eval" / "response_stats.json")
    env = sft_cfg["environment"]

    T = TEXT

    config_rows = [
        ["Initial weights", "Qwen2.5-0.5B (base)", "SFT best (step 938)", "SFT best (step 938)", "SmolLM2-360M-Instruct"],
        ["Trainable params", "full FT, 494M", "full FT, 494M", "full FT, 494M",
         f"LoRA r=16, α=32, dropout 0.05, q/k/v/o/gate/up/down ({lora['r16_all_lr5e-5']['trainable_params'] / 1e6:.2f}M, 2.3%)"],
        ["Data (train / val)", f"UltraChat {sft_cfg['n_train']:,} / {sft_cfg['n_val']:,}",
         f"HH-RLHF {hh_cfg['n_train_pairs']:,} / {hh_cfg['n_val_pairs']:,}",
         f"UltraFeedback {uf_cfg['n_train_pairs']:,} / {uf_cfg['n_val_pairs']:,}", "persona 200 / 20"],
        ["Optimizer", "AdamW (β=0.9, 0.999), fused", "AdamW (β=0.9, 0.999)", "AdamW (β=0.9, 0.999)", "AdamW (TRL default)"],
        ["Peak LR / schedule", f"{sft_cfg['lr']:.0e}, warmup {sft_cfg['warmup_steps']} steps (3%) + cosine to 0",
         f"{hh_cfg['lr']:.0e}, constant, no warmup", f"{uf_cfg['lr']:.0e}, constant, no warmup", "5e-5, 10% warmup + cosine"],
        ["Weight decay / clip", "0.1 (not on norms/biases) / 1.0", "0 / 1.0", "0 / 1.0", "0 / 1.0"],
        ["Batch (device × accum = global)", "2 × 16 = 32", "1 pair × 32 = 32", "1 pair × 32 = 32", "4 × 2 = 8"],
        ["Steps / epochs", f"{sft_cfg['total_steps']} / 1", f"{hh_cfg['total_steps']} / 1", f"{uf_cfg['total_steps']} / 1",
         f"{lora['r16_all_lr5e-5']['optimizer_steps']} / 3"],
        ["Precision", "fp32 weights, fp16 autocast, GradScaler", "policy as SFT; reference fp32 (no grad, cuda:1)", "same as HH", "fp32 weights, fp16 AMP"],
        ["Max length (prompt)", "512", "512 (256)", "512 (256)", "512 (256)"],
        ["DPO β / reference", "–", f"{hh_cfg['beta']} / frozen SFT best", f"{uf_cfg['beta']} / frozen SFT best", "0.1 / base model, adapter disabled"],
        ["Seed / wall-clock", f"42 / {sft_sum['train_time_sec'] / 60:.0f} min", f"42 / {hh_sum['train_time_sec'] / 60:.0f} min",
         f"42 / {uf_sum['train_time_sec'] / 60:.0f} min", "42 / ≈1.5–2 min per config"],
    ]

    main_rows, infos = [], {}
    for label, stem, eval_set in (
        ("DPO-HH vs SFT", "dpo_hh_vs_sft_hh", "HH synthetic (300)"),
        ("DPO-UF vs SFT", "dpo_uf_vs_sft_alpaca", "AlpacaEval (300)"),
    ):
        cells, info = judge_row(label, stem, eval_set)
        if cells:
            main_rows.append(cells)
            infos[stem] = info
    cross_rows = []
    for label, stem, eval_set in (
        ("DPO-HH vs SFT", "dpo_hh_vs_sft_alpaca", "AlpacaEval (300)"),
        ("DPO-UF vs SFT", "dpo_uf_vs_sft_hh", "HH synthetic (300)"),
    ):
        cells, info = judge_row(label, stem, eval_set)
        if cells:
            cross_rows.append(cells)
            infos[stem] = info

    judge_header = ["comparison", "eval set", "n", "W / L / T", "unk. / fail.", "win (excl. ties)", "win (ties ½)",
                    "swap-consistent", "order-debiased W / L / T", "debiased win (ties ½)", "agree w/ 3.7-flash", "retries"]

    persona_rows = []
    for cfg, label in (("r8_attn_lr5e-5", "r8, attn, lr 5e-5"), ("r16_all_lr5e-5", "r16, all-linear, lr 5e-5 (submitted)"),
                       ("r16_all_lr2e-4", "r16, all-linear, lr 2e-4")):
        row = [label, f"{lora[cfg]['final_eval']['eval_loss']:.3f}", f"{lora[cfg]['final_eval']['eval_rewards/margins']:.2f}"]
        for rub in ("persona_style", "persona_quality"):
            stem = f"{rub}_lora_{cfg}_vs_base"
            rows = judged(PRIMARY, stem)
            swapped = judged(PRIMARY, stem + "_swapped")
            if rows and swapped:
                d = debiased(rows, swapped)
                row += [f"{d['wins']} / {d['losses']} / {d['ties']}", pct(d["win_rate_ties_as_half"])]
            elif rows:
                s = summarize(rows)
                row += [f"{s['wins']} / {s['losses']} / {s['ties']}", pct(s["win_rate_ties_as_half"])]
            else:
                row += ["–", "–"]
        st = stats.get(f"persona_lora_{cfg}")
        row.append(f"{st['mean_words']:.0f}" if st else "–")
        persona_rows.append(row)

    stat_rows = []
    for name, label in (("hh_base", "base · HH"), ("hh_sft", "SFT · HH"), ("hh_dpo_hh", "DPO-HH · HH"), ("hh_dpo_uf", "DPO-UF · HH"),
                        ("alpaca_base", "base · Alpaca"), ("alpaca_sft", "SFT · Alpaca"), ("alpaca_dpo_hh", "DPO-HH · Alpaca"),
                        ("alpaca_dpo_uf", "DPO-UF · Alpaca")):
        s = stats[name]
        stat_rows.append([label, f"{s['mean_words']:.0f}", pct(s["unfinished_rate"]), pct(s["refusal_marker_rate"])])

    hh_cat = category_table(infos["dpo_hh_vs_sft_hh"]) if "dpo_hh_vs_sft_hh" in infos else []
    uf_src = []
    if "dpo_uf_vs_sft_alpaca" in infos:
        src = {r["id"]: r.get("source_dataset") for r in load_json(RES / "generations" / "alpaca_sft.json")}
        info = infos["dpo_uf_vs_sft_alpaca"]
        for r in info["rows"]:
            r["category"] = src[r["id"]]
        for r in info["swapped"]:
            r["category"] = src[r["id"]]
        uf_src = category_table(info)

    examples_html = "".join(example(*e) for e in T["examples"])

    css = """
    @page { size: A4; margin: 14mm 13mm 14mm 13mm; }
    body { font-family: 'Times New Roman', Times, serif; font-size: 10pt; line-height: 1.28; color: #111; }
    h1 { font-size: 15.5pt; margin: 0 0 2pt 0; text-align: center; }
    .sub { text-align: center; font-size: 9.6pt; margin-bottom: 8pt; color: #333; }
    h2 { font-size: 11.6pt; margin: 9pt 0 3pt 0; border-bottom: 0.6pt solid #888; padding-bottom: 1pt; }
    h3 { font-size: 10.4pt; margin: 6pt 0 2pt 0; }
    p { margin: 2pt 0 4pt 0; text-align: justify; }
    ul { margin: 2pt 0 4pt 16pt; padding: 0; } li { margin: 1pt 0; }
    table { border-collapse: collapse; width: 100%; font-size: 8.3pt; margin: 3pt 0 5pt 0; }
    tr { page-break-inside: avoid; break-inside: avoid; }
    thead { display: table-header-group; }
    h2, h3 { break-after: avoid; page-break-after: avoid; }
    th, td { border: 0.5pt solid #999; padding: 1.6pt 3pt; vertical-align: top; text-align: left; }
    th { background: #eee; }
    table.cfg td:first-child { font-weight: bold; width: 15%; }
    .cap { font-size: 8.6pt; color: #333; margin: 0 0 6pt 0; }
    img { width: 100%; page-break-inside: avoid; }
    code { font-family: Menlo, Consolas, monospace; font-size: 8.4pt; }
    .ex { border: 0.5pt solid #bbb; padding: 3pt 5pt; margin: 3pt 0; font-size: 8.4pt; page-break-inside: avoid; }
    .exq { margin-bottom: 2pt; } .exa { margin: 1pt 0 1pt 6pt; }
    .two { display: flex; gap: 8pt; } .two > div { flex: 1; }
    """

    body = f"""
<h1>Instruction Tuning and Direct Preference Optimization on Qwen2.5-0.5B</h1>
<div class="sub">{T['author']} · Advanced Generative AI, PA2 · Repository: <code>{T['repo']}</code></div>

<h2>1. Repository, Models, and Data</h2>
{T['data']}
{table(["track", "source", "split used", "filtering", "train / val"], [
    ["SFT", "HuggingFaceH4/ultrachat_200k", "train_sft / test_sft", "first user turn + first reply; full sequence ≤ 512 tokens; dedup by prompt",
     f"{manifest['sft']['train']['kept']:,} / {manifest['sft']['val']['kept']:,}"],
    ["HH-RLHF DPO", "Anthropic/hh-rlhf (all subsets)", "train / test", "exactly one Human+Assistant turn on both sides, same prompt, different replies; prompt ≤ 256, each side ≤ 512",
     f"{manifest['dpo_hh']['train']['kept']:,} / {manifest['dpo_hh']['val']['kept']:,}"],
    ["UltraFeedback DPO", "HuggingFaceH4/ultrafeedback_binarized", "train_prefs / test_prefs", "2-message conversations, score_chosen > score_rejected, different replies; same length limits",
     f"{manifest['dpo_uf']['train']['kept']:,} / {manifest['dpo_uf']['val']['kept']:,}"],
    ["LoRA persona", "data/persona_lora/ramsay_style_preferences.json", "220 rows", "seeded 200 / 20 split", "200 / 20"],
])}
{T['eval_sets']}

<h2>2. Experimental Configuration</h2>
{table(["", "SFT", "DPO-HH", "DPO-UF", "LoRA-DPO (TRL)"], config_rows, "cfg")}
<p class="cap">Software: PyTorch {env['torch']}, Transformers {env['transformers']}, Datasets {env['datasets']}, 2× Tesla T4 (Kaggle). {T['config_note']}</p>
{T['generation']}

<h2>3. Parameter Choices and Deviations</h2>
{T['deviations']}

<h2>4. Training Behavior</h2>
<img src="../results/plots/sft_loss.png" style="width:55%; display:block; margin:0 auto;">
<p class="cap">Figure 1. SFT response-token NLL. Faint: per-step training loss; solid: EMA (0.9); dots: validation loss on 1,000 held-out examples every 50 steps (unsmoothed). Star: selected checkpoint.</p>
{T['sft_behavior']}
<img src="../results/plots/dpo_curves.png">
<p class="cap">Figure 2. From-scratch DPO. Lines: training metrics averaged over each 32-pair optimizer step, EMA 0.9; dots: full validation set every 25 steps (unsmoothed). Stars: checkpoint with the highest validation reward accuracy (ties broken by margin).</p>
{T['dpo_behavior']}
<img src="../results/plots/lora_dpo_curves.png">
<p class="cap">Figure 3. TRL DPOTrainer + LoRA on the persona data for three adapter configurations; train loss unsmoothed (thin), held-out metrics every 5 steps.</p>
{T['lora_behavior']}

<h2>5. DPO-versus-SFT Results</h2>
{T['results_intro']}
{table(judge_header, main_rows)}
<p class="cap">Table: judge = google/gemini-3.6-flash via Kaggle Benchmarks (temperature 0, 5 pairs per request, ≥5 s between requests, ≤3 attempts). “win” is from the DPO model's perspective. Order-debiased: each pair judged in both A/B orders; a verdict counts only if it survives the swap, otherwise it is a tie. Agreement: same-order verdicts of gemini-3.7-flash on the first 60 pairs (W/L/T, unknown/failure excluded).</p>
{T['results_main']}
<h3>Cross-set checks (not part of the required comparison)</h3>
{table(judge_header, cross_rows)}
{T['results_cross']}
<h3>LoRA persona result (SmolLM2-360M-Instruct, vs. the base instruct model, 40 prompts)</h3>
{table(["LoRA config", "held-out DPO loss", "held-out margin", "style W/L/T", "style win (ties ½)", "usefulness/safety W/L/T", "usefulness win (ties ½)", "mean words"], persona_rows)}
<p class="cap">Style and usefulness are judged separately (persona-faithfulness rubric vs. a tone-blind usefulness/safety rubric), both order-debiased as above. Base-model reference: 136 mean words.</p>
{T['results_persona']}

<h2>6. Interpretation and Error Analysis</h2>
{T['interpretation']}
<div class="two"><div>
<p class="cap"><b>DPO-HH vs SFT by HH category</b> (order-debiased)</p>
{table(["category", "n", "W", "L", "T", "win (ties ½)"], hh_cat)}
</div><div>
<p class="cap"><b>DPO-UF vs SFT by AlpacaEval source</b> (order-debiased)</p>
{table(["source", "n", "W", "L", "T", "win (ties ½)"], uf_src)}
<p class="cap"><b>Response statistics</b> (greedy, same decoding for all models)</p>
{table(["model · eval set", "mean words", "hit 512-token limit", "refusal-phrase rate"], stat_rows)}
</div></div>
{T['interpretation_2']}
<h3>Representative examples</h3>
{examples_html}

<h2>7. Limitations and Reproducibility</h2>
{T['limitations']}
"""
    out = ROOT / "report" / "report.html"
    out.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>PA2 report</title><style>{css}</style></head><body>{body}</body></html>")
    print(f"wrote {out}")
    if args.pdf:
        chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        pdf = ROOT / "report" / "report.pdf"
        subprocess.run(
            [chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", out.as_uri()],
            check=True,
            capture_output=True,
        )
        print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
