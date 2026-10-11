"""Prose for the report; tables and numbers are generated in build_report.py."""

TEXT = {
    "author": "Muhammad Muttaqi (28100428)",
    "repo": "github.com/muttaqi123/small-model-alignment",
    "data": """
<p><b>Repository.</b> All code is at <code>github.com/muttaqi123/small-model-alignment</code>. The main pieces are
<code>alignment/batching.py</code> (tokenization and masking), <code>alignment/logprobs.py</code> (response log-probabilities),
<code>alignment/losses.py</code> (SFT and DPO losses), and the two training loops <code>alignment/sft.py</code> and <code>alignment/dpo.py</code>,
which I wrote without the HF <code>Trainer</code> or TRL. Generation is in <code>alignment/generate.py</code>, and
<code>submission_adapter.py</code> just wraps these functions. I only used TRL and PEFT for the persona task
(<code>scripts/train_lora_dpo.py</code>). The Kaggle notebooks come from <code>kaggle/build_notebooks.py</code>, and the judging runs as a
Kaggle Benchmarks task (<code>kaggle/judge_task.py</code>).</p>
<p><b>Models.</b> I used <code>Qwen/Qwen2.5-0.5B</code> (base) for SFT and both from-scratch DPO runs, and
<code>HuggingFaceTB/SmolLM2-360M-Instruct</code> for the LoRA persona task. <b>Seeds.</b> I used seed 42 for everything: dataset shuffles,
data-loader order, LoRA initialization and the A/B order of the pairs. <b>Prompt format.</b> For SFT, DPO and all Qwen generations I used
ChatML with no system message, <code>&lt;|im_start|&gt;user\\n{instruction}&lt;|im_end|&gt;\\n&lt;|im_start|&gt;assistant\\n</code>, and the
response ends with <code>&lt;|im_end|&gt;</code>. I set <code>&lt;|im_end|&gt;</code> as the tokenizer's EOS and kept
<code>&lt;|endoftext|&gt;</code> as PAD. For SmolLM2 I kept its own chat template, since that is what TRL applies to conversational
preference data. I reduced every dataset to single-turn examples, shuffled with seed 42, and filtered by length with the training tokenizer
so that nothing gets truncated during training. For UltraChat this means replies longer than 512 tokens were dropped instead of cut, so every
SFT target still ends in EOS.</p>
""",
    "eval_sets": """
<p><b>Fixed evaluation sets</b> (all in <code>eval_sets/</code>): the provided synthetic helpful/harmless set
(<code>hh_rlhf_eval_300_seed42.json</code>, 300 prompts, 10 categories of 30); a 300-prompt AlpacaEval subset, which I sampled with
<code>random.Random(42).sample</code> from the 805 prompts in <code>tatsu-lab/alpaca_eval</code> (<code>alpaca_eval_300_seed42.json</code>);
and 40 persona prompts that I wrote myself (<code>persona_style_eval_40.json</code>; 10 each of cooking, coding, general help and
safety-sensitive). I checked that none of them were close copies of the 220 persona training prompts and replaced the five that were.</p>
""",
    "config_note": """Every run uses response-only masking: prompt and padding labels are −100, and log-probabilities are summed over
response tokens only (the logits at position t score the label at t+1, computed in fp32). For the SFT loss I take the mean NLL over all
response tokens in an accumulation window, so each optimizer step averages over all response tokens of its 32 examples. For DPO I use summed
log-probabilities per pair and the loss −log σ(β[(log π<sub>θ</sub>(y<sub>w</sub>|x) − log π<sub>θ</sub>(y<sub>l</sub>|x)) −
(log π<sub>ref</sub>(y<sub>w</sub>|x) − log π<sub>ref</sub>(y<sub>l</sub>|x))]), averaged over the batch; chosen and rejected go through
each model in one padded forward pass. TRL and PEFT were installed from the ranges <code>trl&gt;=0.19,&lt;0.24</code> and
<code>peft&gt;=0.15,&lt;0.18</code>.""",
    "generation": """
<p><b>Generation settings.</b> Within each comparison I used the same settings for every model: greedy decoding (no temperature or top-p),
repetition penalty 1.1, no n-gram blocking, no length penalty, fp16 weights, batch size 16 with left padding, and stopping on
<code>&lt;|im_end|&gt;</code> or <code>&lt;|endoftext|&gt;</code>. Max new tokens was 512 for the Qwen models (AlpacaEval and HH sets)
and 256 for the SmolLM2 persona comparison. Before each full generation run I generated 3 prompts and read the outputs to check the
checkpoint path, tokenizer, EOS behaviour and length.</p>
""",
    "deviations": """
<table><thead><tr><th>setting</th><th>guide</th><th>what I used</th><th>type</th><th>reason and effect</th></tr></thead><tbody>
<tr><td>Long examples</td><td>context 512 (truncate)</td><td>filter: keep SFT examples ≤ 512 tokens; DPO pairs with prompt ≤ 256 and both sides ≤ 512</td><td>performance</td>
<td>A truncated SFT target loses its EOS, which teaches the model not to stop. Filtering keeps every target complete. The cost is a bias toward
shorter replies (I went through 51,794 UltraChat conversations to keep 30,000). The SFT model hits the 512-token limit on only 1–5% of prompts,
compared with 34–52% for the base model.</td></tr>
<tr><td>UltraFeedback validation</td><td>1,000 pairs</td><td>969</td><td>data limit</td><td>Only 969 single-turn, non-tied <code>test_prefs</code> pairs pass the length filter. The manual allows fewer “when available”.</td></tr>
<tr><td>DPO reference precision</td><td>fp16 allowed</td><td>fp32 on cuda:1, same autocast as the policy</td><td>diagnostic</td>
<td>There was enough memory (about 2 GB). Because the numerics match exactly, the rewards at step 0 are exactly 0 (loss = log 2, accuracy 0,
margin 0, Fig. 2), so any later reward comes from learning and not from fp16 rounding.</td></tr>
<tr><td>DPO optimizer</td><td>RMSprop or AdamW</td><td>AdamW, weight decay 0, constant LR 1e-6</td><td>choice</td><td>Within the guide, with no schedule as in the reference recipe.</td></tr>
<tr><td>SFT weight decay</td><td>0.1</td><td>0.1 on weight matrices, 0 on norms and biases</td><td>choice</td><td>This is standard; pulling RMSNorm gains toward 0 does not help as regularization.</td></tr>
<tr><td>Validation frequency</td><td>–</td><td>SFT every 50 steps, DPO every 25 steps (plus step 0)</td><td>diagnostic</td><td>About 20 points per curve. It cost about 15 min (SFT) and 30–40 min (each DPO run) of GPU time.</td></tr>
<tr><td>LoRA persona (TRL)</td><td>student choice</td><td>r 16, α 32, dropout 0.05, all 7 linear projections, LR 5e-5 cosine (10% warmup), batch 4×2, 3 epochs, β 0.1</td><td>performance</td>
<td>I picked this after comparing 3 configurations (Fig. 3, §5). Rank 8 on attention only hardly changed the style, and LR 2e-4 pushed the
held-out loss to almost 0 and produced abusive, incoherent text. Rank 16 on all projections (2.3% of the parameters) lets the adapter change
both the attention and the MLP layers. 5e-5 is a normal LoRA learning rate (roughly 10–50× a full fine-tuning LR). Three epochs over 200 pairs
(75 steps) gave a stable held-out loss without collapsing to zero, and batch 8 fits easily on one T4.</td></tr>
<tr><td>Decoding</td><td>greedy</td><td>greedy + repetition penalty 1.1</td><td>quality</td>
<td>In my pilot run the base model got stuck in loops without it. I used it for every model so that the comparisons only reflect the checkpoint.</td></tr>
</tbody></table>
<p>I did not need any change for memory: all the guide batch sizes fit on a T4 (SFT 2×16; DPO 1 pair × 32, with the policy on cuda:0 and the
reference on cuda:1). So none of my deviations were resource-driven. The fp16 GradScaler skipped the first few DPO steps because the initial loss
scale of 65,536 overflowed (3 of 469 steps in each run). It skipped no SFT steps.</p>
""",
    "sft_behavior": """
<p><b>SFT.</b> The validation NLL dropped from 1.474 (the base model at step 0) to 1.378 in the first 50 steps. Most of this is the model
learning the ChatML format and when to stop. After that it kept going down slowly and reached 1.317 (perplexity 3.73) at the end of the epoch.
The training and validation losses stay close to each other, so there is no overfitting, which makes sense for a single pass over 30k examples.
The lowest validation loss is at the last step, so my <code>best</code> and <code>final</code> checkpoints are the same (step 938). The loss is
fairly flat between steps 50 and 300, while the learning rate is still near its peak, and starts falling again as the cosine schedule decays.
Gradient norms stayed below 7 and fp16 never skipped a step.</p>
""",
    "dpo_behavior": """
<p><b>DPO.</b> Both runs start at exactly log 2 with zero rewards, since the policy and the reference are identical, and both learn a clear
preference. On UltraFeedback the validation reward accuracy jumps to 0.65 within 25 steps and then climbs to 0.70, while the margin grows to
0.66 and the loss drops to 0.570. HH-RLHF is clearly harder: accuracy rises more slowly to 0.65, and the margin only reaches 0.25 (loss 0.631).
I think this comes from the data. Many HH pairs are almost the same or both mediocre, while UltraFeedback pairs differ clearly in quality and
length. In both runs the rewards of the chosen <i>and</i> the rejected responses become negative (UF at the end: r<sub>w</sub> = −0.29,
r<sub>l</sub> = −0.94). So DPO mostly lowers the probability of rejected responses faster than chosen ones, rather than raising the chosen
ones; this effect is known as “likelihood displacement”. With a small fixed LR of 1e-6 for one epoch, neither curve has flattened out, and I
see no sign of over-optimization since the train and validation metrics match. Selecting by validation reward accuracy picked the last step in
both runs (for UF, 0.704 at steps 450 and 469, with the tie broken by the larger margin), so best = final. These curves do not tell me how good
the responses are, so §5 evaluates the generations directly.</p>
""",
    "lora_behavior": """
<p><b>LoRA-DPO.</b> All three adapters reach 100% held-out reward accuracy within 10 steps. The persona pairs (theatrical, insulting chosen
vs. neutral rejected) are very easy to separate, so accuracy does not say much here, and I compare them by margin and loss instead. The rank-8
attention-only adapter (1.6M parameters) moves slowly (held-out loss 0.456, margin 0.56). Rank 16 on all projections at 5e-5 reaches loss
0.109 and margin 2.28, and flattens out as the cosine schedule ends. At 2e-4 the held-out loss is below 0.01 by step 25 (0.0008 at the end) and
the margin keeps growing to 9.0, which is a clear sign of over-optimization: the reward of the rejected (neutral, correct) answers falls to
−10.1, meaning their log-probability dropped by about 101 nats compared with the reference.</p>
""",
    "results_intro": """
<p>For each DPO track I compared the submitted checkpoint (<code>best</code> = step 469) with the SFT checkpoint on the same prompts and with the
same decoding, using blinded pair files (<code>results/eval/pairs/</code>). The order of the two responses is randomized per prompt (seed 42),
and the judge only sees “Response A” and “Response B”. I judged every pair twice, once in each order, to measure and remove position bias.
For the safety set the judge prompt includes the expected behaviour for the category (for example “refuse and offer a safe alternative”) and
asks the judge to reward appropriate refusals, privacy protection, honest uncertainty and safe redirection, and to penalize refusing harmless
requests. For AlpacaEval the judge scores helpfulness, correctness, instruction-following, concision, formatting and hallucinations. If a
judgment is invalid, only the unresolved pairs of that batch are asked again, up to 3 times. A pair whose response never parses to A, B or tie
is marked “unknown”, and a pair whose requests all fail is a “failure”. Both are left out of the win rates and logged in
<code>*_failures.jsonl</code>.</p>
""",
    "results_main": """
<p><b>HH-RLHF DPO (safety set).</b> DPO-HH beats SFT: 130 wins, 90 losses and 80 ties in the single randomized order (56.7% win rate
counting ties as half; sign test on the decided pairs p = 0.008). When I keep only the verdicts that hold in both orders, I get 93 / 67 / 140
(54.3%, p = 0.048). That is a small improvement, but a statistically real one. <b>UltraFeedback DPO (AlpacaEval).</b> DPO-UF does not beat SFT:
119 / 124 / 57 (49.2%, p = 0.80), and 78 / 90 / 132 (48.0%, p = 0.40) after removing order effects, so there is no difference beyond noise.
All 1,200 judgments parsed on the first try, with 0 unknown, 0 failures and 0 retries. The judge picked “Response A” in 56–61% of decided pairs
and kept its verdict after swapping the order only 68–73% of the time, which is why I also report the order-corrected numbers (pairs where the
two orders disagree count as ties). On a 60-pair sample, gemini-3.7-flash agreed with gemini-3.6-flash on 82–83% of verdicts and pointed the
same way (59.2% and 50.0%).</p>
""",
    "results_cross": """
<p>I also compared each DPO model with SFT on the <i>other</i> evaluation set (single order). DPO-HH ties with SFT on AlpacaEval
(108 / 107 / 85, 50.2%, p = 1.0), so the safety training cost nothing measurable in general helpfulness. DPO-UF ties with SFT on the safety set
(50.3%), so the helpfulness training neither helped nor hurt safety.</p>
""",
    "results_persona": """
<p>The judge found my submitted adapter (rank 16, all projections, LR 5e-5) more in character than the base model (16 / 12 / 12, 55.0%), but
it lost some usefulness (12 / 18 / 10, 42.5%). The rank-8 attention adapter barely changed the outputs (32 of the 40 style comparisons end as
ties after the swap) and stayed level on usefulness (50.0%). The LR 2e-4 adapter has the strongest style (60.0%) but is much less useful and
safe (5 / 29 / 6, 20.0%), which fits its collapsed held-out loss. That is why I submit the middle configuration: it is the only one that
clearly changed the style, at a moderate cost in usefulness. The style judgments are the least reliable numbers in this report (swap consistency
55–75%, agreement with 3.7-flash 50–80%), while the usefulness judgments are much more stable (75–85% and 80–90%).</p>
""",
    "interpretation": """
<p><b>Which model is better, by how much, and why.</b> <i>What I measured:</i> on the safety set DPO-HH is better than SFT by 4–7 points of
win rate (54.3% order-corrected, 56.7% single order); on AlpacaEval DPO-UF and SFT are about equal (48.0–49.2%, not significant); and neither
DPO model changes results on the other set. <i>My interpretation:</i> the HH improvement is real but small, and it is uneven across categories
(left table). The biggest gains are in medical safety (70%), hate/harassment (63%) and privacy (62%), where DPO-HH more often suggests seeing
a professional, steers toward respectful behaviour, or leaves out the most harmful suggestion. It is <i>worse</i> on cyber abuse (35%) and
legal/financial safety (43%). What stood out most to me is how many ties there are: 27% of HH pairs in a single order and 47% after the order
correction, and in most of the tie explanations the judge says that <i>both</i> models failed, for example both listing weapons for “hurt
someone in a fight” or both writing the phishing e-mail they were asked for. With β = 0.1, LR 1e-6 and one epoch, DPO only moves the 0.5B model
a little away from an SFT model that never learned to refuse (UltraChat has almost no refusals, and on the HH set the refusal-phrase rate is
2–4% for every model). So the preference training changes <i>how</i> the model complies (shorter answers, 144 → 119 words, more hedging and
referrals) more than it teaches it to refuse. The data limits the effect too: single-turn HH pairs are noisy, with many chosen responses only
slightly better than the rejected ones, which matches the low validation margin (0.25) and accuracy (65%) compared with UltraFeedback.</p>
<p>For UltraFeedback the reward metrics looked good (validation accuracy 70%, margin 0.66), but the judged quality did not improve. The visible
change is in form. DPO-UF answers are 35% longer (142 → 191 words) and use lists more often (48% vs 39% of answers), which is the style of the
longer, more structured chosen responses in UltraFeedback. But a 0.5B model does not have much extra knowledge to fill that space. Across both
orders, the judge's explanations mention hallucinated or incorrect content in 144 of 254 DPO-UF losses but in only 76 of 229 wins (keyword count),
and the DPO-UF answers that lose are the ones that grew the most compared with SFT (197 vs 129 words). Since I told the judge not to reward
length, the more polished style does not make up for the extra mistakes. The result is the same across the AlpacaEval sources (46–52%, right table).</p>
""",
    "interpretation_2": """
<p><b>Possible explanations, linked to evidence.</b> (i) <i>Not enough training</i>: both DPO curves are still improving at step 469, the
margins are small, and the best checkpoint is the last one, so longer training or a sweep over LR and β would probably make the effect larger.
I did not run this because of the GPU budget. (ii) <i>A strong but limited SFT baseline</i>: SFT already fixed the main problems of the base model
(looping and never stopping: from 34–52% down to 1–5% of answers hitting the length limit), so most of the remaining errors are knowledge errors,
which 15k preference pairs cannot fix. (iii) <i>A mismatch with the preference data</i>: HH pairs reward deflecting more than clearly refusing,
and UltraFeedback rewards length and structure, which helps a bigger model more than a 0.5B one. (iv) <i>Judge sensitivity</i>: about a third of
the verdicts flip when the order is swapped, so differences of a few points are close to the limit of what 300 prompts can show. The
order-corrected numbers and the second judge both point in the same direction, though. For the persona task the trade-off is clear: the harder
I pushed DPO on a small style dataset, the more in-character the model became and the less useful and safe it was (table above; the LR 2e-4
adapter insults a user who says they feel hopeless, as shown in the examples below).</p>
""",
    "examples": [
        ("hh_dpo_hh", "hh_sft", "hh-eval-42-0004", "DPO-HH (win, both orders)", "SFT"),
        ("hh_dpo_hh", "hh_sft", "hh-eval-42-0000", "DPO-HH (tie: judge says both fail)", "SFT"),
        ("hh_dpo_hh", "hh_sft", "hh-eval-42-0006", "DPO-HH (loss)", "SFT"),
        ("alpaca_dpo_uf", "alpaca_sft", "alpaca-eval-281", "DPO-UF (win, both orders)", "SFT"),
        ("alpaca_dpo_uf", "alpaca_sft", "alpaca-eval-228", "DPO-UF (loss, both orders)", "SFT"),
        ("persona_lora_r16_all_lr5e-5", "persona_base", "persona-eval-30", "LoRA r16, lr 5e-5 (submitted)", "SmolLM2 base"),
        ("persona_lora_r16_all_lr2e-4", "persona_base", "persona-eval-36", "LoRA r16, lr 2e-4 (over-optimized)", "SmolLM2 base"),
    ],
    "limitations": """
<ul>
<li><b>Judge bias and noise.</b> One LLM judge (gemini-3.6-flash) decides every outcome. It has a measurable preference for the first
response (see “swap-consistent”), which I correct for by judging both orders, and it only moderately agrees with a second judge
(gemini-3.7-flash) on a 60-pair sample per comparison. It may also prefer longer or more confident answers even though I told it not to.</li>
<li><b>Small evaluation sets and a single seed.</b> With 300 prompts per set, a win rate has a 95% interval of roughly ±5–6 points, and the
persona results use only 40 prompts. I trained each model once (seed 42), so I could not measure seed variance.</li>
<li><b>Limited compute.</b> One epoch of DPO at LR 1e-6 is a short run (469 steps), and I did not try longer training or a larger β/LR sweep
within the Kaggle budget. The HH evaluation prompts are synthetic and templated, and I judged AlpacaEval pairwise against SFT rather than with
the official length-controlled AlpacaEval protocol.</li>
<li><b>Heuristic statistics.</b> The refusal-phrase rate is a keyword heuristic that I only use to describe behaviour.</li>
</ul>
<p><b>Reproducibility.</b> Configs and metrics: <code>results/{sft,dpo_hh,dpo_uf}/run_config.json</code>, <code>metrics.jsonl</code>,
<code>summary.json</code>; LoRA: <code>results/lora_dpo/*/summary.json</code>. Data: <code>results/data/data_manifest.json</code> (counts and the
id of every example). Generations: <code>results/generations/*.json</code> (with <code>.meta.json</code> decoding settings). Pairs:
<code>results/eval/pairs/</code>; judgments: <code>results/eval/judge_gemini-3.6-flash/</code> and <code>judge_gemini-3.7-flash/</code>; tables:
<code>results/eval/summary*.md</code>; plots: <code>results/plots/</code>; Kaggle logs: <code>results/logs/</code>. Notebooks:
<code>kaggle/alignment-0*/</code> (each one pins the git commit it ran). Tests: <code>PYTHONPATH=. pytest -q tests/</code> (the public smoke
tests plus my own edge-case and judge tests). The tables in this report are generated from these files by
<code>python report/build_report.py --pdf</code>.</p>
""",
}
