# Small-Model Alignment: SFT + DPO on Qwen2.5-0.5B

Post-training pipeline for `Qwen/Qwen2.5-0.5B`:

1. **SFT** (from scratch, no HF `Trainer`) on 30,000 single-turn UltraChat examples.
2. **DPO** (from scratch, no TRL) from the SFT checkpoint, run twice:
   on 15,000 Anthropic HH-RLHF pairs and on 15,000 UltraFeedback pairs.
3. **LoRA-DPO persona tuning** of `HuggingFaceTB/SmolLM2-360M-Instruct` with TRL `DPOTrainer` + PEFT
   on `data/persona_lora/ramsay_style_preferences.json`.
4. **Evaluation**: greedy generations on fixed prompt sets, blinded A/B pair files, and a resumable
   LLM judge run through Kaggle Benchmarks (`kaggle_benchmarks`).

The final report is in [`report/report.pdf`](report/). The assignment manual is in [`docs/assignment_manual.pdf`](docs/assignment_manual.pdf).

## Layout

```text
submission_adapter.py        graded interface -> thin wrappers around alignment/
alignment/
  prompting.py               ChatML prompt template, tokenizer setup (EOS = <|im_end|>)
  batching.py                tokenization, EOS-before-truncation, right padding, -100 masks
  logprobs.py                shifted response-only summed log-probabilities (fp32)
  losses.py                  SFT token-mean NLL, DPO loss, reward metrics
  data.py                    UltraChat / HH-RLHF / UltraFeedback single-turn preparation
  sft.py                     SFT training loop (python -m alignment.sft)
  dpo.py                     DPO training loop (python -m alignment.dpo)
  generate.py                batched greedy generation (python -m alignment.generate)
scripts/
  prepare_data.py            build outputs/data/*.jsonl + data_manifest.json
  make_alpaca_subset.py      eval_sets/alpaca_eval_300_seed42.json
  train_lora_dpo.py          TRL DPOTrainer + LoRA persona runs
  build_pairs.py             blinded, order-randomized DPO-vs-SFT pair files
  judge_pairwise_kaggle.py   batched, retrying, resumable judge + summaries
  plot_metrics.py            training plots
kaggle/
  build_notebooks.py         generates the Kaggle notebooks for every stage
  judge_task.py              Kaggle Benchmarks task that runs the judge
eval_sets/                   fixed evaluation prompts (HH synthetic 300, AlpacaEval 300 seed 42, persona 40)
results/                     metrics, generations, pair files, judgments, plots copied back from Kaggle
tests/                       public smoke tests + stricter edge-case and judge tests
```

## Implementation notes

* **Prompt format** (SFT, DPO and evaluation):
  `<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>`.
  The tokenizer's EOS is set to `<|im_end|>` and PAD to `<|endoftext|>`.
* **Masking**: prompt and response are tokenized separately (`add_special_tokens=False`) and
  concatenated; EOS is appended to the response *before* right-truncation to `max_length`; prompt and
  padding labels are `-100`; rows are right-padded (the adapter pads to exactly `max_length`).
* **Log-probabilities**: logits at position *t* score the label at *t+1*; sums over non-`-100`
  labels only, computed in fp32.
* **SFT loss reduction**: mean NLL over all unmasked response tokens (token-weighted across the whole
  gradient-accumulation window during training).
* **DPO**: `-log σ(β[(log π(y_w|x) − log π(y_l|x)) − (log π_ref(y_w|x) − log π_ref(y_l|x))])`, batch mean.
  The reference is a frozen copy of the SFT checkpoint on the second GPU, always under `torch.no_grad()`.
* **Precision**: fp32 trainable weights, fp16 autocast, `GradScaler`, gradient clipping 1.0.

## Reproducing

```bash
python -m pip uninstall -y -q torchao || true
python -m pip install -q "transformers>=4.43.0,<5" "datasets>=2.18,<5" huggingface_hub accelerate tqdm pyyaml sentencepiece safetensors pytest matplotlib
python -m pip install -q -e . --no-deps
PYTHONPATH=. python -m pytest -q tests/

python scripts/prepare_data.py --out_dir outputs/data
python -m alignment.sft --train_path outputs/data/sft_train.jsonl --val_path outputs/data/sft_val.jsonl --output_dir outputs/sft --eval_every 50
python -m alignment.dpo --sft_path outputs/sft/best --train_path outputs/data/dpo_hh_train.jsonl --val_path outputs/data/dpo_hh_val.jsonl --output_dir outputs/dpo_hh --eval_every 25
python -m alignment.dpo --sft_path outputs/sft/best --train_path outputs/data/dpo_uf_train.jsonl --val_path outputs/data/dpo_uf_val.jsonl --output_dir outputs/dpo_uf --eval_every 25
CUDA_VISIBLE_DEVICES=0 python scripts/train_lora_dpo.py --output_dir outputs/lora_dpo
```

On Kaggle (GPU T4 ×2) the same commands are run by the notebooks generated with
`python kaggle/build_notebooks.py --username <kaggle-user> --commit <sha>` and pushed with
`kaggle kernels push -p kaggle/<stage>`:
`alignment-00-pilot` → `alignment-01-sft` → (`alignment-02-dpo-hh` ∥ `alignment-03-dpo-uf`), plus
`alignment-04-lora-dpo`. Each later stage attaches the previous stage's output via `kernel_sources`.

Judging runs from saved pair files only:

```bash
python scripts/build_pairs.py --model_path results/generations/alpaca_dpo_uf.json --model_name dpo_uf \
  --baseline_path results/generations/alpaca_sft.json --baseline_name sft --rubric general \
  --output_path results/eval/pairs/dpo_uf_vs_sft_pairs.jsonl
python scripts/judge_pairwise_kaggle.py --pairs_path results/eval/pairs/dpo_uf_vs_sft_pairs.jsonl \
  --output_path results/eval/dpo_uf_vs_sft_judged.jsonl --judge_model google/gemini-3.6-flash \
  --batch_size 5 --sleep_seconds 5 --max_retries 3 --reuse_judgments
```

(`judge_pairwise_kaggle.py` needs `kaggle_benchmarks`, which is available inside a Kaggle benchmark
notebook; `kaggle/judge_task.py` wraps it as a benchmark task.)
