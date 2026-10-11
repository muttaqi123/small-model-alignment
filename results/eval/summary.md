| comparison | n | wins | losses | ties | unknown | failures | win rate (excl. ties) | win rate (ties = 1/2) | judge picked A |
|---|---|---|---|---|---|---|---|---|---|
| dpo_hh_vs_sft_alpaca | 300 | 108 | 107 | 85 | 0 | 0 | 50.2% | 50.2% | 61.4% |
| dpo_hh_vs_sft_hh | 300 | 130 | 90 | 80 | 0 | 0 | 59.1% | 56.7% | 60.5% |
| dpo_uf_vs_sft_alpaca | 300 | 119 | 124 | 57 | 0 | 0 | 49.0% | 49.2% | 56.0% |
| dpo_uf_vs_sft_hh | 300 | 120 | 118 | 62 | 0 | 0 | 50.4% | 50.3% | 59.2% |
| persona_quality_lora_r16_all_lr2e-4_vs_base | 40 | 6 | 34 | 0 | 0 | 0 | 15.0% | 15.0% | 47.5% |
| persona_quality_lora_r16_all_lr5e-5_vs_base | 40 | 15 | 22 | 3 | 0 | 0 | 40.5% | 41.2% | 64.9% |
| persona_quality_lora_r8_attn_lr5e-5_vs_base | 40 | 14 | 17 | 9 | 0 | 0 | 45.2% | 46.2% | 80.6% |
| persona_style_lora_r16_all_lr2e-4_vs_base | 40 | 21 | 19 | 0 | 0 | 0 | 52.5% | 52.5% | 50.0% |
| persona_style_lora_r16_all_lr5e-5_vs_base | 40 | 19 | 15 | 6 | 0 | 0 | 55.9% | 55.0% | 58.8% |
| persona_style_lora_r8_attn_lr5e-5_vs_base | 40 | 7 | 5 | 28 | 0 | 0 | 58.3% | 52.5% | 75.0% |

| comparison | swap consistency | order-debiased W/L/T | order-debiased win rate | second-judge win rate | judge agreement |
|---|---|---|---|---|---|
| dpo_hh_vs_sft_alpaca | - | - | - | 46.7% | 81.7% |
| dpo_hh_vs_sft_hh | 72.7% | 93/67/140 | 54.3% | 59.2% | 81.7% |
| dpo_uf_vs_sft_alpaca | 68.0% | 78/90/132 | 48.0% | 50.0% | 83.3% |
| dpo_uf_vs_sft_hh | - | - | - | 60.0% | 81.7% |
| persona_quality_lora_r16_all_lr2e-4_vs_base | 85.0% | 5/29/6 | 20.0% | 17.5% | 85.0% |
| persona_quality_lora_r16_all_lr5e-5_vs_base | 77.5% | 12/18/10 | 42.5% | 45.0% | 90.0% |
| persona_quality_lora_r8_attn_lr5e-5_vs_base | 75.0% | 11/11/18 | 50.0% | 42.5% | 80.0% |
| persona_style_lora_r16_all_lr2e-4_vs_base | 55.0% | 15/7/18 | 60.0% | 72.5% | 60.0% |
| persona_style_lora_r16_all_lr5e-5_vs_base | 75.0% | 16/12/12 | 55.0% | 56.2% | 80.0% |
| persona_style_lora_r8_attn_lr5e-5_vs_base | 75.0% | 3/5/32 | 47.5% | 37.5% | 50.0% |

**dpo_hh_vs_sft_hh** by category

| category | n | W | L | T | win rate (ties = 1/2) |
|---|---|---|---|---|---|
| benign_helpfulness | 30 | 18 | 10 | 2 | 63.3% |
| calibration_and_uncertainty | 30 | 10 | 8 | 12 | 53.3% |
| cyber_abuse | 30 | 6 | 13 | 11 | 38.3% |
| harmful_instructions | 30 | 15 | 7 | 8 | 63.3% |
| hate_or_harassment | 30 | 13 | 4 | 13 | 65.0% |
| legal_financial_safety | 30 | 14 | 14 | 2 | 50.0% |
| medical_safety | 30 | 19 | 7 | 4 | 70.0% |
| privacy_and_doxxing | 30 | 13 | 6 | 11 | 61.7% |
| safe_transformation | 30 | 10 | 12 | 8 | 46.7% |
| self_harm_support | 30 | 12 | 9 | 9 | 55.0% |

**dpo_uf_vs_sft_hh** by category

| category | n | W | L | T | win rate (ties = 1/2) |
|---|---|---|---|---|---|
| benign_helpfulness | 30 | 12 | 15 | 3 | 45.0% |
| calibration_and_uncertainty | 30 | 14 | 12 | 4 | 53.3% |
| cyber_abuse | 30 | 16 | 9 | 5 | 61.7% |
| harmful_instructions | 30 | 8 | 9 | 13 | 48.3% |
| hate_or_harassment | 30 | 7 | 13 | 10 | 40.0% |
| legal_financial_safety | 30 | 14 | 9 | 7 | 58.3% |
| medical_safety | 30 | 12 | 15 | 3 | 45.0% |
| privacy_and_doxxing | 30 | 10 | 8 | 12 | 53.3% |
| safe_transformation | 30 | 16 | 14 | 0 | 53.3% |
| self_harm_support | 30 | 11 | 14 | 5 | 45.0% |

**persona_quality_lora_r16_all_lr2e-4_vs_base** by category

| category | n | W | L | T | win rate (ties = 1/2) |
|---|---|---|---|---|---|
| coding | 10 | 2 | 8 | 0 | 20.0% |
| cooking | 10 | 1 | 9 | 0 | 10.0% |
| general_help | 10 | 0 | 10 | 0 | 0.0% |
| safety_sensitive | 10 | 3 | 7 | 0 | 30.0% |

**persona_quality_lora_r16_all_lr5e-5_vs_base** by category

| category | n | W | L | T | win rate (ties = 1/2) |
|---|---|---|---|---|---|
| coding | 10 | 4 | 6 | 0 | 40.0% |
| cooking | 10 | 4 | 5 | 1 | 45.0% |
| general_help | 10 | 2 | 8 | 0 | 20.0% |
| safety_sensitive | 10 | 5 | 3 | 2 | 60.0% |

**persona_quality_lora_r8_attn_lr5e-5_vs_base** by category

| category | n | W | L | T | win rate (ties = 1/2) |
|---|---|---|---|---|---|
| coding | 10 | 4 | 4 | 2 | 50.0% |
| cooking | 10 | 4 | 4 | 2 | 50.0% |
| general_help | 10 | 4 | 4 | 2 | 50.0% |
| safety_sensitive | 10 | 2 | 5 | 3 | 35.0% |

**persona_style_lora_r16_all_lr2e-4_vs_base** by category

| category | n | W | L | T | win rate (ties = 1/2) |
|---|---|---|---|---|---|
| coding | 10 | 4 | 6 | 0 | 40.0% |
| cooking | 10 | 9 | 1 | 0 | 90.0% |
| general_help | 10 | 3 | 7 | 0 | 30.0% |
| safety_sensitive | 10 | 5 | 5 | 0 | 50.0% |

**persona_style_lora_r16_all_lr5e-5_vs_base** by category

| category | n | W | L | T | win rate (ties = 1/2) |
|---|---|---|---|---|---|
| coding | 10 | 2 | 4 | 4 | 40.0% |
| cooking | 10 | 10 | 0 | 0 | 100.0% |
| general_help | 10 | 1 | 8 | 1 | 15.0% |
| safety_sensitive | 10 | 6 | 3 | 1 | 65.0% |

**persona_style_lora_r8_attn_lr5e-5_vs_base** by category

| category | n | W | L | T | win rate (ties = 1/2) |
|---|---|---|---|---|---|
| coding | 10 | 1 | 2 | 7 | 45.0% |
| cooking | 10 | 3 | 0 | 7 | 65.0% |
| general_help | 10 | 1 | 2 | 7 | 45.0% |
| safety_sensitive | 10 | 2 | 1 | 7 | 55.0% |
