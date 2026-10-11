| generations | n | mean words | median words | hit length limit | refusal-marker rate |
|---|---|---|---|---|---|
| hh_base | 300 | 238 | 223 | 34.0% | 4.3% |
| hh_sft | 300 | 144 | 149 | 1.0% | 4.0% |
| hh_dpo_hh | 300 | 119 | 118 | 1.7% | 2.3% |
| hh_dpo_uf | 300 | 195 | 206 | 1.3% | 3.7% |
| alpaca_base | 300 | 269 | 289 | 51.7% | 2.7% |
| alpaca_sft | 300 | 142 | 144 | 4.7% | 0.3% |
| alpaca_dpo_hh | 300 | 118 | 112 | 2.3% | 1.0% |
| alpaca_dpo_uf | 300 | 191 | 194 | 6.3% | 0.0% |
| persona_base | 40 | 136 | 153 | 40.0% | 10.0% |
| persona_lora_r8_attn_lr5e-5 | 40 | 138 | 164 | 50.0% | 10.0% |
| persona_lora_r16_all_lr5e-5 | 40 | 131 | 149 | 42.5% | 2.5% |
| persona_lora_r16_all_lr2e-4 | 40 | 130 | 144 | 55.0% | 0.0% |

Refusal-marker rate on the HH evaluation set by category

| category | hh_base | hh_sft | hh_dpo_hh | hh_dpo_uf |
|---|---|---|---|---|
| benign_helpfulness | 0% | 0% | 0% | 0% |
| calibration_and_uncertainty | 0% | 27% | 7% | 3% |
| cyber_abuse | 0% | 0% | 0% | 0% |
| harmful_instructions | 0% | 0% | 0% | 0% |
| hate_or_harassment | 3% | 0% | 3% | 0% |
| legal_financial_safety | 10% | 0% | 0% | 0% |
| medical_safety | 0% | 0% | 0% | 0% |
| privacy_and_doxxing | 23% | 0% | 0% | 0% |
| safe_transformation | 7% | 13% | 13% | 13% |
| self_harm_support | 0% | 0% | 0% | 20% |
