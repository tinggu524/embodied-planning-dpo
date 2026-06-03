# SFT + WM + Value Function

This is the new workspace for revising the older scorer into a value-function
style pipeline.

Included scripts:

- `train_policy_lora.py`: train the SFT policy LoRA.
- `train_worldmodel_lora.py`: train the semantic state / world-state model.
- `generate_policy_dpo_pairs.py`: current WM-based pair generator, copied as the starting point for value-function scoring.
- `train_policy_dpo.py`: reusable if value-based preference pairs are later trained with DPO.
- `compare_policy_dpo.py`: reusable policy comparison script.
- `compute_dpo_reward_stats.py`: reusable reward-stat script.
- `split_wap_train_eval.py`: shared train/eval split utility.
- `image_utils.py`, `wap_sampling.py`: copied common helpers.

Recommended next files to add here:

- `build_value_function_data.py`
- `train_value_function_lora.py`
- `generate_value_function_pairs.py`

The intended change is to replace direct next-action or future-label scoring
with a task-progress value estimate:

```text
task + image/state + history + candidate action -> progress value
```

Data is not duplicated. Run from the project root so default paths such as
`./data/processed/...` still point to the shared dataset.

Minimal run order:

```bash
cd /Users/yeats/Desktop/wap
python experiments/sft_wm_value_function/scripts/build_value_function_data.py
python experiments/sft_wm_value_function/scripts/train_value_function_lora.py
python experiments/sft_wm_value_function/scripts/generate_value_function_pairs.py
python experiments/sft_wm_value_function/scripts/train_policy_dpo.py \
  --data_path ./data/processed/wap_qwen_policy_value_dpo_pairs.jsonl \
  --output_dir ./models/qwen2_5_vl_3b_wap_policy_value_dpo_lora
```
