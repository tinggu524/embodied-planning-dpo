# SFT + DPO

This folder keeps the current SFT + DPO experiment scripts together.

Included scripts:

- `train_policy_lora.py`: train the SFT policy LoRA.
- `generate_policy_dpo_pairs.py`: generate preference pairs for DPO.
- `train_policy_dpo.py`: train the DPO LoRA from preference pairs.
- `compare_policy_dpo.py`: compare SFT and DPO policy outputs.
- `compute_dpo_reward_stats.py`: compute chosen/rejected reward statistics.
- `merge_compare_results.py`: merge comparison shards.
- `split_wap_train_eval.py`: create train/eval splits from processed jsonl files.
- `image_utils.py`, `wap_sampling.py`: copied common helpers.

Data is not duplicated. Run from the project root so default paths such as
`./data/processed/...` still point to the shared dataset.

Example:

```bash
cd /Users/yeats/Desktop/wap
python experiments/sft_dpo/scripts/train_policy_lora.py
python experiments/sft_dpo/scripts/train_policy_dpo.py
```

