# Experiment folders

This directory separates the runnable Python scripts by experiment line.

- `sft_dpo/`: current SFT + DPO pipeline.
- `sft_wm_value_function/`: new workspace for SFT + world model + value-function style scoring.

Datasets are not copied here. Both folders reuse the root-level paths:

- `data/raw/`
- `data/processed/`
- `models/`
- `results/`

Run scripts from the project root, for example:

```bash
cd /Users/yeats/Desktop/wap
python experiments/sft_dpo/scripts/train_policy_lora.py
```

