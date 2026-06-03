# Value-Function DPO

[English](README.md) | [中文](README_zh.md)

This method evaluates a task-progress value function as a critic for high-level embodied planning. It is used as a comparison against the successor-state consistency method.

## Idea

The value model scores a predicted post-action semantic state from 0 to 100 according to task progress.

For DPO pair construction:

```text
chosen   = higher-value policy candidate
rejected = lower-value policy candidate
```

This design tests whether the learned value function can act as a critic over policy-generated candidates, instead of always treating the expert trajectory action as the only valid positive action.

## Pipeline

```text
WAP trajectory data
        ↓
Policy LoRA SFT
        ↓
Semantic state predictor LoRA SFT
        ↓
Build value-function SFT data from trajectory progress
        ↓
Train task-progress value model
        ↓
Policy generates candidate actions
        ↓
State predictor predicts each candidate's post-action state
        ↓
Value function scores candidate states
        ↓
Construct value-DPO pairs
        ↓
DPO fine-tunes the policy
```

## Results

```text
Value SFT samples: 713,567
Value-DPO pairs: 1,793
DPO strict accuracy: 87.7%
Reward margin: 0.2277
```

Online value-selector evaluation:

```text
Policy top-1 accuracy: 86.9%
Value selector accuracy: 84.8%
Value selector accuracy on selectable samples: 86.6%
GT candidate coverage: 91.2%
```

The value function provides a useful but noisy preference signal. The main limitation is that trajectory-progress labels are relatively coarse, which reduces preference separability.

## Scripts

Method-specific scripts:

```text
scripts/build_value_function_data.py
scripts/train_value_function_lora.py
scripts/generate_value_function_pairs.py
scripts/evaluate_value_selector.py
```

Shared scripts used by this method:

```text
../../shared/scripts/train_policy_lora.py
../../shared/scripts/train_worldmodel_lora.py
../../shared/scripts/train_policy_dpo.py
../../shared/scripts/compute_dpo_reward_stats.py
../../shared/scripts/compare_policy_dpo.py
../../shared/scripts/split_wap_train_eval.py
../../shared/scripts/image_utils.py
../../shared/scripts/wap_sampling.py
```

## Example Run

```bash
cd /Users/yeats/Desktop/wap
python shared/scripts/train_policy_lora.py
python shared/scripts/train_worldmodel_lora.py
python methods/value_function_dpo/scripts/build_value_function_data.py
python methods/value_function_dpo/scripts/train_value_function_lora.py
python methods/value_function_dpo/scripts/generate_value_function_pairs.py
python shared/scripts/train_policy_dpo.py \
  --data_path ./data/processed/wap_qwen_policy_value_dpo_pairs.jsonl \
  --output_dir ./models/qwen2_5_vl_3b_wap_policy_value_dpo_lora
```

The repository does not include raw data, processed JSONL files, checkpoints, or LoRA adapters.
