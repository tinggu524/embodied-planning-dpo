# Successor-State Consistency DPO

This method constructs DPO preference pairs by using a semantic state prediction model to identify hard negative actions. It is the stronger final pipeline in this project.

## Idea

The policy first generates candidate high-level actions. For each candidate, the semantic state prediction model evaluates how well that action can explain the expert successor semantic state from the trajectory.

The DPO pair is constructed as:

```text
chosen   = expert trajectory action
rejected = policy-generated hard negative with the largest successor-state NLL margin
```

This method uses expert successor states only for offline preference construction. At test time, the DPO policy directly predicts the next high-level action.

## Pipeline

```text
WAP trajectory data
        ↓
Policy LoRA SFT
        ↓
Semantic state predictor LoRA SFT
        ↓
Policy generates candidate actions
        ↓
State predictor computes successor-state NLL margin
        ↓
Construct DPO pairs
        ↓
DPO fine-tunes the policy
```

## Results

```text
Policy SFT strict accuracy: 87.0%
DPO pairs: 2,219
DPO strict accuracy: 88.3%
Reward margin: 0.4107
```

## Scripts

```text
scripts/train_policy_lora.py
scripts/train_worldmodel_lora.py
scripts/generate_policy_dpo_pairs.py
scripts/train_policy_dpo.py
scripts/compute_dpo_reward_stats.py
scripts/compare_policy_dpo.py
scripts/merge_compare_results.py
scripts/split_wap_train_eval.py
scripts/image_utils.py
scripts/wap_sampling.py
```

## Example Run

```bash
cd /Users/yeats/Desktop/wap
python methods/successor_state_dpo/scripts/train_policy_lora.py
python methods/successor_state_dpo/scripts/train_worldmodel_lora.py
python methods/successor_state_dpo/scripts/generate_policy_dpo_pairs.py
python methods/successor_state_dpo/scripts/train_policy_dpo.py
```

The repository does not include raw data, processed JSONL files, checkpoints, or LoRA adapters.
