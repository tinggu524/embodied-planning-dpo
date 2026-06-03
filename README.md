# Embodied Planning with Multimodal SFT and Preference Optimization

<p align="center">
  <a href="https://github.com/tinggu524/embodied-planning-dpo"><img src="https://img.shields.io/badge/Project%20Page-GitHub-4b5563?style=for-the-badge&logo=github" alt="Project Page"></a>
  <a href="https://github.com/tinggu524/embodied-planning-dpo"><img src="https://img.shields.io/badge/GitHub-Repo-181717?style=for-the-badge&logo=github" alt="GitHub"></a>
  <a href="#methods"><img src="https://img.shields.io/badge/Method-LoRA%20SFT%20%2B%20DPO-2e7d32?style=for-the-badge" alt="Method"></a>
  <a href="#data-and-model-paths"><img src="https://img.shields.io/badge/Dataset-WAP%20Trajectories-1565c0?style=for-the-badge" alt="Dataset"></a>
  <a href="https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct"><img src="https://img.shields.io/badge/Model-Qwen2.5--VL--3B-f57c00?style=for-the-badge" alt="Model"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-6a1b9a?style=for-the-badge" alt="License"></a>
</p>

<p align="center">
  <strong>English</strong> / <a href="README_zh.md">中文</a>
</p>

This repository contains a multimodal high-level planning project for embodied AI tasks based on the World-Aware Planning (WAP) trajectory data. The project studies how to improve a vision-language policy with supervised fine-tuning and preference optimization.

The core question is:

> Given a task instruction, the current visual observation, and previous high-level actions, how can a multimodal policy predict the next semantic action more reliably?

The project compares two preference-construction strategies:

1. **Value-Function DPO**: uses a learned task-progress value model to score candidate post-action semantic states.
2. **Successor-State Consistency DPO**: uses a semantic state prediction model to identify hard negative actions that cannot explain the expert successor state.

Policy SFT, semantic state predictor training, DPO training, and shared evaluation utilities are placed under `shared/scripts/` because both methods use the same policy and semantic-state backbone.

<a id="table-of-contents"></a>

## 📚 Table of Contents

- [🧭 Project Overview](#project-overview)
- [🧠 Methods](#methods)
- [📊 Results Summary](#results-summary)
- [🗂️ Repository Structure](#repository-structure)
- [🛠️ Main Scripts](#main-scripts)
- [📁 Data and Model Paths](#data-and-model-paths)
- [📝 Notes](#notes)
- [⚖️ License](#license)

<a id="project-overview"></a>

## 🧭 Project Overview

The base policy is trained with Qwen2.5-VL-3B-Instruct and LoRA SFT. It predicts high-level semantic actions rather than low-level robot controls.

Input:

```text
Task instruction
Current image observation
Previous high-level actions
```

Output:

```text
Next high-level semantic action
```

Example actions include:

```text
find a newspaper
pick up the newspaper
open the fridge
put down the tomato
done
```

<a id="methods"></a>

## 🧠 Methods

### 🔹 1. Policy SFT

The initial policy is trained from WAP trajectory data. Each training sample contains an instruction, a visual observation, the history of previous high-level actions, and the expert next action.

Result:

```text
Strict action accuracy: 87.0%
```

### 🔹 2. Value-Function DPO

This method trains a value model that estimates task progress from a predicted post-action semantic state.

Pipeline:

```text
SFT policy generates candidate actions
        ↓
Semantic state predictor predicts post-action states
        ↓
Value function scores each predicted state from 0 to 100
        ↓
Higher-value action is used as chosen
Lower-value action is used as rejected
        ↓
DPO fine-tunes the policy
```

This method evaluates whether a learned critic can guide action selection without directly treating the dataset action as the only valid next step.

Result:

```text
Value-DPO pairs: 1,793
DPO strict action accuracy: 87.7%
Reward margin: 0.2277
```

Additional online selector evaluation:

```text
Policy top-1 accuracy: 86.9%
Value selector accuracy: 84.8%
Value selector accuracy on selectable samples: 86.6%
GT candidate coverage: 91.2%
```

This suggests that the value function provides a meaningful but noisy preference signal, mainly limited by coarse trajectory-progress labels.

### 🔹 3. Successor-State Consistency DPO

This method constructs preference pairs by comparing how well different candidate actions explain the expert successor semantic state.

Pipeline:

```text
SFT policy generates candidate actions
        ↓
For each candidate action, replace "Action to execute"
        ↓
Semantic state predictor computes NLL of the expert successor state
        ↓
GT action is used as chosen
Candidate with the largest NLL margin is used as rejected
        ↓
DPO fine-tunes the policy
```

The intuition is that an incorrect high-level action should make the expert successor state harder for the semantic state predictor to generate.

Result:

```text
DPO pairs: 2,219
DPO strict action accuracy: 88.3%
Reward margin: 0.4107
```

This method is used as the stronger final pipeline in this project.

<a id="results-summary"></a>

## 📊 Results Summary

| Method | Preference Signal | DPO Pairs | Strict Accuracy | Reward Margin |
| --- | --- | ---: | ---: | ---: |
| Policy SFT | Supervised expert action | - | 87.0% | - |
| Value-Function DPO | Task-progress value score | 1,793 | 87.7% | 0.2277 |
| Successor-State Consistency DPO | Expert successor-state NLL margin | 2,219 | 88.3% | 0.4107 |

<a id="repository-structure"></a>

## 🗂️ Repository Structure

```text
.
├── README.md
├── README_zh.md
├── data_preparation/
│   ├── convert_wap.py
│   └── resize_wap_images.py
├── shared/
│   └── scripts/
│       ├── train_policy_lora.py
│       ├── train_worldmodel_lora.py
│       ├── train_policy_dpo.py
│       ├── compute_dpo_reward_stats.py
│       ├── compare_policy_dpo.py
│       ├── merge_compare_results.py
│       ├── split_wap_train_eval.py
│       ├── image_utils.py
│       └── wap_sampling.py
└── methods/
    ├── value_function_dpo/
    │   ├── README.md
    │   ├── README_zh.md
    │   └── scripts/
    │       ├── build_value_function_data.py
    │       ├── train_value_function_lora.py
    │       ├── generate_value_function_pairs.py
    │       └── evaluate_value_selector.py
    └── successor_state_dpo/
        ├── README.md
        ├── README_zh.md
        └── scripts/
            └── generate_policy_dpo_pairs.py
```

Large files such as raw data, processed JSONL files, model checkpoints, LoRA adapters, logs, and evaluation outputs are intentionally excluded from the repository.

<a id="main-scripts"></a>

## 🛠️ Main Scripts

Common data preparation:

```text
data_preparation/convert_wap.py
data_preparation/resize_wap_images.py
```

Shared training, DPO, and evaluation scripts:

```text
shared/scripts/train_policy_lora.py
shared/scripts/train_worldmodel_lora.py
shared/scripts/train_policy_dpo.py
shared/scripts/compute_dpo_reward_stats.py
shared/scripts/compare_policy_dpo.py
shared/scripts/merge_compare_results.py
```

Value-function DPO:

```text
methods/value_function_dpo/scripts/build_value_function_data.py
methods/value_function_dpo/scripts/train_value_function_lora.py
methods/value_function_dpo/scripts/generate_value_function_pairs.py
methods/value_function_dpo/scripts/evaluate_value_selector.py
```

Successor-state consistency DPO:

```text
methods/successor_state_dpo/scripts/generate_policy_dpo_pairs.py
```

<a id="data-and-model-paths"></a>

## 📁 Data and Model Paths

The scripts assume the following local paths:

```text
data/raw/World-Aware-Planning/
data/processed/
models/Qwen2.5-VL-3B-Instruct/
models/qwen2_5_vl_3b_wap_policy_lora/
models/qwen2_5_vl_3b_wap_worldmodel_lora/
models/qwen2_5_vl_3b_wap_value_function_lora/
```

These files are not included in this repository.

<a id="notes"></a>

## 📝 Notes

- This project operates at the high-level semantic action level.
- The successor-state consistency method uses expert successor states only for offline preference construction, not for test-time inference.
- At inference time, the final DPO policy directly predicts the next high-level action from the current observation, instruction, and action history.

<a id="license"></a>

## ⚖️ License

This project is released under the [MIT License](LICENSE).
