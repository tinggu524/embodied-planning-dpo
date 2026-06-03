# 面向具身智能的多模态高层规划与偏好优化

[English](README.md) | [中文](README_zh.md)

本仓库整理了一个基于 World-Aware Planning (WAP) 轨迹数据的多模态高层规划项目。项目关注的问题是：在给定任务指令、当前视觉观测和历史高层动作的情况下，如何让视觉语言策略模型更可靠地预测下一步语义动作。

项目包含两个偏好构造方案，并通过 DPO 对策略模型进行进一步优化：

1. **任务进度价值函数 DPO**：训练一个 value function，对候选动作后的预测语义状态进行任务进度打分。
2. **后继语义状态一致性 DPO**：使用语义状态预测模型，构造不能解释专家后继状态的 hard negative 动作。

Policy SFT、语义状态预测模型训练、DPO 训练和通用评估工具放在 `shared/scripts/` 下，因为两个方法共用同一个策略模型和语义状态预测模块。

## 项目任务

基础策略模型使用 `Qwen2.5-VL-3B-Instruct` 进行 LoRA SFT，预测的是高层语义动作，而不是底层机器人控制信号。

输入形式：

```text
任务指令
当前图像观测
历史高层动作
```

输出形式：

```text
下一步高层语义动作
```

动作示例：

```text
find a newspaper
pick up the newspaper
open the fridge
put down the tomato
done
```

## 方法概览

### 1. Policy SFT

首先从 WAP 轨迹数据中构建高层规划训练集。每条样本包含任务指令、当前图像、历史动作和专家下一步动作。

结果：

```text
严格动作准确率：87.0%
```

### 2. 任务进度价值函数 DPO

该方法训练一个 value function，用于估计预测语义状态距离任务完成的进度。

流程：

```text
SFT policy 生成候选动作
        ↓
语义状态预测模型预测每个动作后的状态
        ↓
Value function 对预测状态打 0 到 100 的进度分
        ↓
高分候选作为 chosen
低分候选作为 rejected
        ↓
使用 DPO 继续微调 policy
```

这个方案的目的不是简单把专家动作固定为正样本，而是测试一个学习到的 critic 是否可以在 policy 生成的候选动作之间提供偏好信号。

结果：

```text
Value-DPO pairs：1,793
DPO 后严格动作准确率：87.7%
Reward margin：0.2277
```

额外的在线 value selector 评估：

```text
Policy top-1 accuracy：86.9%
Value selector accuracy：84.8%
Selectable samples 上的 value selector accuracy：86.6%
GT candidate coverage：91.2%
```

结果表明，value function 能提供一定偏好信号，但受限于轨迹进度标签较粗，候选动作区分度不足。

### 3. 后继语义状态一致性 DPO

该方法通过比较不同候选动作解释专家后继语义状态的难度来构造偏好对。

流程：

```text
SFT policy 生成候选动作
        ↓
对每个候选动作替换 “Action to execute”
        ↓
语义状态预测模型计算专家后继状态的 NLL
        ↓
专家动作作为 chosen
NLL margin 最大的错误候选作为 rejected
        ↓
使用 DPO 继续微调 policy
```

直觉是：如果一个高层动作是错误的，那么在给定该动作的条件下，模型应该更难生成真实轨迹中的后继语义状态。

结果：

```text
DPO pairs：2,219
DPO 后严格动作准确率：88.3%
Reward margin：0.4107
```

该方案是本项目中效果更好的最终方法。

## 结果对比

| 方法 | 偏好信号 | DPO Pairs | 严格动作准确率 | Reward Margin |
| --- | --- | ---: | ---: | ---: |
| Policy SFT | 专家动作监督 | - | 87.0% | - |
| 任务进度价值函数 DPO | Value function 进度分 | 1,793 | 87.7% | 0.2277 |
| 后继语义状态一致性 DPO | 专家后继状态 NLL margin | 2,219 | 88.3% | 0.4107 |

## 仓库结构

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

原始数据、处理后的 JSONL、模型权重、LoRA adapter、日志和评估结果等大文件不会提交到仓库中。

## 主要脚本

通用数据预处理：

```text
data_preparation/convert_wap.py
data_preparation/resize_wap_images.py
```

共用训练、DPO 和评估脚本：

```text
shared/scripts/train_policy_lora.py
shared/scripts/train_worldmodel_lora.py
shared/scripts/train_policy_dpo.py
shared/scripts/compute_dpo_reward_stats.py
shared/scripts/compare_policy_dpo.py
shared/scripts/merge_compare_results.py
```

任务进度价值函数 DPO：

```text
methods/value_function_dpo/scripts/build_value_function_data.py
methods/value_function_dpo/scripts/train_value_function_lora.py
methods/value_function_dpo/scripts/generate_value_function_pairs.py
methods/value_function_dpo/scripts/evaluate_value_selector.py
```

后继语义状态一致性 DPO：

```text
methods/successor_state_dpo/scripts/generate_policy_dpo_pairs.py
```

## 本地路径约定

脚本默认使用以下本地路径：

```text
data/raw/World-Aware-Planning/
data/processed/
models/Qwen2.5-VL-3B-Instruct/
models/qwen2_5_vl_3b_wap_policy_lora/
models/qwen2_5_vl_3b_wap_worldmodel_lora/
models/qwen2_5_vl_3b_wap_value_function_lora/
```

这些文件需要在本地或训练服务器上自行准备，不包含在 Git 仓库中。

## 说明

- 本项目工作在高层语义动作层面，不涉及底层连续控制。
- 后继语义状态一致性方法只在离线偏好构造阶段使用专家后继状态，测试时不使用未来轨迹信息。
- 最终推理时，DPO policy 直接根据当前观测、任务指令和历史动作预测下一步高层动作。
