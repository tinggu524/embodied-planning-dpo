# 后继语义状态一致性 DPO

[English](README.md) | [中文](README_zh.md)

该方法使用语义状态预测模型来构造 DPO 偏好对，是本项目中效果更好的最终方案。

## 核心思路

首先由 SFT policy 生成多个候选高层动作。对于每个候选动作，将其放入语义状态预测模型的输入中，计算模型生成专家后继语义状态的平均 token NLL。

如果一个候选动作是错误的，那么在给定该动作的条件下，模型通常更难生成真实轨迹中的后继状态，因此 NLL 会更高。

DPO pair 的构造方式为：

```text
chosen   = 专家轨迹动作
rejected = policy 生成的 hard negative，且与专家动作的 successor-state NLL margin 最大
```

这里的专家后继状态只用于离线偏好构造。测试时，DPO policy 不读取未来状态，而是直接预测下一步动作。

## 流程

```text
WAP 轨迹数据
        ↓
Policy LoRA SFT
        ↓
语义状态预测模型 LoRA SFT
        ↓
Policy 生成候选动作
        ↓
语义状态预测模型计算专家后继状态 NLL
        ↓
构造 DPO 偏好对
        ↓
DPO 微调 policy
```

## 偏好过滤

为了提高偏好对质量，生成 pair 时会进行多种过滤：

```text
去除重复或近似重复动作
过滤动词-对象几乎相同的候选
检查推理文本和最终动作是否一致
保留 successor-state NLL margin 足够大的 hard negative
```

最终保留下来的 rejected action 不是随机错误动作，而是 policy 容易生成、但无法很好解释专家后继状态的困难负样本。

## 实验结果

```text
Policy SFT 严格动作准确率：87.0%
DPO pairs：2,219
DPO 后严格动作准确率：88.3%
Reward margin：0.4107
```

相比任务进度价值函数 DPO，该方法在准确率和 reward margin 上都有更明显提升。

## 主要脚本

方法特有脚本：

```text
scripts/generate_policy_dpo_pairs.py
```

该方法使用的共用脚本：

```text
../../shared/scripts/train_policy_lora.py
../../shared/scripts/train_worldmodel_lora.py
../../shared/scripts/train_policy_dpo.py
../../shared/scripts/compute_dpo_reward_stats.py
../../shared/scripts/compare_policy_dpo.py
../../shared/scripts/merge_compare_results.py
../../shared/scripts/split_wap_train_eval.py
../../shared/scripts/image_utils.py
../../shared/scripts/wap_sampling.py
```

## 示例运行

```bash
cd /Users/yeats/Desktop/wap

python shared/scripts/train_policy_lora.py

python shared/scripts/train_worldmodel_lora.py

python methods/successor_state_dpo/scripts/generate_policy_dpo_pairs.py

python shared/scripts/train_policy_dpo.py
```

原始数据、处理后的 JSONL、模型权重、LoRA adapter、日志和评估结果不包含在本仓库中。
