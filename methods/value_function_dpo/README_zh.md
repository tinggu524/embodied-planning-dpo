# 任务进度价值函数 DPO

<p align="center">
  <a href="README.md">English</a> / <strong>中文</strong>
</p>

该方法训练一个任务进度 value function，用它作为 critic 给 policy 生成的候选动作打分，并构造 DPO 偏好对。它在本项目中作为对比方案，用于分析 value-function critic 在高层具身规划任务中的效果。

## 核心思路

Value function 的输入包含任务指令、当前图像、历史动作、候选动作以及该候选动作后的预测语义状态。模型输出一个 0 到 100 的整数分数，表示当前预测状态距离任务完成的进度。

DPO pair 的构造方式为：

```text
chosen   = value 分数更高的 policy candidate
rejected = value 分数更低的 policy candidate
```

这个设计不是把专家动作固定作为 chosen，而是测试学习到的 value function 是否可以在 policy 自己生成的候选动作之间提供有效偏好信号。

## 流程

```text
WAP 轨迹数据
        ↓
Policy LoRA SFT
        ↓
语义状态预测模型 LoRA SFT
        ↓
根据轨迹进度构造 value-function SFT 数据
        ↓
训练任务进度 value function
        ↓
Policy 生成候选动作
        ↓
语义状态预测模型预测每个候选动作后的状态
        ↓
Value function 对候选状态打分
        ↓
构造 value-DPO 偏好对
        ↓
DPO 微调 policy
```

## Value 数据构造

Value label 主要来自轨迹进度：

```text
中间步骤：根据当前 step 在整条轨迹中的位置映射到 0 到 100
终止步骤：done 对应任务完成，value = 100
```

最终构造出的 value SFT 数据规模为：

```text
Value SFT samples：713,567
Train samples：641,733
Eval samples：71,834
```

## 实验结果

```text
Value-DPO pairs：1,793
DPO 后严格动作准确率：87.7%
Reward margin：0.2277
```

在线 value selector 评估：

```text
Policy top-1 accuracy：86.9%
Value selector accuracy：84.8%
Selectable samples 上的 value selector accuracy：86.6%
GT candidate coverage：91.2%
```

该结果说明 value function 能提供一定偏好信号，但直接用它在线选择动作时并没有超过原始 policy top-1。主要原因是轨迹进度标签较粗，很多相邻状态的分数差异不足，导致候选动作之间的偏好区分不够强。

## 主要脚本

方法特有脚本：

```text
scripts/build_value_function_data.py
scripts/train_value_function_lora.py
scripts/generate_value_function_pairs.py
scripts/evaluate_value_selector.py
```

该方法使用的共用脚本：

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

## 示例运行

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

原始数据、处理后的 JSONL、模型权重、LoRA adapter、日志和评估结果不包含在本仓库中。
