import argparse
import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, Iterable, Set, Tuple


INPUT_FILES = {
    "policy": "./data/processed/wap_qwen_policy_sft.jsonl",
    "worldmodel": "./data/processed/wap_qwen_worldmodel_sft.jsonl",
    "mix": "./data/processed/wap_qwen_policy_worldmodel_sft.jsonl",
}

OUTPUT_FILES = {
    "policy": (
        "./data/processed/wap_qwen_policy_train.jsonl",
        "./data/processed/wap_qwen_policy_eval.jsonl",
    ),
    "worldmodel": (
        "./data/processed/wap_qwen_worldmodel_train.jsonl",
        "./data/processed/wap_qwen_worldmodel_eval.jsonl",
    ),
    "mix": (
        "./data/processed/wap_qwen_policy_worldmodel_train.jsonl",
        "./data/processed/wap_qwen_policy_worldmodel_eval.jsonl",
    ),
}


def trajectory_key(item: Dict) -> Tuple[str, int]:
    return (
        item.get("split", "unknown"),
        int(item.get("trajectory_index", -1)),
    )


def iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def collect_trajectories(path: str):
    by_split = defaultdict(set)
    for item in iter_jsonl(path):
        split, traj_idx = trajectory_key(item)
        by_split[split].add(traj_idx)
    return by_split


def build_eval_keys(eval_ratio: float, seed: int) -> Set[Tuple[str, int]]:
    rng = random.Random(seed)
    by_split = collect_trajectories(INPUT_FILES["policy"])
    eval_keys = set()

    for split, traj_indices in sorted(by_split.items()):
        traj_indices = sorted(traj_indices)
        rng.shuffle(traj_indices)
        eval_count = max(1, round(len(traj_indices) * eval_ratio))
        for traj_idx in traj_indices[:eval_count]:
            eval_keys.add((split, traj_idx))

        print(
            f"split={split}: trajectories={len(traj_indices)}, "
            f"eval={eval_count}, train={len(traj_indices) - eval_count}"
        )

    return eval_keys


def split_file(name: str, eval_keys: Set[Tuple[str, int]]):
    input_path = INPUT_FILES[name]
    train_path, eval_path = OUTPUT_FILES[name]
    counts = Counter()
    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    with open(train_path, "w", encoding="utf-8") as train_f, open(
        eval_path,
        "w",
        encoding="utf-8",
    ) as eval_f:
        for item in iter_jsonl(input_path):
            if trajectory_key(item) in eval_keys:
                eval_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                counts["eval"] += 1
            else:
                train_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                counts["train"] += 1

    print(
        f"{name}: train={counts['train']} -> {train_path}, "
        f"eval={counts['eval']} -> {eval_path}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.eval_ratio < 1:
        raise ValueError("--eval_ratio must be between 0 and 1")

    eval_keys = build_eval_keys(args.eval_ratio, args.seed)
    print(f"total eval trajectories: {len(eval_keys)}")

    for name in ["policy", "worldmodel", "mix"]:
        split_file(name, eval_keys)


if __name__ == "__main__":
    main()
