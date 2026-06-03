import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoProcessor

ROOT_DIR = Path(__file__).resolve().parents[3]
SHARED_SCRIPTS = ROOT_DIR / "shared" / "scripts"
SUCCESSOR_STATE_SCRIPTS = ROOT_DIR / "methods" / "successor_state_dpo" / "scripts"
for scripts_dir in (SHARED_SCRIPTS, SUCCESSOR_STATE_SCRIPTS):
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

from generate_policy_dpo_pairs import extract_action, generate_candidate_actions, load_lora_model, normalize_action
from generate_value_function_pairs import score_action
from wap_sampling import balanced_reservoir_sample


BASE_MODEL_PATH = "/root/Qwen2.5-VL-3B-Instruct"
POLICY_LORA_PATH = "./models/qwen2_5_vl_3b_wap_policy_lora"
WORLD_MODEL_LORA_PATH = "./models/qwen2_5_vl_3b_wap_worldmodel_lora"
VALUE_MODEL_LORA_PATH = "./models/qwen2_5_vl_3b_wap_value_function_lora"
POLICY_DATA_PATH = "./data/processed/wap_qwen_policy_eval.jsonl"
RESULTS_PATH = "./results/value_selector_eval_results.json"

MAX_EVAL = 1000
EVAL_SEED = 2028
NUM_CANDIDATES = 5
IMAGE_SIZE = 224
WORLD_MODEL_MAX_NEW_TOKENS = 192
VALUE_MAX_NEW_TOKENS = 16
REASONING_MAX_NEW_TOKENS = 256
ACTION_MAX_NEW_TOKENS = 32
SPLIT_GROUPS = [("reference",), ("spatial",), ("symbolic",), ("visual",)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", default=BASE_MODEL_PATH)
    parser.add_argument("--policy_lora_path", default=POLICY_LORA_PATH)
    parser.add_argument("--world_model_lora_path", default=WORLD_MODEL_LORA_PATH)
    parser.add_argument("--value_model_lora_path", default=VALUE_MODEL_LORA_PATH)
    parser.add_argument("--policy_data_path", default=POLICY_DATA_PATH)
    parser.add_argument("--results_path", default=RESULTS_PATH)
    parser.add_argument("--max_eval", type=int, default=MAX_EVAL)
    parser.add_argument("--num_candidates", type=int, default=NUM_CANDIDATES)
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--world_model_max_new_tokens", type=int, default=WORLD_MODEL_MAX_NEW_TOKENS)
    parser.add_argument("--value_max_new_tokens", type=int, default=VALUE_MAX_NEW_TOKENS)
    parser.add_argument("--reasoning_max_new_tokens", type=int, default=REASONING_MAX_NEW_TOKENS)
    parser.add_argument("--action_max_new_tokens", type=int, default=ACTION_MAX_NEW_TOKENS)
    parser.add_argument(
        "--candidate_prompt_mode",
        choices=["action_only", "reasoning"],
        default="action_only",
    )
    parser.add_argument("--num_eval_shards", type=int, default=1)
    parser.add_argument("--eval_shard_index", type=int, default=0)
    return parser.parse_args()


def iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def sample_key(item: Dict) -> Tuple[str, int, int]:
    return (
        item.get("split", ""),
        int(item.get("trajectory_index", -1)),
        int(item.get("step_index", -1)),
    )


def load_eval_samples(path: str, max_eval: int) -> List[Dict]:
    return balanced_reservoir_sample(
        jsonl_path=path,
        max_samples=max_eval,
        seed=EVAL_SEED,
        group_fields=["split"],
        groups=SPLIT_GROUPS,
    )


def exact_accuracy(records: List[Dict], key: str, selected_only: bool = False) -> float:
    rows = [record for record in records if (record.get(key) if selected_only else True)]
    if not rows:
        return 0.0
    return sum(record.get(key) == record["gt"] for record in rows) / len(rows)


def compute_metrics(records: List[Dict]) -> Dict:
    selected = [record for record in records if record.get("value_selector")]
    gt_in_candidates = [record for record in records if record.get("gt_in_candidates")]
    metrics = {
        "evaluated": len(records),
        "selectable": len(selected),
        "selection_coverage": len(selected) / len(records) if records else 0.0,
        "gt_in_candidates": len(gt_in_candidates),
        "gt_candidate_coverage": len(gt_in_candidates) / len(records) if records else 0.0,
        "policy_top1_accuracy": exact_accuracy(records, "policy_top1"),
        "value_selector_accuracy_all": exact_accuracy(records, "value_selector"),
        "value_selector_accuracy_selected_only": exact_accuracy(
            records,
            "value_selector",
            selected_only=True,
        ),
    }
    selected_values = [
        record["value_selector_score"]
        for record in selected
        if record.get("value_selector_score") is not None
    ]
    if selected_values:
        metrics["value_selector_score_mean"] = sum(selected_values) / len(selected_values)
    return metrics


def print_metrics(metrics: Dict):
    print("=" * 80)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


def main():
    args = parse_args()
    if args.num_eval_shards < 1:
        raise ValueError("--num_eval_shards must be >= 1")
    if not 0 <= args.eval_shard_index < args.num_eval_shards:
        raise ValueError("--eval_shard_index must satisfy 0 <= eval_shard_index < num_eval_shards")
    if not os.path.exists(args.policy_data_path):
        raise FileNotFoundError(args.policy_data_path)

    eval_samples = load_eval_samples(args.policy_data_path, args.max_eval)
    if args.num_eval_shards > 1:
        eval_samples = [
            item
            for idx, item in enumerate(eval_samples)
            if idx % args.num_eval_shards == args.eval_shard_index
        ]
    if not eval_samples:
        raise RuntimeError("No eval samples loaded.")

    print(
        f"Evaluating samples: {len(eval_samples)} "
        f"(shard {args.eval_shard_index + 1}/{args.num_eval_shards})"
    )
    print("Loading processor and LoRA models...")
    processor = AutoProcessor.from_pretrained(args.base_model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    policy_model = load_lora_model(args.base_model_path, args.policy_lora_path)
    world_model = load_lora_model(args.base_model_path, args.world_model_lora_path)
    value_model = load_lora_model(args.base_model_path, args.value_model_lora_path)

    records = []
    for item in tqdm(eval_samples, desc="Value selector eval"):
        gt = normalize_action(extract_action(item["messages"][1]["content"]))
        candidates, raw_outputs, _ = generate_candidate_actions(
            model=policy_model,
            processor=processor,
            policy_item=item,
            num_candidates=args.num_candidates,
            image_size=args.image_size,
            prompt_mode=args.candidate_prompt_mode,
            reasoning_max_new_tokens=args.reasoning_max_new_tokens,
            action_max_new_tokens=args.action_max_new_tokens,
        )

        scoring_failures = []
        scored_actions = []
        for action in candidates:
            scored = score_action(
                world_model=world_model,
                value_model=value_model,
                processor=processor,
                policy_item=item,
                action=action,
                image_size=args.image_size,
                world_model_max_new_tokens=args.world_model_max_new_tokens,
                value_max_new_tokens=args.value_max_new_tokens,
                scoring_failures=scoring_failures,
            )
            if scored is not None:
                scored_actions.append(scored)

        selected: Optional[Dict] = None
        if scored_actions:
            selected = max(scored_actions, key=lambda record: record["value"])

        gt_score = None
        for scored in scored_actions:
            if normalize_action(scored["action"]) == gt:
                gt_score = scored["value"]
                break

        records.append(
            {
                "key": sample_key(item),
                "image": item["messages"][0]["content"][0]["image"],
                "gt": gt,
                "candidates": candidates,
                "policy_top1": candidates[0] if candidates else "",
                "gt_in_candidates": gt in candidates,
                "value_selector": selected["action"] if selected else "",
                "value_selector_score": selected["value"] if selected else None,
                "gt_candidate_score": gt_score,
                "candidate_scores": [
                    {
                        "action": scored["action"],
                        "value": scored["value"],
                        "value_output": scored["value_output"],
                    }
                    for scored in sorted(
                        scored_actions,
                        key=lambda record: record["value"],
                        reverse=True,
                    )
                ],
                "scoring_failures": scoring_failures,
                "raw_policy_outputs": raw_outputs,
            }
        )

    metrics = compute_metrics(records)
    print_metrics(metrics)

    if args.results_path:
        os.makedirs(os.path.dirname(args.results_path) or ".", exist_ok=True)
        with open(args.results_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metrics": metrics,
                    "records": records,
                    "config": {
                        "policy_data_path": args.policy_data_path,
                        "max_eval": args.max_eval,
                        "num_candidates": args.num_candidates,
                        "candidate_prompt_mode": args.candidate_prompt_mode,
                        "num_eval_shards": args.num_eval_shards,
                        "eval_shard_index": args.eval_shard_index,
                        "policy_lora_path": args.policy_lora_path,
                        "world_model_lora_path": args.world_model_lora_path,
                        "value_model_lora_path": args.value_model_lora_path,
                        "action_match_mode": "exact",
                    },
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Saved value selector eval results to {args.results_path}")

    del policy_model, world_model, value_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
