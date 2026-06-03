import argparse
import json
import os
from typing import Dict, List

from compare_policy_dpo import accuracy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_paths", nargs="+", required=True)
    parser.add_argument("--output_path", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    records: List[Dict] = []
    configs = []
    for path in args.input_paths:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        records.extend(payload["records"])
        configs.append(payload.get("config", {}))

    metrics = {"evaluated": len(records)}
    if any("reasoning_policy" in record for record in records):
        metrics["reasoning_policy_direct_accuracy"] = accuracy(
            records,
            "reasoning_policy",
        )
    if any("policy_dpo" in record for record in records):
        metrics["reasoning_dpo_direct_accuracy"] = accuracy(
            records,
            "policy_dpo",
        )
    metrics["action_match_mode"] = "exact"

    print("=" * 80)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "records": records,
                "config": {
                    "input_paths": args.input_paths,
                    "action_match_mode": "exact",
                    "shard_configs": configs,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved merged compare results to {args.output_path}")


if __name__ == "__main__":
    main()
