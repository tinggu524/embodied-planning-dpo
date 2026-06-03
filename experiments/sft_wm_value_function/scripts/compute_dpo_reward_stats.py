import argparse
import json
import os
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

from train_policy_dpo import (
    BASE_MODEL_PATH,
    IMAGE_SIZE,
    POLICY_LORA_PATH,
    PolicyDPODataset,
    collate_dpo_features,
    load_policy_model,
    optional_int,
    sequence_logps,
)


DATA_PATH = "./data/processed/wap_qwen_policy_reasoning_dpo_wm_pairs_4k_margin_gt_0p05.jsonl"
DPO_LORA_PATH = "./models/qwen2_5_vl_3b_wap_policy_reasoning_dpo_4k_m005_lora"
OUTPUT_PATH = "./results/dpo_reward_stats.json"
DPO_BETA = 0.05


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", default=BASE_MODEL_PATH)
    parser.add_argument("--ref_lora_path", default=POLICY_LORA_PATH)
    parser.add_argument("--dpo_lora_path", default=DPO_LORA_PATH)
    parser.add_argument("--data_path", default=DATA_PATH)
    parser.add_argument("--output_path", default=OUTPUT_PATH)
    parser.add_argument("--beta", type=float, default=DPO_BETA)
    parser.add_argument("--max_samples", type=optional_int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--max_response_tokens", type=optional_int, default=256)
    return parser.parse_args()


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


@torch.no_grad()
def collect_model_logps(model, dataloader: DataLoader) -> Tuple[List[float], List[float]]:
    model.eval()
    device = next(model.parameters()).device
    chosen_logps = []
    rejected_logps = []

    for batch in tqdm(dataloader):
        pair_count = int(batch.pop("pair_count").item())
        labels = batch.pop("labels")
        batch = move_batch_to_device(batch, device)
        labels = labels.to(device)

        outputs = model(**batch)
        logps = sequence_logps(outputs.logits, labels)
        chosen_logps.extend(logps[:pair_count].float().cpu().tolist())
        rejected_logps.extend(logps[pair_count:].float().cpu().tolist())

    return chosen_logps, rejected_logps


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main():
    args = parse_args()
    processor = AutoProcessor.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
    )
    processor.tokenizer.padding_side = "left"

    dataset = PolicyDPODataset(
        args.data_path,
        processor,
        max_samples=args.max_samples,
        image_size=args.image_size,
        max_response_tokens=args.max_response_tokens,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_dpo_features,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Collecting DPO policy log probabilities...")
    dpo_model = load_policy_model(args.base_model_path, args.dpo_lora_path, trainable=False)
    dpo_model.to(device)
    dpo_chosen, dpo_rejected = collect_model_logps(dpo_model, dataloader)
    del dpo_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Collecting reference policy log probabilities...")
    ref_model = load_policy_model(args.base_model_path, args.ref_lora_path, trainable=False)
    ref_model.to(device)
    ref_chosen, ref_rejected = collect_model_logps(ref_model, dataloader)
    del ref_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    reward_chosen = [
        args.beta * (policy_logp - ref_logp)
        for policy_logp, ref_logp in zip(dpo_chosen, ref_chosen)
    ]
    reward_rejected = [
        args.beta * (policy_logp - ref_logp)
        for policy_logp, ref_logp in zip(dpo_rejected, ref_rejected)
    ]
    reward_margin = [
        chosen - rejected
        for chosen, rejected in zip(reward_chosen, reward_rejected)
    ]

    stats = {
        "num_pairs": len(reward_margin),
        "beta": args.beta,
        "reward_chosen_mean": mean(reward_chosen),
        "reward_rejected_mean": mean(reward_rejected),
        "reward_margin_mean": mean(reward_margin),
        "policy_chosen_logp_mean": mean(dpo_chosen),
        "policy_rejected_logp_mean": mean(dpo_rejected),
        "ref_chosen_logp_mean": mean(ref_chosen),
        "ref_rejected_logp_mean": mean(ref_rejected),
        "config": {
            "base_model_path": args.base_model_path,
            "ref_lora_path": args.ref_lora_path,
            "dpo_lora_path": args.dpo_lora_path,
            "data_path": args.data_path,
            "max_samples": args.max_samples,
            "max_response_tokens": args.max_response_tokens,
        },
    }

    print("=" * 80)
    for key, value in stats.items():
        if key == "config":
            continue
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"Saved reward stats to {args.output_path}")


if __name__ == "__main__":
    main()
