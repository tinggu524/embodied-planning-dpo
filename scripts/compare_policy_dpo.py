import argparse
import copy
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from wap_sampling import balanced_reservoir_sample
from image_utils import open_rgb_image


BASE_MODEL_PATH = "/root/Qwen2.5-VL-3B-Instruct"
POLICY_LORA_PATH = "./models/qwen2_5_vl_3b_wap_policy_lora"
DPO_LORA_PATH = "./models/qwen2_5_vl_3b_wap_policy_dpo_lora"
POLICY_DATA_PATH = "./data/processed/wap_qwen_policy_eval.jsonl"
RESULTS_PATH = "./results/policy_dpo_compare_results.json"

MAX_EVAL = 300
EVAL_SEED = 2028
MAX_NEW_TOKENS = 256
IMAGE_SIZE = 224
SPLIT_GROUPS = [("reference",), ("spatial",), ("symbolic",), ("visual",)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", default=BASE_MODEL_PATH)
    parser.add_argument("--policy_lora_path", default=POLICY_LORA_PATH)
    parser.add_argument("--dpo_lora_path", default=DPO_LORA_PATH)
    parser.add_argument("--policy_data_path", default=POLICY_DATA_PATH)
    parser.add_argument("--results_path", default=RESULTS_PATH)
    parser.add_argument("--max_eval", type=int, default=MAX_EVAL)
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--num_eval_shards", type=int, default=1)
    parser.add_argument("--eval_shard_index", type=int, default=0)
    parser.add_argument(
        "--dpo_prompt_mode",
        choices=["reasoning"],
        default="reasoning",
        help="Use reasoning prompt/output parsing for the DPO policy.",
    )
    parser.add_argument(
        "--skip_reasoning_policy",
        action="store_true",
        help="Skip the reasoning SFT baseline and evaluate only the reasoning DPO policy.",
    )
    return parser.parse_args()


def sample_key(item: Dict) -> Tuple[str, int, int]:
    return (
        item.get("split", ""),
        int(item.get("trajectory_index", -1)),
        int(item.get("step_index", -1)),
    )


def normalize_action(action: Optional[str]) -> str:
    if action is None:
        return ""
    action = action.strip().lower()
    action = re.sub(r"\s+", " ", action)
    return action.strip(" .")


def extract_action(text: str, allow_raw: bool = False) -> Optional[str]:
    text = text.split("<|im_end|>")[0].strip()
    if "Action:" in text:
        action_text = text.split("Action:", 1)[1].strip()
    elif allow_raw:
        action_text = text
    else:
        return None
    if not action_text:
        return None
    return action_text.splitlines()[0].strip()


def iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def load_eval_samples(path: str, max_eval: int) -> List[Dict]:
    return balanced_reservoir_sample(
        jsonl_path=path,
        max_samples=max_eval,
        seed=EVAL_SEED,
        group_fields=["split"],
        groups=SPLIT_GROUPS,
    )


def load_lora_model(base_model_path: str, lora_path: str):
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()
    return model


def input_device(model):
    return next(model.parameters()).device


def build_prompt(processor, item: Dict, action_only: bool) -> str:
    user_message = item["messages"][0]
    content = copy.deepcopy(user_message["content"])
    if action_only:
        original_text = content[-1]["text"]
        instruction_match = re.search(r"(?m)^Instruction:\s*(.*)$", original_text)
        history_match = re.search(r"(?m)^Previous high-level actions:\s*(.*)$", original_text)
        instruction = instruction_match.group(1).strip() if instruction_match else ""
        history = history_match.group(1).strip() if history_match else "None"
        content[-1]["text"] = (
            "Task type: POLICY_ACTION_ONLY\n"
            f"Instruction: {instruction}\n"
            f"Previous high-level actions: {history}\n"
            "Current observation is provided as the image.\n"
            "Return only the next high-level action. Do not explain.\n"
            "Action:"
        )

    return processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_text(
    model,
    processor,
    item: Dict,
    action_only: bool,
    image_size: int,
    max_new_tokens: int,
) -> str:
    image_path = item["messages"][0]["content"][0]["image"]
    image = open_rgb_image(image_path, image_size)
    prompt = build_prompt(processor, item, action_only=action_only)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(input_device(model))

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
    output = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return output.split("<|im_end|>")[0].strip()


def generate_action(
    model,
    processor,
    item: Dict,
    action_only: bool,
    image_size: int,
) -> Tuple[str, str]:
    output = generate_text(
        model=model,
        processor=processor,
        item=item,
        action_only=action_only,
        image_size=image_size,
        max_new_tokens=32 if action_only else MAX_NEW_TOKENS,
    )
    action = normalize_action(extract_action(output, allow_raw=action_only))
    if action or action_only:
        return action, output

    fallback = generate_text(
        model=model,
        processor=processor,
        item=item,
        action_only=True,
        image_size=image_size,
        max_new_tokens=32,
    )
    return (
        normalize_action(extract_action(fallback, allow_raw=True)),
        output + "\n\n[FALLBACK]\n" + fallback,
    )


def accuracy(records: List[Dict], key: str) -> float:
    if not records:
        return 0.0
    return sum(record[key] == record["gt"] for record in records) / len(records)


def main():
    args = parse_args()
    if args.num_eval_shards < 1:
        raise ValueError("--num_eval_shards must be >= 1")
    if not 0 <= args.eval_shard_index < args.num_eval_shards:
        raise ValueError("--eval_shard_index must satisfy 0 <= eval_shard_index < num_eval_shards")
    if not os.path.exists(args.policy_data_path):
        raise FileNotFoundError(
            f"{args.policy_data_path} does not exist. Run split_wap_train_eval.py first."
        )

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

    processor = AutoProcessor.from_pretrained(args.base_model_path, trust_remote_code=True)

    records = [
        {
            "key": sample_key(item),
            "image": item["messages"][0]["content"][0]["image"],
            "gt": normalize_action(extract_action(item["messages"][1]["content"])),
        }
        for item in eval_samples
    ]

    if not args.skip_reasoning_policy:
        print("Running reasoning SFT policy direct...")
        policy_model = load_lora_model(args.base_model_path, args.policy_lora_path)
        for idx, item in enumerate(tqdm(eval_samples)):
            policy_action, policy_output = generate_action(
                policy_model,
                processor,
                item,
                action_only=False,
                image_size=args.image_size,
            )
            records[idx]["policy"] = policy_action
            records[idx]["policy_output"] = policy_output
            records[idx]["reasoning_policy"] = policy_action
            records[idx]["reasoning_policy_output"] = policy_output
        del policy_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Running reasoning DPO policy direct...")
    dpo_model = load_lora_model(args.base_model_path, args.dpo_lora_path)
    for idx, item in enumerate(tqdm(eval_samples)):
        dpo_action, dpo_output = generate_action(
            dpo_model,
            processor,
            item,
            action_only=False,
            image_size=args.image_size,
        )
        records[idx]["policy_dpo"] = dpo_action
        records[idx]["policy_dpo_output"] = dpo_output
    del dpo_model

    metrics = {"evaluated": len(records)}
    if not args.skip_reasoning_policy:
        metrics["reasoning_policy_direct_accuracy"] = accuracy(records, "reasoning_policy")
    metrics["reasoning_dpo_direct_accuracy"] = accuracy(records, "policy_dpo")
    metrics["action_match_mode"] = "exact"

    print("=" * 80)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

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
                        "num_eval_shards": args.num_eval_shards,
                        "eval_shard_index": args.eval_shard_index,
                        "policy_lora_path": args.policy_lora_path,
                        "dpo_lora_path": args.dpo_lora_path,
                        "dpo_prompt_mode": args.dpo_prompt_mode,
                        "action_match_mode": "exact",
                        "skip_reasoning_policy": args.skip_reasoning_policy,
                    },
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Saved DPO compare results to {args.results_path}")


if __name__ == "__main__":
    main()
