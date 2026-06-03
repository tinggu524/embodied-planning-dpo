import argparse
import copy
import json
import os
import re
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

from generate_policy_dpo_pairs import (
    extract_action,
    generate_candidate_actions,
    generate_reasoning_for_action,
    input_device,
    is_near_duplicate_action,
    load_lora_model,
    make_action_only_prompt_messages,
    make_reasoning_prompt_messages,
    normalize_action,
    response_action_matches,
)
from image_utils import open_rgb_image
from wap_sampling import balanced_reservoir_sample


BASE_MODEL_PATH = "/root/Qwen2.5-VL-3B-Instruct"
POLICY_LORA_PATH = "./models/qwen2_5_vl_3b_wap_policy_lora"
WORLD_MODEL_LORA_PATH = "./models/qwen2_5_vl_3b_wap_worldmodel_lora"
VALUE_MODEL_LORA_PATH = "./models/qwen2_5_vl_3b_wap_value_function_lora"
POLICY_DATA_PATH = "./data/processed/wap_qwen_policy_train.jsonl"
OUTPUT_PATH = "./data/processed/wap_qwen_policy_value_dpo_pairs.jsonl"

MAX_SAMPLES = 20000
EVAL_SEED = 2031
NUM_CANDIDATES = 5
IMAGE_SIZE = 224
WORLD_MODEL_MAX_NEW_TOKENS = 192
VALUE_MAX_NEW_TOKENS = 16
REASONING_MAX_NEW_TOKENS = 256
ACTION_MAX_NEW_TOKENS = 32
MIN_VALUE_MARGIN = 10.0
SPLIT_GROUPS = [("reference",), ("spatial",), ("symbolic",), ("visual",)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", default=BASE_MODEL_PATH)
    parser.add_argument("--policy_lora_path", default=POLICY_LORA_PATH)
    parser.add_argument("--world_model_lora_path", default=WORLD_MODEL_LORA_PATH)
    parser.add_argument("--value_model_lora_path", default=VALUE_MODEL_LORA_PATH)
    parser.add_argument("--policy_data_path", default=POLICY_DATA_PATH)
    parser.add_argument("--output_path", default=OUTPUT_PATH)
    parser.add_argument("--max_samples", type=int, default=MAX_SAMPLES)
    parser.add_argument("--num_candidates", type=int, default=NUM_CANDIDATES)
    parser.add_argument("--min_value_margin", type=float, default=MIN_VALUE_MARGIN)
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
    parser.add_argument(
        "--dpo_response_mode",
        choices=["action_only", "reasoning"],
        default="action_only",
    )
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--debug_skips_path", default="")
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


def extract_field(text: str, field: str, default: str = "") -> str:
    match = re.search(rf"(?m)^{re.escape(field)}:\s*(.*)$", text)
    return match.group(1).strip() if match else default


def instruction_and_history(policy_item: Dict) -> Tuple[str, str]:
    text = policy_item["messages"][0]["content"][-1]["text"]
    return (
        extract_field(text, "Instruction"),
        extract_field(text, "Previous high-level actions", "None"),
    )


def load_policy_samples(path: str, max_samples: int) -> List[Dict]:
    return balanced_reservoir_sample(
        jsonl_path=path,
        max_samples=max_samples,
        seed=EVAL_SEED,
        group_fields=["split"],
        groups=SPLIT_GROUPS,
    )


def build_world_model_prompt_messages(policy_item: Dict, action: str) -> List[Dict]:
    instruction, history = instruction_and_history(policy_item)
    content = copy.deepcopy(policy_item["messages"][0]["content"])
    content[-1]["text"] = (
        "Task type: WORLD_MODEL\n"
        f"Instruction: {instruction}\n"
        f"Previous high-level actions: {history}\n"
        "Current observation is provided as the image.\n"
        f"Action to execute: {action}\n"
        "Predict the semantic world state after this high-level action is executed. "
        "Do not output joint angles or low-level controls.\n"
        "Output format:\n"
        "Predicted state: ...\n"
        "Likely next action: ...\n"
        "Progress: in_progress/done"
    )
    return [{"role": "user", "content": content}]


def extract_predicted_state(world_model_output: str) -> str:
    text = world_model_output.split("<|im_end|>")[0].strip()
    if "Predicted state:" in text:
        text = text.split("Predicted state:", 1)[1].strip()
    stop_positions = []
    for label in ["Likely next action:", "Progress:"]:
        pos = text.find(label)
        if pos >= 0:
            stop_positions.append(pos)
    if stop_positions:
        text = text[: min(stop_positions)].strip()
    return text


def build_value_prompt_messages(
    policy_item: Dict,
    action: str,
    predicted_state: str,
    retry: bool = False,
    previous_output: str = "",
) -> List[Dict]:
    instruction, history = instruction_and_history(policy_item)
    retry_text = ""
    if retry:
        retry_text = (
            "\n\nYour previous answer could not be parsed as a numeric value:\n"
            f"{previous_output}\n"
            "Return exactly one line now. Do not explain."
        )
    content = copy.deepcopy(policy_item["messages"][0]["content"])
    content[-1]["text"] = (
        "Task type: VALUE_FUNCTION\n"
        f"Instruction: {instruction}\n"
        f"Previous high-level actions: {history}\n"
        "Current observation is provided as the image.\n"
        f"Action executed: {action}\n"
        "Predicted semantic state after action:\n"
        f"{predicted_state}\n\n"
        "Score how close this predicted post-action state is to completing the instruction. "
        "Use only task progress and do not rely on any hidden reference action.\n"
        "Output format:\n"
        "Value: <integer from 0 to 100>\n"
        "Return exactly one line and no explanation."
        f"{retry_text}"
    )
    return [{"role": "user", "content": content}]


def generate_from_messages(
    model,
    processor,
    prompt_messages: List[Dict],
    image_size: int,
    max_new_tokens: int,
) -> str:
    image_path = prompt_messages[0]["content"][0]["image"]
    image = open_rgb_image(image_path, image_size)
    prompt = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
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


def parse_value(value_output: str) -> Optional[float]:
    text = value_output.split("<|im_end|>")[0]
    match = re.search(r"Value\s*[:：]\s*(-?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    value = float(match.group(1) if match.lastindex else match.group(0))
    return max(0.0, min(100.0, value))


def score_action(
    world_model,
    value_model,
    processor,
    policy_item: Dict,
    action: str,
    image_size: int,
    world_model_max_new_tokens: int,
    value_max_new_tokens: int,
    scoring_failures: Optional[List[Dict]] = None,
) -> Optional[Dict]:
    world_output = generate_from_messages(
        model=world_model,
        processor=processor,
        prompt_messages=build_world_model_prompt_messages(policy_item, action),
        image_size=image_size,
        max_new_tokens=world_model_max_new_tokens,
    )
    predicted_state = extract_predicted_state(world_output)
    if not predicted_state:
        if scoring_failures is not None:
            scoring_failures.append(
                {
                    "action": action,
                    "reason": "missing_predicted_state",
                    "world_model_output": world_output,
                }
            )
        return None

    value_output = generate_from_messages(
        model=value_model,
        processor=processor,
        prompt_messages=build_value_prompt_messages(policy_item, action, predicted_state),
        image_size=image_size,
        max_new_tokens=value_max_new_tokens,
    )
    value = parse_value(value_output)
    if value is None:
        retry_output = generate_from_messages(
            model=value_model,
            processor=processor,
            prompt_messages=build_value_prompt_messages(
                policy_item,
                action,
                predicted_state,
                retry=True,
                previous_output=value_output,
            ),
            image_size=image_size,
            max_new_tokens=max(value_max_new_tokens, 32),
        )
        retry_value = parse_value(retry_output)
        if retry_value is not None:
            value_output = value_output + "\n\n[VALUE_RETRY]\n" + retry_output
            value = retry_value
        else:
            if scoring_failures is not None:
                scoring_failures.append(
                    {
                        "action": action,
                        "reason": "missing_numeric_value",
                        "predicted_state": predicted_state,
                        "world_model_output": world_output,
                        "value_output": value_output,
                        "retry_value_output": retry_output,
                    }
                )
            return None
    return {
        "action": action,
        "value": value,
        "world_model_output": world_output,
        "predicted_state": predicted_state,
        "value_output": value_output,
    }


def choose_pair(scored_actions: List[Dict], min_margin: float) -> Optional[Tuple[Dict, Dict, float]]:
    scored_actions = sorted(scored_actions, key=lambda item: item["value"], reverse=True)
    for chosen in scored_actions:
        for rejected in reversed(scored_actions):
            if is_near_duplicate_action(chosen["action"], rejected["action"]):
                continue
            margin = chosen["value"] - rejected["value"]
            if margin >= min_margin:
                return chosen, rejected, margin
    return None


def build_dpo_record(
    policy_item: Dict,
    chosen_action: str,
    rejected_action: str,
    scored_actions: List[Dict],
    value_margin: float,
    dpo_response_mode: str,
    chosen_response: Optional[str] = None,
    rejected_response: Optional[str] = None,
) -> Dict:
    if dpo_response_mode == "reasoning":
        prompt_messages = make_reasoning_prompt_messages(policy_item)
        chosen = chosen_response or f"Action: {chosen_action}"
        rejected = rejected_response or f"Action: {rejected_action}"
        task_type = "policy_reasoning_value_dpo"
    else:
        prompt_messages = make_action_only_prompt_messages(policy_item)
        chosen = f"Action: {chosen_action}"
        rejected = f"Action: {rejected_action}"
        task_type = "policy_value_dpo"

    return {
        "task_type": task_type,
        "split": policy_item.get("split"),
        "trajectory_index": policy_item.get("trajectory_index"),
        "step_index": policy_item.get("step_index"),
        "prompt_messages": prompt_messages,
        "chosen": chosen.strip(),
        "rejected": rejected.strip(),
        "metadata": {
            "dpo_response_mode": dpo_response_mode,
            "chosen_action": chosen_action,
            "rejected_action": rejected_action,
            "value_margin": value_margin,
            "candidate_scores": [
                {
                    "action": item["action"],
                    "value": item["value"],
                    "predicted_state": item["predicted_state"],
                    "value_output": item["value_output"],
                }
                for item in sorted(scored_actions, key=lambda record: record["value"], reverse=True)
            ],
            "negative_source": "policy_candidate_selected_by_value_function_margin",
        },
    }


def build_skip_record(policy_item: Dict, reason: str, extra: Optional[Dict] = None) -> Dict:
    record = {
        "reason": reason,
        "split": policy_item.get("split"),
        "trajectory_index": policy_item.get("trajectory_index"),
        "step_index": policy_item.get("step_index"),
    }
    if extra:
        record.update(extra)
    return record


def main():
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")
    if not os.path.exists(args.policy_data_path):
        raise FileNotFoundError(args.policy_data_path)

    policy_samples = load_policy_samples(args.policy_data_path, args.max_samples)
    if args.num_shards > 1:
        policy_samples = [
            item
            for idx, item in enumerate(policy_samples)
            if idx % args.num_shards == args.shard_index
        ]
    if not policy_samples:
        raise RuntimeError("No policy samples loaded.")

    processor = AutoProcessor.from_pretrained(args.base_model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    policy_model = load_lora_model(args.base_model_path, args.policy_lora_path)
    world_model = load_lora_model(args.base_model_path, args.world_model_lora_path)
    value_model = load_lora_model(args.base_model_path, args.value_model_lora_path)
    print(
        f"Processing samples: {len(policy_samples)} "
        f"(shard {args.shard_index + 1}/{args.num_shards})"
    )

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    if args.debug_skips_path:
        os.makedirs(os.path.dirname(args.debug_skips_path) or ".", exist_ok=True)
    debug_f = open(args.debug_skips_path, "w", encoding="utf-8") if args.debug_skips_path else None

    kept = 0
    skipped = 0
    with open(args.output_path, "w", encoding="utf-8") as out_f:
        for policy_item in tqdm(policy_samples):
            candidates, raw_outputs, _ = generate_candidate_actions(
                model=policy_model,
                processor=processor,
                policy_item=policy_item,
                num_candidates=args.num_candidates,
                image_size=args.image_size,
                prompt_mode=args.candidate_prompt_mode,
                reasoning_max_new_tokens=args.reasoning_max_new_tokens,
                action_max_new_tokens=args.action_max_new_tokens,
            )
            if len(candidates) < 2:
                skipped += 1
                if debug_f:
                    debug_f.write(
                        json.dumps(
                            build_skip_record(
                                policy_item,
                                "too_few_candidates",
                                {"candidates": candidates, "raw_outputs": raw_outputs},
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                continue

            scored_actions = []
            scoring_failures = []
            for action in candidates:
                scored = score_action(
                    world_model=world_model,
                    value_model=value_model,
                    processor=processor,
                    policy_item=policy_item,
                    action=action,
                    image_size=args.image_size,
                    world_model_max_new_tokens=args.world_model_max_new_tokens,
                    value_max_new_tokens=args.value_max_new_tokens,
                    scoring_failures=scoring_failures,
                )
                if scored is not None:
                    scored_actions.append(scored)

            if len(scored_actions) < 2:
                skipped += 1
                if debug_f:
                    debug_f.write(
                        json.dumps(
                            build_skip_record(
                                policy_item,
                                "too_few_scored_candidates",
                                {
                                    "candidates": candidates,
                                    "scored_actions": [
                                        {"action": item["action"], "value": item["value"]}
                                        for item in scored_actions
                                    ],
                                    "scoring_failures": scoring_failures,
                                },
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                continue

            selected = choose_pair(scored_actions, args.min_value_margin)
            if selected is None:
                skipped += 1
                if debug_f:
                    debug_f.write(
                        json.dumps(
                            build_skip_record(
                                policy_item,
                                "value_margin_below_threshold",
                                {
                                    "scores": [
                                        {"action": item["action"], "value": item["value"]}
                                        for item in scored_actions
                                    ]
                                },
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                continue

            chosen, rejected, value_margin = selected
            chosen_response = None
            rejected_response = None
            if args.dpo_response_mode == "reasoning":
                chosen_response = generate_reasoning_for_action(
                    policy_model,
                    processor,
                    policy_item,
                    chosen["action"],
                    args.image_size,
                    args.reasoning_max_new_tokens,
                )
                rejected_response = generate_reasoning_for_action(
                    policy_model,
                    processor,
                    policy_item,
                    rejected["action"],
                    args.image_size,
                    args.reasoning_max_new_tokens,
                )
                if not response_action_matches(chosen_response, chosen["action"]):
                    skipped += 1
                    if debug_f:
                        debug_f.write(
                            json.dumps(
                                build_skip_record(policy_item, "chosen_reasoning_action_mismatch"),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    continue
                if not response_action_matches(rejected_response, rejected["action"]):
                    skipped += 1
                    if debug_f:
                        debug_f.write(
                            json.dumps(
                                build_skip_record(policy_item, "rejected_reasoning_action_mismatch"),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    continue

            record = build_dpo_record(
                policy_item=policy_item,
                chosen_action=chosen["action"],
                rejected_action=rejected["action"],
                scored_actions=scored_actions,
                value_margin=value_margin,
                dpo_response_mode=args.dpo_response_mode,
                chosen_response=chosen_response,
                rejected_response=rejected_response,
            )
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    if debug_f:
        debug_f.close()
    print(f"Saved value-DPO pairs: {kept} -> {args.output_path}")
    print(f"Skipped samples: {skipped}")
    if args.debug_skips_path:
        print(f"Saved skipped-sample debug data to {args.debug_skips_path}")


if __name__ == "__main__":
    main()
