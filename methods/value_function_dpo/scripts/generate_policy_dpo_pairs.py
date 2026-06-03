import argparse
import copy
import json
import os
import re
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from wap_sampling import balanced_reservoir_sample
from image_utils import open_rgb_image


BASE_MODEL_PATH = "/root/Qwen2.5-VL-3B-Instruct"
POLICY_LORA_PATH = "./models/qwen2_5_vl_3b_wap_policy_lora"
WORLD_MODEL_LORA_PATH = "./models/qwen2_5_vl_3b_wap_worldmodel_lora"

POLICY_DATA_PATH = "./data/processed/wap_qwen_policy_train.jsonl"
WORLD_MODEL_DATA_PATH = "./data/processed/wap_qwen_worldmodel_train.jsonl"
OUTPUT_PATH = "./data/processed/wap_qwen_policy_dpo_wm_pairs.jsonl"

MAX_SAMPLES = 20000
EVAL_SEED = 2027
NUM_CANDIDATES = 5
MAX_NEW_TOKENS = 256
ACTION_MAX_NEW_TOKENS = 32
IMAGE_SIZE = 224
SPLIT_GROUPS = [("reference",), ("spatial",), ("symbolic",), ("visual",)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", default=BASE_MODEL_PATH)
    parser.add_argument("--policy_lora_path", default=POLICY_LORA_PATH)
    parser.add_argument("--world_model_lora_path", default=WORLD_MODEL_LORA_PATH)
    parser.add_argument("--policy_data_path", default=POLICY_DATA_PATH)
    parser.add_argument("--world_model_data_path", default=WORLD_MODEL_DATA_PATH)
    parser.add_argument("--output_path", default=OUTPUT_PATH)
    parser.add_argument("--max_samples", type=int, default=MAX_SAMPLES)
    parser.add_argument("--num_candidates", type=int, default=NUM_CANDIDATES)
    parser.add_argument("--min_margin", type=float, default=0.0)
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--reasoning_max_new_tokens", type=int, default=MAX_NEW_TOKENS)
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
        help=(
            "Use action-only chosen/rejected responses, or keep full reasoning+action "
            "responses for DPO pairs."
        ),
    )
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--debug_skips_path", default="")
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


ACTION_VERBS = [
    "pick up",
    "put down",
    "turn on",
    "turn off",
    "open",
    "close",
    "find",
    "slice",
]


OBJECT_NORMALIZATION = {
    "butter knife": "knife",
    "remote control": "remote",
    "arm chair": "chair",
    "counter top": "countertop",
    "garbage can": "trash can",
}

OBJECT_STOPWORDS = {
    "a",
    "an",
    "the",
    "some",
    "any",
    "one",
    "of",
    "to",
    "near",
    "beside",
    "inside",
    "outside",
    "left",
    "right",
    "top",
    "bottom",
    "upper",
    "lower",
}


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


def singularize_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def normalize_object_text(text: str) -> str:
    text = normalize_action(text)
    for phrase, replacement in OBJECT_NORMALIZATION.items():
        text = re.sub(rf"\b{re.escape(phrase)}\b", replacement, text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    tokens = [
        singularize_token(token)
        for token in text.split()
        if token and token not in OBJECT_STOPWORDS
    ]
    return " ".join(tokens)


def parse_action_slot(action: str) -> Tuple[str, str, Tuple[str, ...], str]:
    action = normalize_action(action)
    verb = ""
    raw_object = action
    for verb in ACTION_VERBS:
        if action == verb:
            raw_object = ""
            break
        if action.startswith(verb + " "):
            raw_object = action[len(verb) + 1 :].strip()
            break
    else:
        verb = ""

    obj = normalize_object_text(raw_object)
    tokens = tuple(obj.split())
    head = tokens[-1] if tokens else ""
    return verb, obj, tokens, head


def action_object(action: str) -> str:
    return parse_action_slot(action)[1]


def action_verb(action: str) -> str:
    return parse_action_slot(action)[0]


def object_slots_are_similar(first: str, second: str) -> bool:
    _, first_obj, first_tokens, first_head = parse_action_slot(first)
    _, second_obj, second_tokens, second_head = parse_action_slot(second)
    if not first_obj or not second_obj:
        return False
    if first_obj == second_obj:
        return True

    first_set = set(first_tokens)
    second_set = set(second_tokens)
    if first_set and second_set:
        if first_set <= second_set or second_set <= first_set:
            return True
        overlap = len(first_set & second_set) / min(len(first_set), len(second_set))
        if overlap >= 0.75:
            return True

    if first_head and second_head and first_head == second_head:
        return True

    char_similarity = SequenceMatcher(None, first_obj, second_obj).ratio()
    return char_similarity >= 0.88


def is_near_duplicate_action(first: str, second: str) -> bool:
    first = normalize_action(first)
    second = normalize_action(second)
    if first == second:
        return True
    return object_slots_are_similar(first, second)


def is_single_step_action(action: str) -> bool:
    action = action.strip()
    if not action:
        return False
    lowered = action.lower()
    blocked_fragments = [
        "->",
        "=>",
        "\n",
        "reasoning:",
        "action:",
        "step ",
        " then ",
        ";",
    ]
    if any(fragment in lowered for fragment in blocked_fragments):
        return False
    if len(action.split()) > 8:
        return False
    return True


def response_action_matches(response: str, action: str) -> bool:
    parsed = normalize_action(extract_action(response, allow_raw=False))
    return parsed == normalize_action(action)


def iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def require_file(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} does not exist. Run split_wap_train_eval.py first if this is a train/eval file."
        )


def load_policy_samples(path: str, max_samples: int) -> List[Dict]:
    return balanced_reservoir_sample(
        jsonl_path=path,
        max_samples=max_samples,
        seed=EVAL_SEED,
        group_fields=["split"],
        groups=SPLIT_GROUPS,
    )


def load_world_model_matches(path: str, needed_keys: set) -> Dict[Tuple[str, int, int], Dict]:
    matches = {}
    for item in iter_jsonl(path):
        key = sample_key(item)
        if key in needed_keys:
            matches[key] = item
        if len(matches) == len(needed_keys):
            break
    return matches


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


def build_policy_prompt(processor, policy_item: Dict, action_only: bool = False) -> str:
    user_message = policy_item["messages"][0]
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


def make_action_only_prompt_messages(policy_item: Dict) -> List[Dict]:
    user_message = policy_item["messages"][0]
    content = copy.deepcopy(user_message["content"])
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
    return [{"role": "user", "content": content}]


def make_reasoning_prompt_messages(policy_item: Dict) -> List[Dict]:
    user_message = policy_item["messages"][0]
    return [{"role": "user", "content": copy.deepcopy(user_message["content"])}]


def build_forced_reasoning_prompt(processor, policy_item: Dict, action: str) -> str:
    user_message = policy_item["messages"][0]
    content = copy.deepcopy(user_message["content"])
    content[-1]["text"] = (
        content[-1]["text"].rstrip()
        + "\n\n"
        + "Write a concise reasoning trace for the next high-level action below. "
        + "End the response with exactly the requested action.\n"
        + f"Requested next high-level action: {action}\n"
        + "Output format:\n"
        + "Reasoning: ...\n"
        + f"Action: {action}"
    )
    return processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def clean_generated_response(text: str) -> str:
    return text.split("<|im_end|>")[0].strip()


def force_response_action(response: str, action: str) -> str:
    response = clean_generated_response(response)
    if "Action:" in response:
        prefix = response.split("Action:", 1)[0].rstrip()
        return f"{prefix}\nAction: {action}".strip()
    if response:
        return f"{response.rstrip()}\nAction: {action}"
    return f"Reasoning: The selected action is the next high-level step.\nAction: {action}"


def generate_from_prompt(
    model,
    processor,
    policy_item: Dict,
    prompt: str,
    sample: bool,
    num_candidates: int,
    image_size: int,
    max_new_tokens: int,
) -> List[str]:
    user_message = policy_item["messages"][0]
    image_path = user_message["content"][0]["image"]
    image = open_rgb_image(image_path, image_size)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(input_device(model))

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": sample,
        "num_return_sequences": num_candidates if sample else 1,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "pad_token_id": processor.tokenizer.pad_token_id,
    }
    if sample:
        generation_kwargs["temperature"] = 0.7
        generation_kwargs["top_p"] = 0.9

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generation_kwargs)

    generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def generate_candidate_actions(
    model,
    processor,
    policy_item: Dict,
    num_candidates: int,
    image_size: int,
    prompt_mode: str,
    reasoning_max_new_tokens: int,
    action_max_new_tokens: int,
) -> Tuple[List[str], List[str], Dict[str, str]]:
    action_only = prompt_mode == "action_only"
    prompt = build_policy_prompt(processor, policy_item, action_only=action_only)
    max_new_tokens = action_max_new_tokens if action_only else reasoning_max_new_tokens
    outputs = generate_from_prompt(
        model=model,
        processor=processor,
        policy_item=policy_item,
        prompt=prompt,
        sample=False,
        num_candidates=num_candidates,
        image_size=image_size,
        max_new_tokens=max_new_tokens,
    )
    outputs.extend(
        generate_from_prompt(
            model=model,
            processor=processor,
            policy_item=policy_item,
            prompt=prompt,
            sample=True,
            num_candidates=num_candidates,
            image_size=image_size,
            max_new_tokens=max_new_tokens,
        )
    )

    candidates = []
    candidate_responses = {}
    for output in outputs:
        response = clean_generated_response(output)
        action = normalize_action(extract_action(response, allow_raw=action_only))
        if action and is_single_step_action(action) and action not in candidates:
            candidates.append(action)
            candidate_responses[action] = response

    candidates = candidates[:num_candidates]
    candidate_responses = {
        action: response
        for action, response in candidate_responses.items()
        if action in candidates
    }
    return candidates, outputs, candidate_responses


def generate_reasoning_for_action(
    model,
    processor,
    policy_item: Dict,
    action: str,
    image_size: int,
    max_new_tokens: int,
) -> str:
    prompt = build_forced_reasoning_prompt(processor, policy_item, action)
    outputs = generate_from_prompt(
        model=model,
        processor=processor,
        policy_item=policy_item,
        prompt=prompt,
        sample=False,
        num_candidates=1,
        image_size=image_size,
        max_new_tokens=max_new_tokens,
    )
    return force_response_action(outputs[0] if outputs else "", action)


def replace_world_model_action(world_model_item: Dict, action: str) -> Dict:
    item = copy.deepcopy(world_model_item)
    text = item["messages"][0]["content"][1]["text"]
    lines = []
    replaced = False
    for line in text.splitlines():
        if line.startswith("Action to execute:"):
            lines.append(f"Action to execute: {action}")
            replaced = True
        else:
            lines.append(line)
    if not replaced:
        lines.append(f"Action to execute: {action}")
    item["messages"][0]["content"][1]["text"] = "\n".join(lines)
    return item


def left_pad_sequence(sequences, padding_value=0):
    max_len = max(seq.size(0) for seq in sequences)
    padded = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        if pad_len > 0:
            pad = seq.new_full((pad_len,), padding_value)
            seq = torch.cat([pad, seq], dim=0)
        padded.append(seq)
    return torch.stack(padded, dim=0)


def world_model_nll_batch(
    model,
    processor,
    world_model_items: List[Dict],
    image_size: int,
) -> List[float]:
    encoded = []
    pixel_values = []
    image_grid_thw = []

    for world_model_item in world_model_items:
        user_message = world_model_item["messages"][0]
        assistant_text = world_model_item["messages"][1]["content"]
        image_path = user_message["content"][0]["image"]
        image = open_rgb_image(image_path, image_size)

        prompt = processor.apply_chat_template(
            [{"role": "user", "content": user_message["content"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = prompt + assistant_text
        full_inputs = processor(text=[full_text], images=[image], return_tensors="pt", padding=True)
        prompt_inputs = processor.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )

        labels = full_inputs["input_ids"][0].clone()
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels[:prompt_len] = -100
        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        item = {
            "input_ids": full_inputs["input_ids"][0],
            "attention_mask": full_inputs["attention_mask"][0],
            "labels": labels,
        }
        encoded.append(item)
        if "pixel_values" in full_inputs:
            pixel_values.append(full_inputs["pixel_values"])
        if "image_grid_thw" in full_inputs:
            image_grid_thw.append(full_inputs["image_grid_thw"][0])

    batch = {
        "input_ids": left_pad_sequence(
            [item["input_ids"] for item in encoded],
            padding_value=processor.tokenizer.pad_token_id or 0,
        ),
        "attention_mask": left_pad_sequence(
            [item["attention_mask"] for item in encoded],
            padding_value=0,
        ),
        "labels": left_pad_sequence(
            [item["labels"] for item in encoded],
            padding_value=-100,
        ),
    }
    if pixel_values:
        batch["pixel_values"] = torch.cat(pixel_values, dim=0)
    if image_grid_thw:
        batch["image_grid_thw"] = torch.stack(image_grid_thw, dim=0)

    device = input_device(model)
    labels = batch.pop("labels").to(device)
    batch = {key: value.to(device) for key, value in batch.items()}

    with torch.no_grad():
        outputs = model(**batch)

    shifted_logits = outputs.logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    loss_mask = shifted_labels != -100
    safe_labels = shifted_labels.masked_fill(~loss_mask, 0)
    token_losses = F.cross_entropy(
        shifted_logits.transpose(1, 2),
        safe_labels,
        reduction="none",
    )
    loss_sums = (token_losses * loss_mask).sum(dim=1)
    token_counts = loss_mask.sum(dim=1).clamp_min(1)
    return (loss_sums / token_counts).detach().float().cpu().tolist()


def world_model_nll(model, processor, world_model_item: Dict, image_size: int) -> float:
    user_message = world_model_item["messages"][0]
    assistant_text = world_model_item["messages"][1]["content"]
    image_path = user_message["content"][0]["image"]
    image = open_rgb_image(image_path, image_size)

    prompt = processor.apply_chat_template(
        [{"role": "user", "content": user_message["content"]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = prompt + assistant_text
    full_inputs = processor(text=[full_text], images=[image], return_tensors="pt", padding=True)
    prompt_inputs = processor.tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )

    labels = full_inputs["input_ids"].clone()
    prompt_len = prompt_inputs["input_ids"].shape[1]
    labels[:, :prompt_len] = -100
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100

    device = input_device(model)
    full_inputs = full_inputs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(**full_inputs, labels=labels)
    return float(outputs.loss.item())


def build_dpo_record(
    policy_item: Dict,
    gt_action: str,
    rejected_action: str,
    candidates: List[str],
    gt_nll: float,
    rejected_nll: float,
    dpo_response_mode: str,
    rejected_response: Optional[str] = None,
) -> Dict:
    if dpo_response_mode == "reasoning":
        prompt_messages = make_reasoning_prompt_messages(policy_item)
        chosen = policy_item["messages"][1]["content"].strip()
        rejected = (rejected_response or f"Action: {rejected_action}").strip()
        task_type = "policy_reasoning_dpo"
    else:
        prompt_messages = make_action_only_prompt_messages(policy_item)
        chosen = f"Action: {gt_action}"
        rejected = f"Action: {rejected_action}"
        task_type = "policy_dpo"

    return {
        "task_type": task_type,
        "split": policy_item.get("split"),
        "trajectory_index": policy_item.get("trajectory_index"),
        "step_index": policy_item.get("step_index"),
        "prompt_messages": prompt_messages,
        "chosen": chosen,
        "rejected": rejected,
        "metadata": {
            "dpo_response_mode": dpo_response_mode,
            "gt_action": gt_action,
            "rejected_action": rejected_action,
            "candidate_actions": candidates,
            "gt_world_model_nll": gt_nll,
            "rejected_world_model_nll": rejected_nll,
            "wm_margin": rejected_nll - gt_nll,
            "negative_source": "policy_hard_negative_selected_by_world_model_margin",
        },
    }


def build_skip_record(
    policy_item: Dict,
    reason: str,
    gt_action: str = "",
    candidates: Optional[List[str]] = None,
    raw_outputs: Optional[List[str]] = None,
    hard_negatives: Optional[List[str]] = None,
    best_margin: Optional[float] = None,
) -> Dict:
    return {
        "reason": reason,
        "split": policy_item.get("split"),
        "trajectory_index": policy_item.get("trajectory_index"),
        "step_index": policy_item.get("step_index"),
        "gt_action": gt_action,
        "candidate_actions": candidates or [],
        "hard_negatives": hard_negatives or [],
        "raw_policy_outputs": raw_outputs or [],
        "best_margin": best_margin,
    }


def main():
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")
    require_file(args.policy_data_path)
    require_file(args.world_model_data_path)

    policy_samples = load_policy_samples(args.policy_data_path, args.max_samples)
    needed_keys = {sample_key(item) for item in policy_samples}
    world_model_by_key = load_world_model_matches(args.world_model_data_path, needed_keys)
    policy_samples = [item for item in policy_samples if sample_key(item) in world_model_by_key]
    if args.num_shards > 1:
        policy_samples = [
            item
            for idx, item in enumerate(policy_samples)
            if idx % args.num_shards == args.shard_index
        ]
    if not policy_samples:
        raise RuntimeError("No policy/world-model training samples share the same keys.")

    processor = AutoProcessor.from_pretrained(args.base_model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    deferred_reasoning = (
        args.dpo_response_mode == "reasoning"
        and args.candidate_prompt_mode == "action_only"
    )
    policy_model = load_lora_model(args.base_model_path, args.policy_lora_path)
    world_model = load_lora_model(args.base_model_path, args.world_model_lora_path)
    print(
        f"Processing samples: {len(policy_samples)} "
        f"(shard {args.shard_index + 1}/{args.num_shards})"
    )

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    if args.debug_skips_path:
        os.makedirs(os.path.dirname(args.debug_skips_path) or ".", exist_ok=True)
    kept = 0
    skipped = 0

    debug_f = open(args.debug_skips_path, "w", encoding="utf-8") if args.debug_skips_path else None
    with open(args.output_path, "w", encoding="utf-8") as out_f:
        for policy_item in tqdm(policy_samples):
            gt_action = normalize_action(extract_action(policy_item["messages"][1]["content"]))
            if not gt_action:
                if debug_f:
                    debug_f.write(
                        json.dumps(
                            build_skip_record(policy_item, reason="empty_gt_action"),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                skipped += 1
                continue

            candidates, raw_outputs, candidate_responses = generate_candidate_actions(
                model=policy_model,
                processor=processor,
                policy_item=policy_item,
                num_candidates=args.num_candidates,
                image_size=args.image_size,
                prompt_mode=args.candidate_prompt_mode,
                reasoning_max_new_tokens=args.reasoning_max_new_tokens,
                action_max_new_tokens=args.action_max_new_tokens,
            )
            hard_negatives = [
                action
                for action in candidates
                if action != gt_action and not is_near_duplicate_action(action, gt_action)
            ]
            if not hard_negatives:
                if debug_f:
                    debug_f.write(
                        json.dumps(
                            build_skip_record(
                                policy_item,
                                reason="no_hard_negatives",
                                gt_action=gt_action,
                                candidates=candidates,
                                raw_outputs=raw_outputs,
                                hard_negatives=hard_negatives,
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                skipped += 1
                continue

            wm_item = world_model_by_key[sample_key(policy_item)]
            wm_batch_items = [wm_item]
            wm_batch_items.extend(
                replace_world_model_action(wm_item, action)
                for action in hard_negatives
            )
            nlls = world_model_nll_batch(
                world_model,
                processor,
                wm_batch_items,
                args.image_size,
            )
            gt_nll = nlls[0]
            scored = [
                (nll - gt_nll, nll, action)
                for nll, action in zip(nlls[1:], hard_negatives)
            ]

            scored.sort(reverse=True)
            best_margin, rejected_nll, rejected_action = scored[0]
            if best_margin < args.min_margin:
                if debug_f:
                    debug_f.write(
                        json.dumps(
                            build_skip_record(
                                policy_item,
                                reason="margin_below_threshold",
                                gt_action=gt_action,
                                candidates=candidates,
                                raw_outputs=raw_outputs,
                                hard_negatives=hard_negatives,
                                best_margin=best_margin,
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                skipped += 1
                continue

            rejected_response = candidate_responses.get(rejected_action)
            if deferred_reasoning:
                rejected_response = generate_reasoning_for_action(
                    model=policy_model,
                    processor=processor,
                    policy_item=policy_item,
                    action=rejected_action,
                    image_size=args.image_size,
                    max_new_tokens=args.reasoning_max_new_tokens,
                )
                if not response_action_matches(rejected_response, rejected_action):
                    if debug_f:
                        debug_f.write(
                            json.dumps(
                                build_skip_record(
                                    policy_item,
                                    reason="rejected_reasoning_action_mismatch",
                                    gt_action=gt_action,
                                    candidates=candidates,
                                    raw_outputs=[rejected_response],
                                    hard_negatives=hard_negatives,
                                    best_margin=best_margin,
                                ),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    skipped += 1
                    continue

            record = build_dpo_record(
                policy_item=policy_item,
                gt_action=gt_action,
                rejected_action=rejected_action,
                candidates=candidates,
                gt_nll=gt_nll,
                rejected_nll=rejected_nll,
                dpo_response_mode=args.dpo_response_mode,
                rejected_response=rejected_response,
            )
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
    if debug_f:
        debug_f.close()

    print(f"Saved DPO pairs: {kept} -> {args.output_path}")
    print(f"Skipped samples: {skipped}")
    if args.debug_skips_path:
        print(f"Saved skipped-sample debug data to {args.debug_skips_path}")


if __name__ == "__main__":
    main()
