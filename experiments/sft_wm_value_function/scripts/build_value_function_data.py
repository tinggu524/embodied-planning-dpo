import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple


POLICY_DATA_PATH = "./data/processed/wap_qwen_policy_sft.jsonl"
WORLD_MODEL_DATA_PATH = "./data/processed/wap_qwen_worldmodel_sft.jsonl"
OUTPUT_SFT_PATH = "./data/processed/wap_qwen_value_function_sft.jsonl"
OUTPUT_TRAIN_PATH = "./data/processed/wap_qwen_value_function_train.jsonl"
OUTPUT_EVAL_PATH = "./data/processed/wap_qwen_value_function_eval.jsonl"

VALUE_TASK_TYPE = "value_function"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_data_path", default=POLICY_DATA_PATH)
    parser.add_argument("--world_model_data_path", default=WORLD_MODEL_DATA_PATH)
    parser.add_argument("--output_sft_path", default=OUTPUT_SFT_PATH)
    parser.add_argument("--output_train_path", default=OUTPUT_TRAIN_PATH)
    parser.add_argument("--output_eval_path", default=OUTPUT_EVAL_PATH)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--include_terminal_done",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add terminal policy states whose expert action is done as value=100 samples.",
    )
    return parser.parse_args()


def iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def write_jsonl(path: str, items: Iterable[Dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def sample_key(item: Dict) -> Tuple[str, int, int]:
    return (
        item.get("split", ""),
        int(item.get("trajectory_index", -1)),
        int(item.get("step_index", -1)),
    )


def trajectory_key(item: Dict) -> Tuple[str, int]:
    split, traj_idx, _ = sample_key(item)
    return split, traj_idx


def extract_field(text: str, field: str, default: str = "") -> str:
    match = re.search(rf"(?m)^{re.escape(field)}:\s*(.*)$", text)
    return match.group(1).strip() if match else default


def extract_action(text: str) -> Optional[str]:
    if "Action:" not in text:
        return None
    action = text.split("Action:", 1)[1].strip().splitlines()[0].strip()
    return action or None


def extract_reasoning(text: str) -> str:
    text = text.split("<|im_end|>")[0].strip()
    if "Action:" in text:
        text = text.split("Action:", 1)[0].strip()
    if text.startswith("Reasoning:"):
        text = text[len("Reasoning:") :].strip()
    return text


def normalize_action(action: Optional[str]) -> str:
    if not action:
        return ""
    action = action.strip().lower()
    action = re.sub(r"\s+", " ", action)
    return action.strip(" .")


def extract_predicted_state(world_model_answer: str) -> str:
    text = world_model_answer.split("<|im_end|>")[0].strip()
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


def build_trajectory_lengths(policy_path: str) -> Dict[Tuple[str, int], int]:
    lengths = defaultdict(int)
    for item in iter_jsonl(policy_path):
        key = trajectory_key(item)
        step = int(item.get("step_index", -1))
        lengths[key] = max(lengths[key], step + 1)
    return dict(lengths)


def load_terminal_policy_items(policy_path: str) -> Dict[Tuple[str, int], Dict]:
    terminals = {}
    for item in iter_jsonl(policy_path):
        key = trajectory_key(item)
        if key not in terminals or int(item["step_index"]) > int(terminals[key]["step_index"]):
            terminals[key] = item
    return terminals


def value_for_post_action_step(step_index: int, total_steps: int) -> int:
    if total_steps <= 0:
        return 0
    return max(0, min(100, round(100 * (step_index + 1) / total_steps)))


def build_value_user_text(
    instruction: str,
    history: str,
    action: str,
    predicted_state: str,
) -> str:
    return (
        "Task type: VALUE_FUNCTION\n"
        f"Instruction: {instruction}\n"
        f"Previous high-level actions: {history or 'None'}\n"
        "Current observation is provided as the image.\n"
        f"Action executed: {action}\n"
        "Predicted semantic state after action:\n"
        f"{predicted_state}\n\n"
        "Score how close this predicted post-action state is to completing the instruction. "
        "Use only task progress and do not rely on any hidden reference action.\n"
        "Output format:\n"
        "Value: <integer from 0 to 100>"
    )


def make_value_item(
    image_path: str,
    instruction: str,
    history: str,
    action: str,
    predicted_state: str,
    value: int,
    split: str,
    trajectory_index: int,
    step_index: int,
    source: str,
    total_steps: int,
) -> Dict:
    return {
        "task_type": VALUE_TASK_TYPE,
        "split": split,
        "trajectory_index": trajectory_index,
        "step_index": step_index,
        "value_label": value,
        "source": source,
        "total_steps": total_steps,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {
                        "type": "text",
                        "text": build_value_user_text(
                            instruction=instruction,
                            history=history,
                            action=action,
                            predicted_state=predicted_state,
                        ),
                    },
                ],
            },
            {
                "role": "assistant",
                "content": f"Value: {value}",
            },
        ],
    }


def iter_world_model_value_items(
    world_model_path: str,
    trajectory_lengths: Dict[Tuple[str, int], int],
):
    missing_lengths = 0
    for wm_item in iter_jsonl(world_model_path):
        split, traj_idx, step_idx = sample_key(wm_item)
        total_steps = trajectory_lengths.get((split, traj_idx))
        if not total_steps:
            missing_lengths += 1
            continue

        user_content = wm_item["messages"][0]["content"]
        user_text = user_content[-1]["text"]
        assistant_text = wm_item["messages"][1]["content"]
        predicted_state = extract_predicted_state(assistant_text)
        if not predicted_state:
            continue

        value = value_for_post_action_step(step_idx, total_steps)
        yield make_value_item(
            image_path=user_content[0]["image"],
            instruction=extract_field(user_text, "Instruction"),
            history=extract_field(user_text, "Previous high-level actions", "None"),
            action=extract_field(user_text, "Action to execute"),
            predicted_state=predicted_state,
            value=value,
            split=split,
            trajectory_index=traj_idx,
            step_index=step_idx,
            source="world_model_post_action_state",
            total_steps=total_steps,
        )
    if missing_lengths:
        print(f"Skipped world-model items with missing trajectory length: {missing_lengths}")


def iter_terminal_done_items(
    terminal_policy_items: Dict[Tuple[str, int], Dict],
    trajectory_lengths: Dict[Tuple[str, int], int],
):
    for (split, traj_idx), policy_item in terminal_policy_items.items():
        assistant_text = policy_item["messages"][1]["content"]
        action = normalize_action(extract_action(assistant_text))
        if action != "done":
            continue

        user_content = policy_item["messages"][0]["content"]
        user_text = user_content[-1]["text"]
        total_steps = trajectory_lengths[(split, traj_idx)]
        yield make_value_item(
            image_path=user_content[0]["image"],
            instruction=extract_field(user_text, "Instruction"),
            history=extract_field(user_text, "Previous high-level actions", "None"),
            action="done",
            predicted_state=extract_reasoning(assistant_text)
            or "The task is complete.",
            value=100,
            split=split,
            trajectory_index=traj_idx,
            step_index=int(policy_item["step_index"]),
            source="terminal_done_policy_state",
            total_steps=total_steps,
        )


def build_eval_keys_from_lengths(
    trajectory_lengths: Dict[Tuple[str, int], int],
    eval_ratio: float,
    seed: int,
) -> Set[Tuple[str, int]]:
    rng = random.Random(seed)
    by_split = defaultdict(set)
    for split, traj_idx in trajectory_lengths:
        by_split[split].add(int(traj_idx))

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


def write_value_item(
    item: Dict,
    all_f,
    train_f,
    eval_f,
    eval_keys: Set[Tuple[str, int]],
):
    line = json.dumps(item, ensure_ascii=False) + "\n"
    all_f.write(line)
    key = (item["split"], int(item["trajectory_index"]))
    if key in eval_keys:
        eval_f.write(line)
        return "eval"
    train_f.write(line)
    return "train"


def main():
    args = parse_args()
    if not 0 < args.eval_ratio < 1:
        raise ValueError("--eval_ratio must be between 0 and 1")
    if not os.path.exists(args.policy_data_path):
        raise FileNotFoundError(args.policy_data_path)
    if not os.path.exists(args.world_model_data_path):
        raise FileNotFoundError(args.world_model_data_path)

    trajectory_lengths = build_trajectory_lengths(args.policy_data_path)
    eval_keys = build_eval_keys_from_lengths(
        trajectory_lengths,
        args.eval_ratio,
        args.seed,
    )

    os.makedirs(os.path.dirname(args.output_sft_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_train_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.output_eval_path) or ".", exist_ok=True)

    counts = Counter()
    split_counts = Counter()
    first_item = None

    with open(args.output_sft_path, "w", encoding="utf-8") as all_f, open(
        args.output_train_path,
        "w",
        encoding="utf-8",
    ) as train_f, open(args.output_eval_path, "w", encoding="utf-8") as eval_f:
        for item in iter_world_model_value_items(
            args.world_model_data_path,
            trajectory_lengths,
        ):
            if first_item is None:
                first_item = item
            split_name = write_value_item(item, all_f, train_f, eval_f, eval_keys)
            counts[item["source"]] += 1
            split_counts[split_name] += 1

        if args.include_terminal_done:
            terminal_policy_items = load_terminal_policy_items(args.policy_data_path)
            print(f"Loaded terminal policy states: {len(terminal_policy_items)}")
            terminal_count = 0
            for item in iter_terminal_done_items(
                terminal_policy_items,
                trajectory_lengths,
            ):
                terminal_count += 1
                if first_item is None:
                    first_item = item
                split_name = write_value_item(item, all_f, train_f, eval_f, eval_keys)
                counts[item["source"]] += 1
                split_counts[split_name] += 1
            print(f"Added terminal done value samples: {terminal_count}")

    total = sum(counts.values())
    print(f"Saved value SFT samples: {total} -> {args.output_sft_path}")
    print(f"Saved train samples: {split_counts['train']} -> {args.output_train_path}")
    print(f"Saved eval samples: {split_counts['eval']} -> {args.output_eval_path}")
    print("Source counts:", dict(counts))
    if first_item:
        print("Example:")
        print(json.dumps(first_item, ensure_ascii=False, indent=2)[:3000])


if __name__ == "__main__":
    main()
