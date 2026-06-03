import glob
import json
import os
from typing import Dict, List, Optional

from tqdm import tqdm


DATA_ROOT = "./data/raw/World-Aware-Planning"
IMAGE_ROOT = "./data/raw/World-Aware-Planning/images"
POLICY_OUTPUT_PATH = "./data/processed/wap_qwen_policy_sft.jsonl"
WORLD_MODEL_OUTPUT_PATH = "./data/processed/wap_qwen_worldmodel_sft.jsonl"
MIXED_OUTPUT_PATH = "./data/processed/wap_qwen_policy_worldmodel_sft.jsonl"

SPLITS = ["reference", "spatial", "symbolic", "visual"]


def extract_action(text: str) -> Optional[str]:
    if "Action:" not in text:
        return None
    return text.split("Action:", 1)[1].strip().splitlines()[0].strip()


def extract_reasoning(text: str) -> str:
    text = text.strip()
    if "Action:" in text:
        text = text.split("Action:", 1)[0]
    if text.startswith("Reasoning:"):
        text = text[len("Reasoning:") :]
    return text.strip()


def image_path_for(image_ref: str) -> str:
    return os.path.join(IMAGE_ROOT, os.path.basename(image_ref))


def build_policy_user_text(instruction: str, history_actions: List[str]) -> str:
    history = "None" if not history_actions else " -> ".join(history_actions)

    return (
        "Task type: POLICY\n"
        f"Instruction: {instruction}\n"
        f"Previous high-level actions: {history}\n"
        "Current observation is provided as the image.\n"
        "Predict the next high-level action. The action is a semantic robot step, "
        "not a joint angle or low-level motor command.\n"
        "Output format:\n"
        "Reasoning: ...\n"
        "Action: ..."
    )


def build_world_model_user_text(
    instruction: str,
    history_actions: List[str],
    action: str,
) -> str:
    history = "None" if not history_actions else " -> ".join(history_actions)

    return (
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


def build_world_model_answer(next_assistant_text: str) -> str:
    next_action = extract_action(next_assistant_text) or "unknown"
    next_reasoning = extract_reasoning(next_assistant_text)
    progress = "done" if next_action.lower() == "done" else "in_progress"

    return (
        f"Predicted state: {next_reasoning}\n"
        f"Likely next action: {next_action}\n"
        f"Progress: {progress}"
    )


def make_item(
    image_path: str,
    user_text: str,
    assistant_text: str,
    task_type: str,
    split: str,
    trajectory_index: int,
    step_index: int,
) -> Dict:
    return {
        "task_type": task_type,
        "split": split,
        "trajectory_index": trajectory_index,
        "step_index": step_index,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {
                        "type": "text",
                        "text": user_text,
                    },
                ],
            },
            {
                "role": "assistant",
                "content": assistant_text,
            },
        ],
    }


def collect_steps(sample: Dict) -> List[Dict]:
    conversations = sample["conversations"]
    images = sample["images"]
    steps = []
    image_idx = 0

    for idx, turn in enumerate(conversations):
        if turn.get("from") != "observation":
            continue
        if idx + 1 >= len(conversations):
            continue
        if image_idx >= len(images):
            continue

        assistant_turn = conversations[idx + 1]
        if assistant_turn.get("from") != "gpt":
            image_idx += 1
            continue

        assistant_text = assistant_turn["value"]
        action = extract_action(assistant_text)
        if action is None:
            image_idx += 1
            continue

        steps.append(
            {
                "image": image_path_for(images[image_idx]),
                "assistant_text": assistant_text,
                "action": action,
            }
        )
        image_idx += 1

    return steps


def convert_one_traj(sample: Dict, split: str, trajectory_index: int) -> List[Dict]:
    instruction = sample["conversations"][0]["value"]
    steps = collect_steps(sample)

    results = []
    history_actions = []

    for step_idx, step in enumerate(steps):
        policy_user_text = build_policy_user_text(instruction, history_actions)
        results.append(
            make_item(
                image_path=step["image"],
                user_text=policy_user_text,
                assistant_text=step["assistant_text"],
                task_type="policy",
                split=split,
                trajectory_index=trajectory_index,
                step_index=step_idx,
            )
        )

        if step_idx + 1 < len(steps):
            wm_user_text = build_world_model_user_text(
                instruction=instruction,
                history_actions=history_actions,
                action=step["action"],
            )
            # Use the next step's reasoning/action as a textual proxy for the post-action state.
            wm_answer = build_world_model_answer(steps[step_idx + 1]["assistant_text"])
            results.append(
                make_item(
                    image_path=step["image"],
                    user_text=wm_user_text,
                    assistant_text=wm_answer,
                    task_type="world_model",
                    split=split,
                    trajectory_index=trajectory_index,
                    step_index=step_idx,
                )
            )

        history_actions.append(step["action"])

    return results


def load_split(split: str) -> List[Dict]:
    paths = sorted(glob.glob(os.path.join(DATA_ROOT, split, "*.json")))
    samples = []

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            samples.extend(data)
        else:
            samples.append(data)

    return samples


def main():
    task_counts = {"policy": 0, "world_model": 0}
    first_item = None
    os.makedirs(os.path.dirname(POLICY_OUTPUT_PATH), exist_ok=True)

    output_handles = {
        "policy": open(POLICY_OUTPUT_PATH, "w", encoding="utf-8"),
        "world_model": open(WORLD_MODEL_OUTPUT_PATH, "w", encoding="utf-8"),
    }
    mixed_handle = open(MIXED_OUTPUT_PATH, "w", encoding="utf-8")

    try:
        for split in SPLITS:
            samples = load_split(split)
            n = len(samples)

            print(f"Converting split={split}, trajectories={n}")

            for idx in tqdm(range(n)):
                items = convert_one_traj(samples[idx], split=split, trajectory_index=idx)
                for item in items:
                    if first_item is None:
                        first_item = item
                    task_type = item["task_type"]
                    line = json.dumps(item, ensure_ascii=False) + "\n"
                    output_handles[task_type].write(line)
                    mixed_handle.write(line)
                    task_counts[task_type] += 1
    finally:
        for handle in output_handles.values():
            handle.close()
        mixed_handle.close()

    print(f"Total SFT samples: {sum(task_counts.values())}")
    print(f"Policy samples: {task_counts['policy']}")
    print(f"World-model samples: {task_counts['world_model']}")
    print(f"Saved policy data to {POLICY_OUTPUT_PATH}")
    print(f"Saved world-model data to {WORLD_MODEL_OUTPUT_PATH}")
    print(f"Saved mixed data to {MIXED_OUTPUT_PATH}")
    if first_item:
        print("Example:")
        print(json.dumps(first_item, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
