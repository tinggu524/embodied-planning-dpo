import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

SHARED_SCRIPTS = Path(__file__).resolve().parents[3] / "shared" / "scripts"
if str(SHARED_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPTS))

from image_utils import open_rgb_image
from wap_sampling import balanced_reservoir_sample


MODEL_PATH = "/root/Qwen2.5-VL-3B-Instruct"
DATA_PATH = "./data/processed/wap_qwen_value_function_train.jsonl"
OUTPUT_DIR = "./models/qwen2_5_vl_3b_wap_value_function_lora"
MAX_SAMPLES = 40000
IMAGE_SIZE = 224
ASSISTANT_END = "<|im_end|>"
SAMPLE_SEED = 44
SPLIT_GROUPS = [("reference",), ("spatial",), ("symbolic",), ("visual",)]


def optional_int(value: str) -> Optional[int]:
    if value.lower() in {"none", "null"}:
        return None
    return int(value)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--data_path", default=DATA_PATH)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--max_samples", type=optional_int, default=MAX_SAMPLES)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=float, default=1)
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    return parser.parse_args()


@dataclass
class ValueFunctionDataset(Dataset):
    jsonl_path: str
    processor: Any
    max_samples: Optional[int] = None
    image_size: int = IMAGE_SIZE
    samples: List[Dict] = field(init=False)

    def __post_init__(self):
        self.samples = balanced_reservoir_sample(
            jsonl_path=self.jsonl_path,
            max_samples=self.max_samples,
            seed=SAMPLE_SEED,
            group_fields=["split"],
            groups=SPLIT_GROUPS,
        )
        print("Loaded value samples:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        messages = item["messages"]
        image_path = messages[0]["content"][0]["image"]
        image = open_rgb_image(image_path, self.image_size)

        user_messages = [
            {
                "role": "user",
                "content": messages[0]["content"],
            }
        ]
        assistant_text = messages[1]["content"].strip()
        if not assistant_text.endswith(ASSISTANT_END):
            assistant_text = assistant_text + ASSISTANT_END

        prompt = self.processor.apply_chat_template(
            user_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = prompt + assistant_text

        full_inputs = self.processor(
            text=[full_text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        prompt_inputs = self.processor.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )

        input_ids = full_inputs["input_ids"][0]
        attention_mask = full_inputs["attention_mask"][0]
        labels = input_ids.clone()
        labels[: prompt_inputs["input_ids"].shape[1]] = -100

        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if "pixel_values" in full_inputs:
            result["pixel_values"] = full_inputs["pixel_values"]
        if "image_grid_thw" in full_inputs:
            result["image_grid_thw"] = full_inputs["image_grid_thw"][0]
        return result


def left_pad_sequence(sequences, batch_first=True, padding_value=0):
    max_len = max(seq.size(0) for seq in sequences)
    padded = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        if pad_len > 0:
            pad = seq.new_full((pad_len,), padding_value)
            seq = torch.cat([pad, seq], dim=0)
        padded.append(seq)
    return torch.stack(padded, dim=0 if batch_first else 1)


def collate_fn(features):
    batch = {}
    for key in ["input_ids", "attention_mask", "labels"]:
        batch[key] = left_pad_sequence(
            [f[key] for f in features],
            batch_first=True,
            padding_value=0 if key != "labels" else -100,
        )

    if "pixel_values" in features[0]:
        batch["pixel_values"] = torch.cat([f["pixel_values"] for f in features], dim=0)
    if "image_grid_thw" in features[0]:
        batch["image_grid_thw"] = torch.stack(
            [f["image_grid_thw"] for f in features],
            dim=0,
        )
    return batch


def main():
    cli_args = parse_args()

    processor = AutoProcessor.from_pretrained(
        cli_args.model_path,
        trust_remote_code=True,
    )
    processor.tokenizer.padding_side = "left"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cli_args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    lora_config = LoraConfig(
        r=cli_args.lora_r,
        lora_alpha=cli_args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=cli_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    if cli_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.print_trainable_parameters()

    train_dataset = ValueFunctionDataset(
        cli_args.data_path,
        processor,
        max_samples=cli_args.max_samples,
        image_size=cli_args.image_size,
    )

    args = TrainingArguments(
        output_dir=cli_args.output_dir,
        per_device_train_batch_size=cli_args.per_device_train_batch_size,
        gradient_accumulation_steps=cli_args.gradient_accumulation_steps,
        learning_rate=cli_args.learning_rate,
        num_train_epochs=cli_args.num_train_epochs,
        logging_steps=100,
        save_steps=5000,
        save_total_limit=2,
        bf16=True,
        fp16=False,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        gradient_checkpointing=cli_args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False}
        if cli_args.gradient_checkpointing
        else None,
        ddp_find_unused_parameters=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )
    trainer.train()
    trainer.save_model(cli_args.output_dir)
    processor.save_pretrained(cli_args.output_dir)


if __name__ == "__main__":
    main()
