import argparse
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from image_utils import open_rgb_image


BASE_MODEL_PATH = "/root/Qwen2.5-VL-3B-Instruct"
POLICY_LORA_PATH = "./models/qwen2_5_vl_3b_wap_policy_lora"
DATA_PATH = "./data/processed/wap_qwen_policy_dpo_wm_pairs.jsonl"
OUTPUT_DIR = "./models/qwen2_5_vl_3b_wap_policy_dpo_lora"

MAX_SAMPLES = None
IMAGE_SIZE = 224
ASSISTANT_END = "<|im_end|>"
DPO_BETA = 0.1
SFT_WEIGHT = 0.0


def optional_int(value: str) -> Optional[int]:
    if value.lower() in {"none", "null"}:
        return None
    return int(value)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", default=BASE_MODEL_PATH)
    parser.add_argument("--policy_lora_path", default=POLICY_LORA_PATH)
    parser.add_argument("--data_path", default=DATA_PATH)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--max_samples", type=optional_int, default=MAX_SAMPLES)
    parser.add_argument("--beta", type=float, default=DPO_BETA)
    parser.add_argument("--sft_weight", type=float, default=SFT_WEIGHT)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler_type", default="linear")
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=float, default=1)
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--max_response_tokens", type=optional_int, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    return parser.parse_args()


@dataclass
class PolicyDPODataset(Dataset):
    jsonl_path: str
    processor: Any
    max_samples: Optional[int] = None
    image_size: int = IMAGE_SIZE
    max_response_tokens: Optional[int] = None
    samples: List[Dict] = field(init=False)

    def __post_init__(self):
        self.samples = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if self.max_samples is not None and idx >= self.max_samples:
                    break
                self.samples.append(json.loads(line))
        print("Loaded DPO pairs:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def encode_response(self, item: Dict, response: str) -> Dict:
        prompt_messages = item["prompt_messages"]
        image_path = prompt_messages[0]["content"][0]["image"]
        image = open_rgb_image(image_path, self.image_size)

        prompt = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        response = response.strip()
        if self.max_response_tokens is not None:
            response_ids = self.processor.tokenizer(
                response,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_response_tokens,
            )["input_ids"]
            response = self.processor.tokenizer.decode(
                response_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ).strip()
        if not response.endswith(ASSISTANT_END):
            response = response + ASSISTANT_END
        full_text = prompt + response

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

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            "chosen": self.encode_response(item, item["chosen"]),
            "rejected": self.encode_response(item, item["rejected"]),
        }


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


def collate_dpo_features(features):
    chosen = [feature["chosen"] for feature in features]
    rejected = [feature["rejected"] for feature in features]
    flat = chosen + rejected
    batch = {}

    for key in ["input_ids", "attention_mask", "labels"]:
        batch[key] = left_pad_sequence(
            [feature[key] for feature in flat],
            batch_first=True,
            padding_value=0 if key != "labels" else -100,
        )

    if "pixel_values" in flat[0]:
        batch["pixel_values"] = torch.cat(
            [feature["pixel_values"] for feature in flat],
            dim=0,
        )

    if "image_grid_thw" in flat[0]:
        batch["image_grid_thw"] = torch.stack(
            [feature["image_grid_thw"] for feature in flat],
            dim=0,
        )

    batch["pair_count"] = torch.tensor(len(features), dtype=torch.long)
    return batch


def sequence_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    loss_mask = shifted_labels != -100
    safe_labels = shifted_labels.masked_fill(~loss_mask, 0)
    token_logps = torch.gather(
        F.log_softmax(shifted_logits, dim=-1),
        dim=2,
        index=safe_labels.unsqueeze(2),
    ).squeeze(2)
    return (token_logps * loss_mask).sum(dim=1)


def sequence_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted_logits = logits[:, :-1, :]
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
    return loss_sums / token_counts


class DPOTrainer(Trainer):
    def __init__(
        self,
        *args,
        ref_model=None,
        beta: float = DPO_BETA,
        sft_weight: float = SFT_WEIGHT,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.beta = beta
        self.sft_weight = sft_weight
        self._ref_device = None
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad_(False)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        pair_count = int(inputs.pop("pair_count").item())
        labels = inputs.pop("labels")
        model_device = next(model.parameters()).device
        if self._ref_device != model_device:
            self.ref_model.to(model_device)
            self._ref_device = model_device

        outputs = model(**inputs)
        policy_logps = sequence_logps(outputs.logits, labels)

        with torch.no_grad():
            ref_outputs = self.ref_model(**inputs)
            ref_logps = sequence_logps(ref_outputs.logits, labels)

        policy_chosen, policy_rejected = policy_logps[:pair_count], policy_logps[pair_count:]
        ref_chosen, ref_rejected = ref_logps[:pair_count], ref_logps[pair_count:]

        policy_logratios = policy_chosen - policy_rejected
        ref_logratios = ref_chosen - ref_rejected
        logits = policy_logratios - ref_logratios
        dpo_loss = -F.logsigmoid(self.beta * logits).mean()
        loss = dpo_loss

        if self.sft_weight > 0:
            chosen_nll = sequence_nll(outputs.logits[:pair_count], labels[:pair_count]).mean()
            loss = loss + self.sft_weight * chosen_nll

        if return_outputs:
            return loss, {
                "policy_chosen_logps": policy_chosen.detach(),
                "policy_rejected_logps": policy_rejected.detach(),
                "dpo_loss": dpo_loss.detach(),
            }
        return loss


def load_policy_model(base_model_path: str, lora_path: str, trainable: bool):
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(base_model, lora_path, is_trainable=trainable)
    model.config.use_cache = False
    return model


def main():
    cli_args = parse_args()

    processor = AutoProcessor.from_pretrained(
        cli_args.base_model_path,
        trust_remote_code=True,
    )
    processor.tokenizer.padding_side = "left"

    model = load_policy_model(
        cli_args.base_model_path,
        cli_args.policy_lora_path,
        trainable=True,
    )
    if cli_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    ref_model = load_policy_model(
        cli_args.base_model_path,
        cli_args.policy_lora_path,
        trainable=False,
    )
    model.print_trainable_parameters()

    train_dataset = PolicyDPODataset(
        cli_args.data_path,
        processor,
        max_samples=cli_args.max_samples,
        image_size=cli_args.image_size,
        max_response_tokens=cli_args.max_response_tokens,
    )

    args = TrainingArguments(
        output_dir=cli_args.output_dir,
        per_device_train_batch_size=cli_args.per_device_train_batch_size,
        gradient_accumulation_steps=cli_args.gradient_accumulation_steps,
        learning_rate=cli_args.learning_rate,
        lr_scheduler_type=cli_args.lr_scheduler_type,
        warmup_ratio=cli_args.warmup_ratio,
        num_train_epochs=cli_args.num_train_epochs,
        logging_steps=20,
        save_steps=1000,
        save_total_limit=2,
        bf16=True,
        fp16=False,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        gradient_checkpointing=cli_args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False}
        if cli_args.gradient_checkpointing
        else None,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        beta=cli_args.beta,
        sft_weight=cli_args.sft_weight,
        args=args,
        train_dataset=train_dataset,
        data_collator=collate_dpo_features,
    )
    trainer.train()
    trainer.save_model(cli_args.output_dir)
    processor.save_pretrained(cli_args.output_dir)


if __name__ == "__main__":
    main()
