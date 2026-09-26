"""LoRA fine-tuning of a small model on a domain Q&A set.

This is the "train, don't just call APIs" part of the project. It fine-tunes
Qwen2.5-0.5B-Instruct with LoRA (PEFT) using TRL's SFTTrainer.

Run it on a GPU (a T4 or A10G is plenty for a 0.5B model):

    python finetune.py

Requires: peft, trl, accelerate (add them to requirements.txt before running).

NOTE: validate against your installed TRL version before running - the
SFTConfig/SFTTrainer API has changed across releases. The dataset below is a
real, commonly-used SFT set; swap it for your domain Q&A set (e.g. questions
about the HF docs) to make the fine-tune domain-specific.
"""

from __future__ import annotations

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# ---- Config -----------------------------------------------------------------

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET = "HuggingFaceH4/ultrachat_200k"  # swap for your domain Q&A set
DATASET_SPLIT = "train_sft"  # verify the split exists for your dataset
OUTPUT_DIR = "qwen2.5-0.5b-hf-docs-lora"
HUB_MODEL_ID = "pankaj-009/qwen2.5-0.5b-hf-docs-lora"  # your namespace


def format_chat(example: dict) -> dict:
    """Map a dataset row to the SFT 'messages' format.

    Ultrachat already has a 'messages' column; adapt this to your dataset's
    columns (e.g. build messages from a 'question'/'answer' pair).
    """
    return {"messages": example["messages"]}


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    dataset = load_dataset(DATASET, split=DATASET_SPLIT).select(range(2000))
    dataset = dataset.map(format_chat, remove_columns=dataset.column_names)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        max_seq_length=1024,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        bf16=True,
        push_to_hub=True,
        hub_model_id=HUB_MODEL_ID,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    trainer.push_to_hub()


if __name__ == "__main__":
    main()
