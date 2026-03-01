"""
Unsloth SFT Training Script for HuggingFace Jobs
Uploads this script to HuggingFace Jobs for managed GPU training.
"""

# Install dependencies
print("Installing dependencies...")
import subprocess
subprocess.run(["pip", "install", "unsloth"], check=True)
subprocess.run(["pip", "install", "--upgrade", "transformers>=5.0", "trl", "datasets"], check=True)

import os
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastVisionModel
from pathlib import Path

# Configuration
REPO_ID = os.environ.get("REPO_ID", "unsloth/Ministral-3-3B-Instruct-2512")
DATA_REPO_ID = os.environ.get("DATA_REPO_ID", "yongtae-jp/AiOrDie-dataset")
OUTPUT_REPO_ID = os.environ.get("OUTPUT_REPO_ID", "yongtae-jp/AiOrDie-Ministral-3B")

print(f"Model: {REPO_ID}")
print(f"Data: {DATA_REPO_ID}")
print(f"Output: {OUTPUT_REPO_ID}")

# Load dataset from HuggingFace Hub
print("\nLoading dataset...")
dataset = load_dataset(DATA_REPO_ID)

# If dataset has train/val/test splits, use them
train_data = dataset.get("train", dataset)
val_data = dataset.get("val", None)

print(f"Train samples: {len(train_data)}")
print(f"Val samples: {len(val_data) if val_data else 0}")

# Load model with FastVisionModel
print("\nLoading model...")
model, tokenizer = FastVisionModel.from_pretrained(
    REPO_ID,
    load_in_4bit=False,  # 16bit LoRA (False) or 4bit (True) to reduce memory
    use_gradient_checkpointing="unsloth",  # for long context
)

# Apply LoRA
print("\nApplying LoRA...")
model = FastVisionModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
)

print(f"Model dtype: {model.dtype}")
print(f"VRAM usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# Training configuration
training_args = SFTConfig(
    output_dir="./outputs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to=["tensorboard"],
)

# Create trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=training_args,
    max_seq_length=4096,
)

# Train
print("\n" + "="*60)
print("Starting training...")
print("="*60)
trainer.train()

# Save LoRA adapter
print("\nSaving LoRA adapter...")
output_path = Path("./outputs/aiordie_lora")
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print(f"✓ Model saved to {output_path}")

# Optionally export to GGUF
try:
    print("\nExporting to GGUF...")
    gguf_path = Path("./outputs/aiordie_gguf")
    model.save_pretrained_gguf(
        gguf_path,
        tokenizer,
        quantization_method="q4_k_m"  # ~1.5GB
    )
    print(f"✓ GGUF model saved to {gguf_path}")
except Exception as e:
    print(f"⚠️  GGUF export failed: {e}")

print("\n" + "="*60)
print("Training complete!")
print("="*60)
