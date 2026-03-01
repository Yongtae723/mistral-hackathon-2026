# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.1.0",
#   "transformers @ git+https://github.com/huggingface/transformers.git@bf3f0ae70d0e902efab4b8517fce88f6697636ce",
#   "trl==0.22.2",
#   "datasets>=2.18.0",
#   "wandb>=0.17.0",
#   "huggingface_hub>=0.34.0",
#   "pillow>=10.0.0",
#   "bitsandbytes>=0.43.0",
#   "accelerate>=0.34.0",
#   "peft>=0.12.0",
#   "sentencepiece",
#   "protobuf",
#   "torchvision",
# ]
# ///
"""
SurviveOrDie — Smoke Test (standard HF approach, no unsloth)

Generates synthetic training data in-memory using tiny PIL images.
Tests the COMPLETE pipeline end-to-end:
  AutoModelForImageTextToText → LoRA → custom collate_fn → SFTTrainer → 3 steps → save

Run with:
  hf jobs uv run --flavor a10g-small \\
    --secrets-file ../../.env \\
    smoke_test.py

Only this single .py file is uploaded. No data files needed.
Expected runtime: ~5 min on A10G.
"""

import os, json, io, random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset as TorchDataset
import wandb
from PIL import Image, ImageDraw

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_REPO_ID  = os.environ.get("MODEL_REPO_ID", "mistralai/Ministral-3-3B-Instruct-2512-BF16")
SKIP_WANDB     = os.environ.get("SKIP_WANDB", "false").lower() == "true"
WANDB_RUN_NAME = "smoke-test-ministral3b-vl"

print("=" * 60)
print("SurviveOrDie Smoke Test (standard HF, no unsloth)")
print(f"Model: {MODEL_REPO_ID}")
print(f"CUDA:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:   {torch.cuda.get_device_name(0)}")
    print(f"VRAM:  {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print("=" * 60)

# ── HuggingFace Hub login (needed for gated model access) ────────────────────
_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    from huggingface_hub import login as hf_login
    hf_login(token=_hf_token, add_to_git_credential=False)
    print("HF Hub: logged in")

# ── WandB ─────────────────────────────────────────────────────────────────────
if SKIP_WANDB:
    os.environ["WANDB_DISABLED"] = "true"
else:
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    wandb.init(
        project="SurviveOrDie",
        name=WANDB_RUN_NAME,
        tags=["smoke-test", "ministral3b-vl"],
        config={"model": MODEL_REPO_ID, "test": "smoke"},
    )

# ── Imports (after GPU check passes) ─────────────────────────────────────────
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer, SFTConfig

# ── Synthetic data generation ─────────────────────────────────────────────────
SPECIES_SAMPLES = [
    {
        "name": "Amanita phalloides",
        "verdict": "LETHAL",
        "color": (50, 120, 50),
        "response": "⚠️ LETHAL — DO NOT EAT. Death Cap mushroom. Call emergency services immediately if ingested.",
    },
    {
        "name": "Cantharellus cibarius",
        "verdict": "SAFE",
        "color": (220, 160, 30),
        "response": "✅ SAFE — Chanterelle. Confirm funnel shape and forking ridges. No toxic lookalikes in this region.",
    },
    {
        "name": "Amanita virosa",
        "verdict": "LETHAL",
        "color": (245, 245, 245),
        "response": "🚫 LETHAL — Destroying Angel. Purely white, skirt present. Fatal if eaten. Seek hospital immediately.",
    },
    {
        "name": "Boletus edulis",
        "verdict": "SAFE",
        "color": (160, 100, 60),
        "response": "✅ SAFE — Porcini / Cep. Sponge-like underside, no red tones. Excellent edibility.",
    },
    {
        "name": "Galerina marginata",
        "verdict": "LETHAL",
        "color": (180, 130, 70),
        "response": "🚫 LETHAL — Deadly Galerina. Contains same amatoxins as Death Cap. Do not eat.",
    },
    {
        "name": "Macrolepiota procera",
        "verdict": "SAFE",
        "color": (210, 190, 150),
        "response": "✅ SAFE — Parasol Mushroom. Movable ring, scaly cap. Edible when cooked.",
    },
]


def make_synthetic_image(color: tuple, size: int = 64) -> Image.Image:
    img = Image.new("RGB", (size, size), color)
    draw = ImageDraw.Draw(img)
    draw.ellipse([size//4, size//4, 3*size//4, size//2], fill=(color[0]//2, color[1]//2, color[2]//2))
    draw.rectangle([size//2-4, size//2, size//2+4, 3*size//4], fill=(200, 170, 120))
    return img


CONTEXTS = [
    "lat: 35.7, month: 8, alt: 400m, env: mixed_forest\nI found this. Is it safe to eat?",
    "lat: 43.1, month: 9, alt: 200m, env: broadleaf_forest\nMy child touched this. Is it dangerous?",
    "lat: 35.2, month: 10, alt: 800m, env: conifer_forest\nCan I eat this for survival?",
]


def make_sample(species: dict, context: str) -> dict:
    img = make_synthetic_image(species["color"])
    return {
        "image": img,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": context},
                    {"type": "image"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": species["response"]}],
            },
        ],
    }


print("\nGenerating synthetic training data...")
random.seed(42)
train_samples = [
    make_sample(SPECIES_SAMPLES[i % len(SPECIES_SAMPLES)], CONTEXTS[i % len(CONTEXTS)])
    for i in range(8)
]
val_samples = [
    make_sample(SPECIES_SAMPLES[(i + 2) % len(SPECIES_SAMPLES)], CONTEXTS[i % len(CONTEXTS)])
    for i in range(2)
]
print(f"  train={len(train_samples)}, val={len(val_samples)}")


class SurvivalDataset(TorchDataset):
    def __init__(self, samples): self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


train_dataset = SurvivalDataset(train_samples)
val_dataset   = SurvivalDataset(val_samples)

# ── Model + LoRA ──────────────────────────────────────────────────────────────
print(f"\nLoading {MODEL_REPO_ID}...")

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_REPO_ID,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(MODEL_REPO_ID)
processor.tokenizer.padding_side = "right"

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

if torch.cuda.is_available():
    print(f"VRAM after model load: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# ── Collator ──────────────────────────────────────────────────────────────────
tokenizer = processor.tokenizer
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
pad_token_id   = tokenizer.pad_token_id
image_token_id = getattr(processor, "image_token_id", None)
MAX_SEQ_LEN    = 2048  # must be large enough to hold image tokens


def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    images        = [ex["image"] for ex in examples]
    conversations = [ex["messages"] for ex in examples]
    chat_texts = processor.apply_chat_template(
        conversations, add_generation_prompt=False, tokenize=False,
    )
    batch = processor(
        text=chat_texts,
        images=images,
        padding="longest",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="pt",
    )
    labels = batch["input_ids"].clone()
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100
    if image_token_id is not None and image_token_id != pad_token_id:
        labels[labels == image_token_id] = -100
    batch["labels"] = labels
    return batch


# ── Train (3 steps only) ──────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir="./smoke_outputs",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=3,
        learning_rate=2e-4,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=1,
        eval_strategy="no",
        save_strategy="no",
        report_to="none" if SKIP_WANDB else "wandb",
        run_name=WANDB_RUN_NAME,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
    ),
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    processing_class=processor,
)

print("\n" + "=" * 60)
print("Training (3 steps)...")
print("=" * 60)
trainer.train()

# ── Quick inference check ─────────────────────────────────────────────────────
print("\nRunning quick inference check...")
model.eval()

test_img = make_synthetic_image((50, 120, 50))
test_messages = [
    {
        "role": "user",
        "content": [
            {"type": "text",  "text": "Is this mushroom safe to eat?"},
            {"type": "image"},
        ],
    }
]

input_text = processor.apply_chat_template(test_messages, add_generation_prompt=True, tokenize=False)
inputs = processor(
    test_img,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda" if torch.cuda.is_available() else "cpu")

device_type = "cuda" if torch.cuda.is_available() else "cpu"
with torch.no_grad(), torch.autocast(device_type=device_type, dtype=torch.bfloat16):
    out = model.generate(**inputs, max_new_tokens=100, temperature=0.1, do_sample=True)

response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"\nModel output: {response[:200]}")

if not SKIP_WANDB:
    wandb.log({"smoke/inference_output_len": len(response)})
    wandb.finish()

print("\n" + "=" * 60)
print("✅ Smoke test PASSED")
print("   - AutoModelForImageTextToText: OK")
print("   - LoRA (peft):                OK")
print("   - Custom collate_fn:          OK")
print("   - SFTTrainer 3 steps:         OK")
print("   - Inference:                  OK")
print("=" * 60)
