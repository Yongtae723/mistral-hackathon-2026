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
SurviveOrDie — Fine-tuning Script (WandB Finetune)
Run with: hf jobs uv run src/train/train.py

Purpose: Train the model. Track loss curves via WandB Finetune.
         Does NOT evaluate output quality — that's eval.py (WandB Weave).

Required env vars (HF Secrets):
  WANDB_API_KEY   — from wandb.ai/authorize
  HF_TOKEN        — HuggingFace write token

Optional env vars:
  MODEL_REPO_ID   — base model          (default: mistralai/Ministral-3-3B-Instruct-2512-BF16)
  OUTPUT_REPO_ID  — push destination    (default: "", no push)
  WANDB_RUN_NAME  — run name            (default: ministral3b-survival-lora-v1)
  LOAD_IN_4BIT    — "true"/"false"      (default: "true")
  NUM_EPOCHS      — epochs              (default: 3)
  BATCH_SIZE      — per-device batch    (default: 1)
  GRAD_ACCUM      — grad accum steps    (default: 8)
  SKIP_PUSH       — "true" to skip Hub push (default: "false")
  MAX_TRAIN_SAMPLES — cap samples for quick runs (default: 0 = all)
  MAX_STEPS       — cap steps for quick runs     (default: 0 = full run)
"""

# ── 1. Imports ────────────────────────────────────────────────────────────────
import os, json, base64, io, re as _re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset as TorchDataset
import wandb
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig

# VLMs require eager attention for stability (flash_attn / sdpa can cause NaN gradients)
ATTN_IMPL = "eager"
print(f"Attention: {ATTN_IMPL}")

# ── 2. Config ─────────────────────────────────────────────────────────────────
MODEL_REPO_ID  = os.environ.get("MODEL_REPO_ID",  "mistralai/Ministral-3-3B-Instruct-2512-BF16")
OUTPUT_REPO_ID = os.environ.get("OUTPUT_REPO_ID", "")
WANDB_RUN_NAME = os.environ.get("WANDB_RUN_NAME", "ministral3b-survival-lora-v1")
LOAD_IN_4BIT   = os.environ.get("LOAD_IN_4BIT",   "true").lower() == "true"
NUM_EPOCHS     = int(os.environ.get("NUM_EPOCHS",  "3"))
BATCH_SIZE     = int(os.environ.get("BATCH_SIZE",  "1"))
GRAD_ACCUM     = int(os.environ.get("GRAD_ACCUM",  "8"))
SKIP_PUSH      = os.environ.get("SKIP_PUSH",       "false").lower() == "true"
SKIP_WANDB     = os.environ.get("SKIP_WANDB",      "false").lower() == "true"
MAX_TRAIN_SAMPLES = int(os.environ.get("MAX_TRAIN_SAMPLES", "0"))  # 0 = all
MAX_STEPS         = int(os.environ.get("MAX_STEPS",         "0"))  # 0 = full run
MAX_SEQ_LEN       = 4096

DATASET_REPO_ID = os.environ.get("DATASET_REPO_ID", "yongtae-jp/survive-or-die")
TRAIN_FILE  = Path("./data/train.jsonl")
VAL_FILE    = Path("./data/val.jsonl")

print(f"Model:      {MODEL_REPO_ID}")
print(f"Output:     {OUTPUT_REPO_ID or '(no push)'}")
print(f"4bit:       {LOAD_IN_4BIT}  epochs:{NUM_EPOCHS}  batch:{BATCH_SIZE}  accum:{GRAD_ACCUM}")
if MAX_TRAIN_SAMPLES: print(f"[SMOKE] MAX_TRAIN_SAMPLES={MAX_TRAIN_SAMPLES}")
if MAX_STEPS:         print(f"[SMOKE] MAX_STEPS={MAX_STEPS}")

# ── 3. HF Hub login ───────────────────────────────────────────────────────────
_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    from huggingface_hub import login as hf_login
    hf_login(token=_hf_token, add_to_git_credential=False)
    print("HF Hub: logged in")

# ── 4. WandB Finetune init ────────────────────────────────────────────────────
if SKIP_WANDB:
    print("WandB disabled (SKIP_WANDB=true).")
    os.environ["WANDB_DISABLED"] = "true"

# WandB init is deferred until after data is loaded so we have accurate sample counts

# ── 5. Data loading ───────────────────────────────────────────────────────────
def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


print("\nDownloading data from HF Hub...")
from huggingface_hub import hf_hub_download
for filename in ["train.jsonl", "val.jsonl"]:
    local_path = Path("./data") / filename
    if not local_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=DATASET_REPO_ID,
            filename=filename,        # uploaded to repo root (not data/ subdir)
            repo_type="dataset",
            local_dir="./data",       # save to ./data/train.jsonl
            token=_hf_token,
        )
        print(f"  Downloaded {filename}")
    else:
        print(f"  {filename} already exists, skipping download")

train_records = load_jsonl(TRAIN_FILE)
val_records   = load_jsonl(VAL_FILE)
if MAX_TRAIN_SAMPLES:
    train_records = train_records[:MAX_TRAIN_SAMPLES]
    val_records   = val_records[:max(2, MAX_TRAIN_SAMPLES // 5)]
print(f"  train={len(train_records)}, val={len(val_records)}")

# ── 6. WandB init (now that we have sample counts) ────────────────────────────
if not SKIP_WANDB:
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    wandb.init(
        project="SurviveOrDie",
        name=WANDB_RUN_NAME,
        config={
            "model":         MODEL_REPO_ID,
            "method":        "LoRA",
            "r":             32,
            "lora_alpha":    32,
            "load_in_4bit":  LOAD_IN_4BIT,
            "epochs":        NUM_EPOCHS,
            "learning_rate": 2e-4,
            "batch_size":    BATCH_SIZE,
            "grad_accum":    GRAD_ACCUM,
            "train_samples": len(train_records),
            "val_samples":   len(val_records),
            "task":          "survival_species_identification",
            "attn_impl":     ATTN_IMPL,
        },
        tags=["mistral-hackathon", "survival-ai", "vision-finetune"],
    )
    # Log dataset as Artifact (version tracking)
    ds_artifact = wandb.Artifact(
        name="survive-or-die-dataset",
        type="dataset",
        description="Wilderness survival species identification dataset (5 mushroom species, 5 patterns)",
        metadata={
            "train_samples": len(train_records),
            "val_samples":   len(val_records),
            "hf_repo":       DATASET_REPO_ID,
            "species":       ["Amanita phalloides", "Amanita muscaria",
                              "Flammulina velutipes", "Lentinula edodes", "Hericium erinaceus"],
            "patterns":      ["high_confidence", "medium_confidence", "low_confidence",
                              "emergency", "foraging", "unknown"],
        },
    )
    ds_artifact.add_file(str(TRAIN_FILE))
    ds_artifact.add_file(str(VAL_FILE))
    wandb.log_artifact(ds_artifact)
    print("W&B Artifact: dataset logged")

# ── 7. Tool definitions ───────────────────────────────────────────────────────
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "species_db_lookup",
            "description": "Look up safety information for a species from the survival database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "species_guess": {
                        "type": "string",
                        "description": "Scientific name of the species (e.g. 'Amanita phalloides')",
                    },
                    "category": {
                        "type": "string",
                        "enum": ["mushroom", "plant"],
                        "description": "Whether this is a mushroom or plant",
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Confidence level in the identification",
                    },
                },
                "required": ["species_guess", "category", "confidence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "emergency_protocol",
            "description": "Trigger emergency protocol when user has already ingested a potentially toxic species.",
            "parameters": {
                "type": "object",
                "properties": {
                    "species_guess": {
                        "type": "string",
                        "description": "Scientific name or best guess of the ingested species",
                    },
                    "category": {
                        "type": "string",
                        "description": "Whether this is a mushroom or plant",
                    },
                    "time_since_ingestion": {
                        "type": "string",
                        "description": "Estimated time since ingestion (e.g. '30 minutes', 'unknown')",
                    },
                },
                "required": ["species_guess", "category", "time_since_ingestion"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nearby_species_search",
            "description": "Search for edible and dangerous species near a given location and season.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {
                        "type": "number",
                        "description": "Latitude of the current location",
                    },
                    "month": {
                        "type": "integer",
                        "description": "Current month (1-12)",
                    },
                    "altitude_m": {
                        "type": "integer",
                        "description": "Altitude in meters",
                    },
                    "environment": {
                        "type": "string",
                        "description": "Environment type (e.g. 'coniferous_forest', 'meadow', 'broadleaf_forest')",
                    },
                },
                "required": ["latitude", "month", "altitude_m", "environment"],
            },
        },
    },
]

# ── 8. Data preprocessing ─────────────────────────────────────────────────────
def base64_to_pil(data_uri: str) -> Image.Image:
    _, data = data_uri.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


import re as _re


def decode_messages(messages: list[dict]) -> dict:
    """
    Convert train.jsonl format → Mistral function calling format.

    Extracts PIL images from user message (replacing image_url with image placeholder).
    Preserves actual tool_calls from training data as-is — no reconstruction.
    Tool returns are already simplified in the training data.
    """
    images: list[Image.Image] = []
    chat_messages = []

    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "user":
            new_content = []
            for part in (content if isinstance(content, list) else []):
                if part.get("type") == "image_url":
                    images.append(base64_to_pil(part["image_url"]["url"]))
                    new_content.append({"type": "image"})
                else:
                    new_content.append(part)
            chat_messages.append({"role": "user", "content": new_content})

        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                chat_messages.append({"role": "assistant", "tool_calls": tool_calls})
            elif content:
                chat_messages.append({"role": "assistant", "content": content})

        elif role == "tool":
            chat_messages.append({"role": "tool", "content": content})

    return {"images": images, "messages": chat_messages}


class SurvivalDataset(TorchDataset):
    """PyTorch Dataset holding PIL images in memory (Arrow can't serialize PIL)."""
    def __init__(self, samples: list[dict]):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]


def to_dataset(records: list[dict]) -> SurvivalDataset:
    samples = [decode_messages(rec["messages"]) for rec in records]
    return SurvivalDataset(samples)


print("Preprocessing (decoding base64 images → PIL)...")
train_dataset = to_dataset(train_records)
val_dataset   = to_dataset(val_records)

# ── 8.5. Mini species DB + Quality eval callback ──────────────────────────────
# Build species lookup from embedded tool responses (no extra download needed)
_mini_species_db: dict = {}
for _rec in train_records + val_records:
    for _msg in _rec.get("messages", []):
        if _msg.get("role") == "tool":
            try:
                _d = json.loads(_msg["content"])
                _sp = _d.get("species", "")
                if _sp and _sp not in _mini_species_db:
                    _mini_species_db[_sp] = _d
            except Exception:
                pass
print(f"  mini_species_db: {len(_mini_species_db)} entries for quality eval")


class QualityEvalCallback(TrainerCallback):
    """
    End-to-end quality eval at each epoch.
    Runs full 2-step pipeline (image→tool_call→tool_response→answer)
    on a subset of val_records, then logs 4 metrics to W&B:
      quality/tool_fire_rate  — % of samples that fired a tool call
      quality/safety          — toxic species warned (1.0) vs missed (0.0)
      quality/completeness    — answer contains species name, toxicity, advice
      quality/critical_misses — count of toxic-called-safe (catastrophic)
    """
    N_EVAL = 20  # samples per epoch eval (keep fast)

    def __init__(self, val_records, processor, species_db):
        import random
        rng = random.Random(42)
        self.samples = rng.sample(val_records, min(self.N_EVAL, len(val_records)))
        self.processor = processor
        self.species_db = species_db

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        model.eval()
        try:
            metrics = self._run_eval(model)
            msg = "  ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in metrics.items()
            )
            print(f"\n[QualityEval epoch={state.epoch:.1f}]  {msg}")
            if not SKIP_WANDB:
                wandb.log({f"quality/{k}": v for k, v in metrics.items()},
                          step=state.global_step)
        except Exception as e:
            print(f"[QualityEval] failed: {e}")
        finally:
            model.train()

    # ── internal helpers ────────────────────────────────────────────────────

    def _run_eval(self, model):
        tool_fires, safety_scores, complete_scores, critical_misses = 0, [], [], 0
        for rec in self.samples:
            try:
                result = self._pipeline(model, rec)
                gt = self._ground_truth(rec)
                if not gt:
                    continue
                if result["tool_fired"]:
                    tool_fires += 1
                s = self._safety(result["answer"], gt)
                safety_scores.append(s["score"])
                if s["critical_miss"]:
                    critical_misses += 1
                complete_scores.append(self._completeness(result["answer"], gt))
            except Exception as e:
                print(f"  [quality_eval sample] {e}")
        n = len(self.samples)
        return {
            "tool_fire_rate":  tool_fires / n if n else 0.0,
            "safety":          sum(safety_scores) / len(safety_scores) if safety_scores else 0.0,
            "completeness":    sum(complete_scores) / len(complete_scores) if complete_scores else 0.0,
            "critical_misses": critical_misses,
        }

    def _pipeline(self, model, rec):
        """2-step inference: user→tool_call, then tool_response→final_answer."""
        # Decode user turn
        images, user_content = [], []
        for part in rec["messages"][0].get("content", []):
            if part.get("type") == "image_url":
                images.append(base64_to_pil(part["image_url"]["url"]))
                user_content.append({"type": "image"})
            else:
                user_content.append(part)
        step1 = [{"role": "user", "content": user_content}]

        # Step 1: generate tool call
        t1 = self.processor.apply_chat_template(
            step1, tools=TOOL_DEFINITIONS, add_generation_prompt=True, tokenize=False)
        b1 = self.processor(text=t1, images=images or None,
                            return_tensors="pt")
        b1 = {k: v.to(model.device).to(model.dtype) if v.is_floating_point() else v.to(model.device) for k, v in b1.items()}
        with torch.no_grad():
            o1 = model.generate(**b1, max_new_tokens=128, do_sample=False, use_cache=True)
        s1 = self.processor.tokenizer.decode(
            o1[0][b1["input_ids"].shape[1]:], skip_special_tokens=False)

        tool_calls = self._parse_tool_calls(s1)
        if not tool_calls:
            return {"tool_fired": False, "answer": s1.strip()}

        # Execute tool
        tool_responses = [self._exec_tool(tc) for tc in tool_calls]

        # Step 2: generate final answer
        step2 = step1 + [
            {"role": "assistant", "tool_calls": tool_calls},
            *[{"role": "tool", "content": json.dumps(tr, ensure_ascii=False)}
              for tr in tool_responses],
        ]
        t2 = self.processor.apply_chat_template(
            step2, tools=TOOL_DEFINITIONS, add_generation_prompt=True, tokenize=False)
        b2 = self.processor(text=t2, images=images or None,
                            return_tensors="pt")
        b2 = {k: v.to(model.device).to(model.dtype) if v.is_floating_point() else v.to(model.device) for k, v in b2.items()}
        with torch.no_grad():
            o2 = model.generate(**b2, max_new_tokens=256, do_sample=False, use_cache=True)
        answer = self.processor.tokenizer.decode(
            o2[0][b2["input_ids"].shape[1]:], skip_special_tokens=True)
        return {"tool_fired": True, "answer": answer.strip()}

    def _exec_tool(self, tc):
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments", "{}"))
        except Exception:
            return {"error": "parse_error"}
        name = fn.get("name", "")
        if name == "species_db_lookup":
            sp = args.get("species_guess", "")
            entry = self.species_db.get(sp) or next(
                (v for k, v in self.species_db.items() if k.lower() == sp.lower()), None)
            if not entry:
                return {"error": "not_found", "queried": sp}
            return {
                "species": sp,
                "edibility": entry.get("edibility", "unknown"),
                "toxins": entry.get("toxins", []),
                "season_match": True,
                "altitude_match": True,
                "habitat_match": True,
            }
        if name == "emergency_protocol":
            return {"severity": "LETHAL", "species": args.get("species_guess", ""), "toxins": []}
        if name == "nearby_species_search":
            return {"edible_count": 0, "dangerous_count": 0, "edible_species": [], "dangerous_nearby": []}
        return {"error": f"unknown_tool: {name}"}

    def _parse_tool_calls(self, text):
        m = _re.search(r'\[TOOL_CALLS\]\s*(\[.*?\])', text, _re.DOTALL)
        if m:
            try:
                raw = json.loads(m.group(1))
                return [
                    {"function": {"name": x["name"],
                                  "arguments": json.dumps(x.get("arguments", {}))}}
                    if "name" in x else x
                    for x in raw
                ]
            except Exception:
                pass
        return []

    def _ground_truth(self, rec):
        for msg in rec["messages"]:
            if msg.get("role") == "tool":
                try:
                    return json.loads(msg["content"])
                except Exception:
                    pass
        return {}

    def _safety(self, answer, gt):
        a = answer.lower()
        verdict = str(gt.get("survival_verdict", gt.get("severity", gt.get("edibility", "")))).upper()
        WARN = ["do not eat", "don't eat", "lethal", "poisonous", "toxic",
                "dangerous", "食べないで", "危険", "毒"]
        SAFE = ["safe to eat", "edible", "食べられる", "食用"]
        if verdict in ("LETHAL", "DO_NOT_EAT"):
            warned = any(k in a for k in WARN)
            return {"score": 1.0 if warned else 0.0, "critical_miss": not warned}
        if verdict == "SAFE":
            return {"score": 1.0 if any(k in a for k in SAFE) else 0.5, "critical_miss": False}
        return {"score": 0.5, "critical_miss": False}

    def _completeness(self, answer, gt):
        a = answer.lower()
        verdict = str(gt.get("survival_verdict", gt.get("edibility", ""))).upper()
        species = gt.get("species", "").lower()
        checks = [
            bool(species and species in a),
            any(w in a for w in ["toxic", "edible", "poisonous", "safe", "lethal", "毒", "食用", "危険"]),
            len(answer.strip()) > 80,
        ]
        if verdict in ("LETHAL", "DO_NOT_EAT"):
            checks.append(any(k in a for k in ["hospital", "emergency", "119", "救急", "医者"]))
        return sum(checks) / len(checks)


# ── 9. Model + LoRA ───────────────────────────────────────────────────────────
print(f"\nLoading {MODEL_REPO_ID}...")

quant_kwargs = {}
if LOAD_IN_4BIT:
    quant_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_REPO_ID,
    device_map="auto",
    attn_implementation=ATTN_IMPL,
    **quant_kwargs,
)
processor = AutoProcessor.from_pretrained(MODEL_REPO_ID)
processor.tokenizer.padding_side = "right"

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
# NOTE: do NOT call get_peft_model() here — SFTTrainer applies LoRA internally
# via peft_config=peft_config below.  Wrapping manually first causes
# SFTTrainer to call prepare_model_for_kbit_training() a second time,
# which freezes the LoRA adapters → grad_norm=0, flat loss.

if torch.cuda.is_available():
    print(f"VRAM after base model load: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# ── 10. Collator ──────────────────────────────────────────────────────────────
tokenizer = processor.tokenizer
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
pad_token_id   = tokenizer.pad_token_id
image_token_id = getattr(processor, "image_token_id", None)


def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    all_images: list = []
    conversations: list = []
    for ex in examples:
        all_images.extend(ex["images"])
        conversations.append(ex["messages"])

    # Full conversation text
    chat_texts = processor.apply_chat_template(
        conversations,
        tools=TOOL_DEFINITIONS,
        add_generation_prompt=False,
        tokenize=False,
    )

    # Prompt-only text (user turn only) for computing how much to mask.
    # We mask the user prompt so loss is only computed on assistant responses.
    prompt_texts = processor.apply_chat_template(
        [[conv[0]] for conv in conversations],  # just user turns
        tools=TOOL_DEFINITIONS,
        add_generation_prompt=True,
        tokenize=False,
    )

    batch = processor(
        text=chat_texts,
        images=all_images if all_images else None,
        padding="longest",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="pt",
    )

    # Tokenize prompts (text only, no images) to get prefix lengths
    prompt_ids = processor.tokenizer(
        prompt_texts,
        add_special_tokens=False,
    )["input_ids"]

    labels = batch["input_ids"].clone()

    # Mask user prompt prefix (loss only on assistant turns)
    for i, prompt_ids_row in enumerate(prompt_ids):
        prompt_len = len(prompt_ids_row)
        labels[i, :prompt_len] = -100

    # Mask tool return content (model should not learn to generate tool results)
    for i, conv in enumerate(conversations):
        tool_content = None
        for msg in conv:
            if msg.get("role") == "tool":
                tool_content = msg.get("content", "")
                break
        if tool_content:
            full_text = chat_texts[i] if isinstance(chat_texts, list) else chat_texts
            pos = full_text.find(tool_content)
            if pos >= 0:
                before_ids = processor.tokenizer(
                    full_text[:pos], add_special_tokens=False)["input_ids"]
                after_ids = processor.tokenizer(
                    full_text[:pos + len(tool_content)], add_special_tokens=False)["input_ids"]
                labels[i, len(before_ids):len(after_ids)] = -100

    # Mask padding and image tokens
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100
    if image_token_id is not None and image_token_id != pad_token_id:
        labels[labels == image_token_id] = -100

    batch["labels"] = labels
    return batch


# ── 11. Train ─────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    peft_config=peft_config,   # SFTTrainer applies LoRA + prepare_model_for_kbit_training internally
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    processing_class=processor,
    args=SFTConfig(
        output_dir="./outputs",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_ratio=0.03,
        lr_scheduler_type="linear",
        num_train_epochs=NUM_EPOCHS,
        learning_rate=2e-4,
        max_grad_norm=0.3,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_steps=MAX_STEPS if MAX_STEPS else -1,
        dataloader_pin_memory=True,
        report_to="none" if SKIP_WANDB else "wandb",
        run_name=WANDB_RUN_NAME,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
    ),
)

trainer.add_callback(QualityEvalCallback(val_records, processor, _mini_species_db))

print("\n" + "=" * 60)
print("Training...")
print("=" * 60)
trainer.train()

# ── 12. Save LoRA adapter ─────────────────────────────────────────────────────
lora_out = Path("./outputs/survive_or_die_lora")
trainer.model.save_pretrained(lora_out)  # trainer.model is the PeftModel
processor.save_pretrained(lora_out)
print(f"\nLoRA saved → {lora_out}")

# Log LoRA model as Artifact
if not SKIP_WANDB:
    model_artifact = wandb.Artifact(
        name="survive-or-die-lora",
        type="model",
        description="LoRA adapter fine-tuned on wilderness survival species identification",
        metadata={
            "base_model":    MODEL_REPO_ID,
            "lora_r":        32,
            "lora_alpha":    32,
            "load_in_4bit":  LOAD_IN_4BIT,
            "train_samples": len(train_records),
            "epochs":        NUM_EPOCHS,
            "hf_repo":       OUTPUT_REPO_ID or "not pushed",
        },
    )
    model_artifact.add_dir(str(lora_out))
    wandb.log_artifact(model_artifact)
    print("W&B Artifact: model logged")
    wandb.finish()

# ── 13. Push to HuggingFace Hub ───────────────────────────────────────────────
if SKIP_PUSH or not OUTPUT_REPO_ID:
    print(f"\nSkipping Hub push ({'SKIP_PUSH=true' if SKIP_PUSH else 'OUTPUT_REPO_ID not set'}).")
else:
    token = os.environ.get("HF_TOKEN")
    print(f"\nPushing LoRA → {OUTPUT_REPO_ID}")
    trainer.model.push_to_hub(OUTPUT_REPO_ID, token=token)
    processor.push_to_hub(OUTPUT_REPO_ID, token=token)
    print(f"Pushed → {OUTPUT_REPO_ID}")

print("\n" + "=" * 60)
print("Training complete.")
print(f"  WandB Finetune: https://wandb.ai/yongtae/SurviveOrDie")
print(f"  Next: run eval.py for WandB Weave end-to-end evaluation")
print("=" * 60)
