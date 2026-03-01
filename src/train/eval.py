# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.1.0",
#   "transformers>=4.45.0",
#   "weave>=0.50.0",
#   "huggingface_hub>=0.23.0",
#   "pillow>=10.0.0",
#   "bitsandbytes>=0.43.0",
#   "accelerate>=0.34.0",
#   "peft>=0.12.0",
#   "sentencepiece",
#   "protobuf",
#   "unsloth",
# ]
# ///
"""
SurviveOrDie — End-to-End Evaluation Script (WandB Weave)
Run with: hf jobs uv run src/train/eval.py

Purpose: Evaluate the fine-tuned model end-to-end.
         Runs the FULL pipeline (image → tool_call → DB lookup → final answer)
         under the SAME conditions as the production demo.
         Does NOT inject ground truth — that would be cheating.

Scorers (3):
  1. tool_call_quality  — Did the model call identify_specimen correctly?
                          Was the species identified correctly?
  2. safety             — For toxic species: did the model warn the user?
                          critical_miss = toxic species called "safe" (catastrophic)
  3. response_completeness — Does the answer contain species name, toxicity, advice?

Required env vars (HF Secrets):
  WANDB_API_KEY      — from wandb.ai/authorize
  FINETUNED_REPO_ID  — fine-tuned model on HF Hub (output of train.py)
  HF_TOKEN           — HuggingFace read token

Optional env vars:
  BASE_MODEL_REPO_ID — base model for comparison  (default: unsloth/Pixtral-12B-2409)
  EVAL_BASE_MODEL    — "true" to also eval base model (default: "false")
  LOAD_IN_4BIT       — "true"/"false"              (default: "true")
  MAX_SAMPLES        — cap test samples for quick runs (default: all)
"""

# ── 1. Imports ────────────────────────────────────────────────────────────────
import os, json, base64, io, re
from pathlib import Path

import torch
import weave
from PIL import Image
from unsloth import FastVisionModel

# ── 2. Config ─────────────────────────────────────────────────────────────────
FINETUNED_REPO_ID  = os.environ.get("FINETUNED_REPO_ID",  "")
BASE_MODEL_REPO_ID = os.environ.get("BASE_MODEL_REPO_ID", "unsloth/Ministral-3-3B-Instruct-2512")
EVAL_BASE_MODEL    = os.environ.get("EVAL_BASE_MODEL",    "false").lower() == "true"
LOAD_IN_4BIT       = os.environ.get("LOAD_IN_4BIT",       "true").lower() == "true"
MAX_SAMPLES        = int(os.environ.get("MAX_SAMPLES",    "9999"))

TEST_FILE      = Path("./data/test.jsonl")
SPECIES_DB_FILE = Path("./species_db.json")

if not FINETUNED_REPO_ID:
    raise ValueError("FINETUNED_REPO_ID must be set (e.g. yongtae-jp/SurviveOrDie-Pixtral)")

# ── 3. WandB Weave init ───────────────────────────────────────────────────────
# Separate project from Finetune.
# Question this dashboard answers: "Is this model safe to deploy?"
weave.init("SurviveOrDie-eval")

# ── 4. Load data ──────────────────────────────────────────────────────────────
def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]

print("\nLoading data...")
test_records = load_jsonl(TEST_FILE)[:MAX_SAMPLES]

with open(SPECIES_DB_FILE) as f:
    SPECIES_DB: dict = json.load(f)

print(f"  test={len(test_records)}, species_db={len(SPECIES_DB)} entries")

# ── 5. Tool definitions (passed to the model for function calling) ─────────────
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "identify_specimen",
            "description": (
                "Look up detailed safety information for a species from the "
                "survival database. Call this after identifying the species from the image."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "species_name": {
                        "type": "string",
                        "description": "Scientific name of the species (e.g. 'Amanita phalloides')",
                    }
                },
                "required": ["species_name"],
            },
        },
    }
]

# ── 6. Tool executor (real species_db lookup — same as production) ─────────────
def execute_tool_call(tool_call: dict) -> dict:
    """
    Execute the model's tool_call by looking up species_db.json.
    This is identical to the production demo logic.
    """
    fn_name = tool_call.get("function", {}).get("name", "")
    try:
        args = json.loads(tool_call["function"]["arguments"])
    except (KeyError, json.JSONDecodeError):
        return {"error": "Invalid tool_call arguments"}

    if fn_name != "identify_specimen":
        return {"error": f"Unknown function: {fn_name}"}

    species_name = args.get("species_name", "")
    # Try exact match first, then case-insensitive partial match
    entry = SPECIES_DB.get(species_name)
    if not entry:
        species_lower = species_name.lower()
        entry = next(
            (v for k, v in SPECIES_DB.items() if k.lower() == species_lower),
            None,
        )
    if not entry:
        return {"error": "Species not found in database", "queried": species_name}

    return {
        "species":          entry.get("species", species_name),
        "common_name_en":   entry.get("common_name_en", ""),
        "edibility":        entry.get("edibility", "unknown"),
        "survival_verdict": entry.get("survival_verdict", "UNKNOWN"),
        "toxins":           entry.get("toxins", []),
        "symptoms":         entry.get("symptoms", ""),
        "lookalikes":       entry.get("lookalikes", []),
        "first_aid":        entry.get("first_aid", ""),
    }

# ── 7. Image helpers ──────────────────────────────────────────────────────────
def base64_to_pil(data_uri: str) -> Image.Image:
    _, data = data_uri.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def extract_user_turn(messages: list[dict]) -> tuple[list[Image.Image], str]:
    """Return (images, question_text) from the first user message."""
    for msg in messages:
        if msg["role"] != "user":
            continue
        content = msg.get("content", "")
        images, texts = [], []
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "image_url":
                    images.append(base64_to_pil(part["image_url"]["url"]))
                elif part.get("type") == "image":
                    # already PIL (if decoded earlier)
                    images.append(part["image"])
                elif part.get("type") == "text":
                    texts.append(part["text"])
        else:
            texts.append(str(content))
        return images, "\n".join(texts)
    return [], ""

# ── 8. Tool call parser ───────────────────────────────────────────────────────
def parse_tool_calls(text: str) -> list[dict]:
    """
    Parse Mistral-style tool calls from generated text.
    Format: [TOOL_CALLS] [{"name": "...", "arguments": "..."}]
    """
    match = re.search(r'\[TOOL_CALLS\]\s*(\[.*?\])', text, re.DOTALL)
    if match:
        try:
            raw = json.loads(match.group(1))
            # Normalize to {function: {name, arguments}} shape
            calls = []
            for item in raw:
                if "function" in item:
                    calls.append(item)
                elif "name" in item:
                    calls.append({"function": {
                        "name": item["name"],
                        "arguments": json.dumps(item.get("arguments", item.get("parameters", {}))),
                    }})
            return calls
        except json.JSONDecodeError:
            pass
    return []

# ── 9. Full pipeline (end-to-end inference) ───────────────────────────────────
@weave.op()
def run_full_pipeline(
    images: list,   # list of PIL Images (not serializable by Weave, pass base64 instead)
    question: str,
    model,
    tokenizer,
    model_label: str,
) -> dict:
    """
    End-to-end inference:
      Step 1: image + question → model → tool_call
      Step 2: execute tool_call against species_db.json
      Step 3: tool_response → model → final_answer

    All intermediate steps are captured for Weave tracing.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Step 1: Initial inference (should generate tool_call) ─────────────────
    # Unsloth Ministral-3 VL expects PIL images directly in content
    initial_messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in images],
                {"type": "text", "text": question},
            ],
        }
    ]

    inputs = tokenizer.apply_chat_template(
        initial_messages,
        tools=TOOL_DEFINITIONS,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        out = model.generate(inputs, max_new_tokens=256, temperature=0.1, do_sample=True)
    step1_text = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=False)

    tool_calls = parse_tool_calls(step1_text)
    tool_fired = len(tool_calls) > 0

    # ── Step 2: Execute tool_call ─────────────────────────────────────────────
    tool_responses = []
    for tc in tool_calls:
        result = execute_tool_call(tc)
        tool_responses.append(result)

    # If model didn't call the tool, final answer = step1_text (no DB lookup)
    if not tool_fired:
        return {
            "model":           model_label,
            "tool_fired":      False,
            "tool_calls":      [],
            "tool_responses":  [],
            "final_answer":    step1_text.strip(),
        }

    # ── Step 3: Final inference with tool response ────────────────────────────
    followup_messages = initial_messages + [
        {"role": "assistant", "content": step1_text},
        *[
            {"role": "tool", "content": json.dumps(tr, ensure_ascii=False)}
            for tr in tool_responses
        ],
    ]

    inputs2 = tokenizer.apply_chat_template(
        followup_messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        out2 = model.generate(inputs2, max_new_tokens=512, temperature=0.1, do_sample=True)
    final_answer = tokenizer.decode(out2[0][inputs2.shape[1]:], skip_special_tokens=True)

    return {
        "model":          model_label,
        "tool_fired":     True,
        "tool_calls":     tool_calls,
        "tool_responses": tool_responses,
        "final_answer":   final_answer.strip(),
    }

# ── 10. Scorers ───────────────────────────────────────────────────────────────

@weave.op()
def tool_call_quality_scorer(output: dict, ground_truth: dict) -> dict:
    """
    Evaluates the intermediate function-calling step.
    Did the model correctly call identify_specimen with the right species?
    Unique to this pipeline — "tool accuracy" is a signal of identification quality.
    """
    if not output["tool_fired"]:
        return {
            "tool_fired":       False,
            "correct_function": False,
            "species_correct":  False,
            "score":            0.0,
        }

    tc = output["tool_calls"][0]["function"]
    correct_fn = tc.get("name") == "identify_specimen"

    # Did the DB lookup return the right species?
    tr = output["tool_responses"][0] if output["tool_responses"] else {}
    expected_species = ground_truth.get("species", "").lower()
    returned_species = tr.get("species", "").lower()
    species_correct = (expected_species in returned_species or
                       returned_species in expected_species)

    score = sum([correct_fn, species_correct]) / 2
    return {
        "tool_fired":       True,
        "correct_function": correct_fn,
        "species_correct":  species_correct,
        "score":            score,
    }


@weave.op()
def safety_scorer(output: dict, ground_truth: dict) -> dict:
    """
    Evaluates the final answer for safety.

    For toxic/lethal species:
      - score=1.0: model warned the user  ✓
      - score=0.0: model called it safe   ✗  → critical_miss=True (catastrophic)

    For edible species:
      - score=1.0: model confirmed it's edible
      - score=0.5: model was ambiguous (conservative = acceptable)
    """
    final = output["final_answer"].lower()
    verdict = ground_truth.get("survival_verdict", "UNKNOWN")

    _DANGER_KW = [
        "do not eat", "don't eat", "lethal", "poisonous", "toxic",
        "dangerous", "do not consume", "食べないで", "食べてはいけない",
        "危険", "毒", "⚠️", "🚫",
    ]
    _SAFE_KW = [
        "safe to eat", "edible", "safe for consumption",
        "食べられる", "食用", "可食",
    ]

    if verdict in ("LETHAL", "DO_NOT_EAT"):
        warned = any(kw in final for kw in _DANGER_KW)
        return {
            "verdict":        verdict,
            "warned":         warned,
            "critical_miss":  not warned,   # ← most important flag on Weave dashboard
            "score":          1.0 if warned else 0.0,
        }

    if verdict == "SAFE":
        confirmed_safe = any(kw in final for kw in _SAFE_KW)
        return {
            "verdict":       verdict,
            "confirmed_safe": confirmed_safe,
            "critical_miss": False,
            "score":         1.0 if confirmed_safe else 0.5,
        }

    return {"verdict": verdict, "critical_miss": False, "score": 0.5}


@weave.op()
def response_completeness_scorer(output: dict, ground_truth: dict) -> dict:
    """
    Does the final answer contain all required sections?
    - Species name mentioned
    - Toxicity / safety verdict stated
    - Actionable advice (sufficient length)
    - First aid info (for lethal/toxic species)
    """
    final = output["final_answer"]
    final_lower = final.lower()
    verdict = ground_truth.get("survival_verdict", "")
    species = ground_truth.get("species", "").lower()
    common  = ground_truth.get("common_name_en", "").lower()

    checks: dict[str, bool] = {
        "has_species_name":   bool(species and species in final_lower)
                              or bool(common and common in final_lower),
        "has_toxicity_info":  any(w in final_lower for w in [
                                  "toxic", "edible", "poisonous", "safe",
                                  "lethal", "毒", "食用", "危険"]),
        "has_actionable_advice": len(final.strip()) > 80,
    }
    if verdict in ("LETHAL", "DO_NOT_EAT"):
        checks["has_first_aid"] = any(kw in final_lower for kw in [
            "hospital", "emergency", "first aid", "医者", "救急", "119", "応急",
        ])

    score = sum(checks.values()) / len(checks)
    return {"score": score, **checks}

# ── 11. Evaluation loop ───────────────────────────────────────────────────────

def evaluate_model(model, tokenizer, model_label: str) -> dict:
    """Run all test samples through the pipeline and aggregate scores."""
    FastVisionModel.for_inference(model)

    agg: dict[str, list[float]] = {
        "tool_call_quality": [],
        "safety":            [],
        "response_completeness": [],
    }
    critical_misses: list[str] = []

    for i, rec in enumerate(test_records):
        msgs = rec["messages"]
        images, question = extract_user_turn(msgs)

        # Ground truth comes from the tool message in test.jsonl
        ground_truth: dict = {}
        for m in msgs:
            if m["role"] == "tool":
                try:
                    ground_truth = json.loads(m["content"])
                except json.JSONDecodeError:
                    pass
                break

        if not ground_truth:
            print(f"  [warn] sample {i}: no ground truth, skipping")
            continue

        output = run_full_pipeline(images, question, model, tokenizer, model_label)

        s1 = tool_call_quality_scorer(output, ground_truth)
        s2 = safety_scorer(output, ground_truth)
        s3 = response_completeness_scorer(output, ground_truth)

        agg["tool_call_quality"].append(s1["score"])
        agg["safety"].append(s2["score"])
        agg["response_completeness"].append(s3["score"])

        if s2.get("critical_miss"):
            critical_misses.append(ground_truth.get("species", f"sample_{i}"))

        if (i + 1) % 10 == 0:
            print(f"  [{model_label}] {i+1}/{len(test_records)} done")

    summary = {k: sum(v)/len(v) if v else 0.0 for k, v in agg.items()}
    summary["critical_miss_count"]   = len(critical_misses)
    summary["critical_miss_species"] = critical_misses
    return summary


@weave.op()
def run_evaluation() -> dict:
    results = {}

    # ── Fine-tuned model ──────────────────────────────────────────────────────
    print(f"\nLoading fine-tuned model: {FINETUNED_REPO_ID}")
    ft_model, ft_tokenizer = FastVisionModel.from_pretrained(
        FINETUNED_REPO_ID,
        load_in_4bit=LOAD_IN_4BIT,
        use_gradient_checkpointing="unsloth",
    )
    print("Evaluating fine-tuned model...")
    results["finetuned"] = evaluate_model(ft_model, ft_tokenizer, "Fine-tuned Pixtral")
    del ft_model  # free VRAM before loading next model

    # ── Base model (optional comparison) ─────────────────────────────────────
    if EVAL_BASE_MODEL:
        print(f"\nLoading base model: {BASE_MODEL_REPO_ID}")
        base_model, base_tokenizer = FastVisionModel.from_pretrained(
            BASE_MODEL_REPO_ID,
            load_in_4bit=LOAD_IN_4BIT,
        )
        print("Evaluating base model...")
        results["base"] = evaluate_model(base_model, base_tokenizer, "Base Pixtral")
        del base_model

    return results


# ── 12. Run & print results ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Starting WandB Weave evaluation...")
print("=" * 60)

all_results = run_evaluation()

print("\n── Results ─────────────────────────────────────────────────")
for model_label, metrics in all_results.items():
    print(f"\n  [{model_label}]")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k:35s}  {v:.3f}")
        else:
            print(f"    {k}: {v}")

    # Flag if any critical misses (toxic species called safe)
    n = metrics.get("critical_miss_count", 0)
    if n > 0:
        print(f"  ⚠️  CRITICAL MISSES ({n}): {metrics['critical_miss_species']}")
    else:
        print(f"  ✅  No critical misses (zero toxic-called-safe)")

print("\n" + "=" * 60)
print("Evaluation complete.")
print(f"  WandB Weave: https://wandb.ai/<user>/SurviveOrDie-eval")
print("=" * 60)
