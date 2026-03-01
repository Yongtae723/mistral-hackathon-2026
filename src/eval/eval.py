# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.1.0",
#   "transformers @ git+https://github.com/huggingface/transformers.git@bf3f0ae70d0e902efab4b8517fce88f6697636ce",
#   "peft>=0.12.0",
#   "weave>=0.51.0",
#   "wandb>=0.17.0",
#   "huggingface_hub>=0.34.0",
#   "pillow>=10.0.0",
#   "bitsandbytes>=0.43.0",
#   "accelerate>=0.34.0",
#   "sentencepiece",
#   "protobuf",
#   "google-generativeai>=0.8.0",
# ]
# ///
"""
SurviveOrDie — Evaluation Script

Compares 3 models on survival-critical metrics:
  1. Gemini Flash-Lite (cloud, web search grounding)
  2. Base Ministral-3B (offline, no fine-tuning)
  3. Fine-tuned Ministral-3B LoRA (offline, tool-calling)

Tracked via W&B Weave (per-prediction tracing) + W&B Tables (side-by-side comparison).

Run with:
  hf jobs uv run --flavor a10g-small \\
      --secrets-file ../../.env \\
      --env LORA_REPO_ID=yongtae-jp/survive-or-die-lora \\
      eval.py

Required env vars:
  WANDB_API_KEY, HF_TOKEN, GEMINI_API_KEY

Optional env vars:
  MODEL_REPO_ID      — base model (default: mistralai/Ministral-3-3B-Instruct-2512-BF16)
  LORA_REPO_ID       — fine-tuned LoRA (default: yongtae-jp/survive-or-die-lora)
  DATASET_REPO_ID    — test data (default: yongtae-jp/survive-or-die)
  NUM_EVAL_SAMPLES   — samples to evaluate (default: 30)
  MAX_NEW_TOKENS     — max tokens per response (default: 512)
"""

# ── 1. Imports ────────────────────────────────────────────────────────────────
import os, json, base64, io, asyncio
from pathlib import Path

import torch
import weave
import wandb
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import hf_hub_download
import google.generativeai as genai

# ── 2. Config ─────────────────────────────────────────────────────────────────
BASE_MODEL_ID    = os.environ.get("MODEL_REPO_ID",   "mistralai/Ministral-3-3B-Instruct-2512-BF16")
LORA_REPO_ID     = os.environ.get("LORA_REPO_ID",    "yongtae-jp/survive-or-die-lora")
DATASET_REPO_ID  = os.environ.get("DATASET_REPO_ID", "yongtae-jp/survive-or-die")
NUM_EVAL_SAMPLES = int(os.environ.get("NUM_EVAL_SAMPLES", "30"))
MAX_NEW_TOKENS   = int(os.environ.get("MAX_NEW_TOKENS",   "512"))
GEMINI_MODEL     = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")

print(f"Base model:  {BASE_MODEL_ID}")
print(f"LoRA:        {LORA_REPO_ID}")
print(f"Gemini:      {GEMINI_MODEL}")
print(f"Eval samples: {NUM_EVAL_SAMPLES}")

# ── 3. Auth ───────────────────────────────────────────────────────────────────
_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    from huggingface_hub import login as hf_login
    hf_login(token=_hf_token, add_to_git_credential=False)

wandb.login(key=os.environ.get("WANDB_API_KEY"))
wandb.init(project="SurviveOrDie", name="eval-3way-comparison", job_type="eval",
           tags=["evaluation", "weave", "model-comparison", "gemini-flash-lite"])
weave.init("SurviveOrDie")

# Gemini setup
_gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not _gemini_api_key:
    raise ValueError("GEMINI_API_KEY not set")
genai.configure(api_key=_gemini_api_key)
print("W&B + Weave + Gemini initialized")

# ── 4. Download data ──────────────────────────────────────────────────────────
print("\nDownloading test data...")
data_dir = Path("./data")
data_dir.mkdir(exist_ok=True)

for filename in ["test.jsonl", "species_db.json"]:
    local = data_dir / filename
    if not local.exists():
        hf_hub_download(repo_id=DATASET_REPO_ID, filename=filename,
                        repo_type="dataset", local_dir=str(data_dir), token=_hf_token)
        print(f"  Downloaded {filename}")
    else:
        print(f"  {filename} already exists")

with open(data_dir / "species_db.json", encoding="utf-8") as f:
    SPECIES_DB = json.load(f)

with open(data_dir / "test.jsonl", encoding="utf-8") as f:
    all_test = [json.loads(line) for line in f]

print(f"  test={len(all_test)} samples, species_db={len(SPECIES_DB)} species")

# ── 5. Tool definitions (same as train.py) ────────────────────────────────────
TOOL_DEFINITIONS = [
    {"type": "function", "function": {
        "name": "species_db_lookup",
        "description": "Look up safety information for a species from the survival database.",
        "parameters": {"type": "object", "required": ["species_guess", "category", "confidence"],
                       "properties": {
                           "species_guess": {"type": "string"},
                           "category":      {"type": "string", "enum": ["mushroom", "plant"]},
                           "confidence":    {"type": "string", "enum": ["high", "medium", "low"]},
                       }},
    }},
    {"type": "function", "function": {
        "name": "emergency_protocol",
        "description": "Trigger emergency protocol when user has already ingested a potentially toxic species.",
        "parameters": {"type": "object", "required": ["species_guess", "category", "time_since_ingestion"],
                       "properties": {
                           "species_guess":        {"type": "string"},
                           "category":             {"type": "string"},
                           "time_since_ingestion": {"type": "string"},
                       }},
    }},
    {"type": "function", "function": {
        "name": "nearby_species_search",
        "description": "Search for edible and dangerous species near a given location and season.",
        "parameters": {"type": "object", "required": ["latitude", "month", "altitude_m", "environment"],
                       "properties": {
                           "latitude":    {"type": "number"},
                           "month":       {"type": "integer"},
                           "altitude_m":  {"type": "integer"},
                           "environment": {"type": "string"},
                       }},
    }},
]

# ── 6. Tool execution (simulate tool responses locally) ───────────────────────
def _find_species(guess: str) -> dict:
    """Fuzzy-match species name in species_db."""
    guess_lower = guess.lower()
    for name, data in SPECIES_DB.items():
        if guess_lower in name.lower() or name.lower() in guess_lower:
            return data
    return {}


def execute_tool(name: str, arguments: dict) -> str:
    if name == "species_db_lookup":
        data = _find_species(arguments.get("species_guess", ""))
        if not data:
            return json.dumps({"error": "not_found", "survival_verdict": "UNKNOWN_DO_NOT_EAT",
                                "survival_note": "Species not in database — DO NOT EAT."})
        return json.dumps({k: data[k] for k in
                           ["scientific_name", "survival_verdict", "survival_note",
                            "toxins", "symptoms", "first_aid", "lookalikes"] if k in data},
                          ensure_ascii=False)

    elif name == "emergency_protocol":
        data = _find_species(arguments.get("species_guess", ""))
        verdict = data.get("survival_verdict", "UNKNOWN") if data else "UNKNOWN"
        return json.dumps({
            "severity": verdict,
            "species":  arguments.get("species_guess", "unknown"),
            "actions": [
                {"step": 1, "action": "Call emergency services NOW",
                 "numbers": {"JP": "119", "US": "911", "EU": "112"}, "urgency": "IMMEDIATE"},
                {"step": 2, "action": "Do NOT induce vomiting", "urgency": "CRITICAL"},
                {"step": 3, "action": "Note exact time of ingestion", "urgency": "HIGH"},
                {"step": 4, "action": "Go to nearest hospital immediately", "urgency": "IMMEDIATE"},
            ],
            "tell_doctor": f"Possible {arguments.get('species_guess', 'unknown')} ingestion",
            "toxins": data.get("toxins", []) if data else [],
        }, ensure_ascii=False)

    elif name == "nearby_species_search":
        lat   = arguments.get("latitude", 0)
        month = arguments.get("month", 1)
        edible, dangerous = [], []
        for sp_name, data in SPECIES_DB.items():
            lat_range = data.get("distribution_lat_range", [-90, 90])
            seasons   = data.get("season_months", list(range(1, 13)))
            if lat_range[0] <= lat <= lat_range[1] and month in seasons:
                if data.get("survival_verdict") in ["SAFE", "CONDITIONAL_SAFE"]:
                    edible.append({"scientific_name": sp_name,
                                   "common_name_en": data.get("common_name_en", ""),
                                   "key_features": data.get("key_features", "")})
                elif data.get("survival_verdict") in ["LETHAL", "DO_NOT_EAT"]:
                    dangerous.append({"scientific_name": sp_name,
                                      "danger_level": data.get("survival_verdict", "")})
        return json.dumps({"edible_species": edible[:5], "dangerous_nearby": dangerous[:5],
                           "total_edible_found": len(edible)}, ensure_ascii=False)

    return json.dumps({"error": f"unknown tool: {name}"})


# ── 7. Eval dataset preparation ───────────────────────────────────────────────
def prepare_sample(record: dict) -> dict | None:
    """Extract eval-relevant fields from a test.jsonl record."""
    msgs = record.get("messages", [])
    if len(msgs) < 4:
        return None

    user_msg     = msgs[0]
    tool_call_msg = msgs[1]

    # Extract image + question
    image_b64, question = None, ""
    for part in (user_msg.get("content") or []):
        if part.get("type") == "image_url":
            image_b64 = part["image_url"]["url"]
        elif part.get("type") == "text":
            question = part["text"]

    if not image_b64 or not question:
        return None

    # Expected tool + species
    tc = (tool_call_msg.get("tool_calls") or [{}])[0]
    expected_tool = tc.get("function", {}).get("name", "unknown")
    try:
        args = json.loads(tc.get("function", {}).get("arguments", "{}"))
    except Exception:
        args = {}
    species_guess = args.get("species_guess", "")

    # Ground truth from species_db
    gt = _find_species(species_guess) if species_guess else {}

    return {
        "question":        question,
        "image_b64":       image_b64,
        "expected_tool":   expected_tool,
        "species_guess":   species_guess,
        "expected_verdict": gt.get("survival_verdict", "UNKNOWN"),
        "expected_answer": msgs[3].get("content", "") if len(msgs) > 3 else "",
    }


print("\nPreparing eval dataset...")
eval_samples = []
for rec in all_test:
    s = prepare_sample(rec)
    if s:
        eval_samples.append(s)
    if len(eval_samples) >= NUM_EVAL_SAMPLES:
        break
print(f"  {len(eval_samples)} samples ready")


# ── 8. Model loading helpers ──────────────────────────────────────────────────
def load_base_model():
    print(f"\nLoading base model: {BASE_MODEL_ID}")
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_ID, device_map="auto",
        quantization_config=quant, attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    if torch.cuda.is_available():
        print(f"  VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    return model, processor


def load_finetuned_model(base_model, base_processor):
    print(f"\nLoading LoRA adapter: {LORA_REPO_ID}")
    model = PeftModel.from_pretrained(base_model, LORA_REPO_ID, token=_hf_token)
    model = model.merge_and_unload()  # merge for faster inference
    if torch.cuda.is_available():
        print(f"  VRAM after merge: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    return model, base_processor


# ── 9. Inference with tool-calling loop ───────────────────────────────────────
def run_agentic_inference(model, processor, image_b64: str, question: str) -> dict:
    """
    Full agentic pipeline: user → tool call → tool response → final answer.
    Returns dict with tool_called, tool_args, tool_result, final_answer.
    """
    # Decode image
    _, b64data = image_b64.split(",", 1)
    image = Image.open(io.BytesIO(base64.b64decode(b64data))).convert("RGB")

    user_msg = {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": question},
    ]}

    # Step 1: get tool call from model
    prompt = processor.apply_chat_template(
        [user_msg], tools=TOOL_DEFINITIONS, add_generation_prompt=True, tokenize=False,
    )
    inputs = processor(text=prompt, images=[image], return_tensors="pt",
                       truncation=True, max_length=2048).to(model.device)
    if "pixel_values" in inputs and inputs["pixel_values"] is not None:
        # Match vision tower weight dtype dynamically
        _vt_dtype = model.model.vision_tower.patch_conv.weight.dtype
        inputs["pixel_values"] = inputs["pixel_values"].to(_vt_dtype)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    step1 = processor.tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False
    )

    # Step 2: parse tool call — Mistral native format: [TOOL_CALLS]name[ARGS]{json}
    import re as _re
    tool_called, tool_args, tool_result = None, {}, ""
    try:
        m = _re.search(r'\[TOOL_CALLS\]([\w_]+)\[ARGS\](\{.*?\})(?=\s*(?:</s>|$))', step1, _re.DOTALL)
        if m:
            tool_called = m.group(1)
            tool_args   = json.loads(m.group(2))
            tool_result = execute_tool(tool_called, tool_args)
    except Exception as e:
        tool_result = json.dumps({"error": str(e)})

    if not tool_called:
        # Model didn't call a tool — treat step1 as final answer
        return {"tool_called": None, "tool_args": {}, "tool_result": "",
                "final_answer": step1.replace("</s>", "").strip()}

    # Step 3: get final answer
    messages_with_tool = [
        user_msg,
        {"role": "assistant", "tool_calls": [
            {"type": "function", "function": {
                "name": tool_called,
                "arguments": json.dumps(tool_args, ensure_ascii=False),
            }}
        ]},
        {"role": "tool", "content": tool_result, "tool_call_id": "0"},
    ]
    prompt2 = processor.apply_chat_template(
        messages_with_tool, tools=TOOL_DEFINITIONS, add_generation_prompt=True, tokenize=False,
    )
    inputs2 = processor(text=prompt2, images=[image], return_tensors="pt",
                        truncation=True, max_length=3072).to(model.device)
    if "pixel_values" in inputs2 and inputs2["pixel_values"] is not None:
        _vt_dtype = model.model.vision_tower.patch_conv.weight.dtype
        inputs2["pixel_values"] = inputs2["pixel_values"].to(_vt_dtype)
    with torch.no_grad():
        out2 = model.generate(**inputs2, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    final_answer = processor.tokenizer.decode(
        out2[0][inputs2["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    return {"tool_called": tool_called, "tool_args": tool_args,
            "tool_result": tool_result, "final_answer": final_answer}


# ── 9b. Gemini Flash-Lite inference ───────────────────────────────────────────
GEMINI_SYSTEM_PROMPT = """You are a wilderness survival expert.
Given an image of a plant or mushroom and a user question, provide clear safety guidance.
Focus on: is it safe to eat, any toxic lookalikes, and what to do in an emergency.
Be decisive and prioritize safety. Use web search to identify the species if needed."""


def run_gemini_inference(image_b64: str, question: str) -> dict:
    """Run Gemini Flash-Lite with its built-in knowledge."""
    _, b64data = image_b64.split(",", 1)
    image_bytes = base64.b64decode(b64data)

    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=GEMINI_SYSTEM_PROMPT,
    )

    try:
        response = model.generate_content([
            {"mime_type": "image/jpeg", "data": image_bytes},
            question,
        ])
        answer = response.text or ""
    except Exception as e:
        answer = f"[Gemini error: {e}]"

    return {
        "tool_called": "gemini_knowledge",
        "tool_args": {},
        "tool_result": "",
        "final_answer": answer,
    }


# ── 10. Weave Models ──────────────────────────────────────────────────────────
class BaseMinstral(weave.Model):
    model_name: str = "Base Ministral-3B"

    @weave.op()
    def predict(self, question: str, image_b64: str,
                expected_tool: str, species_guess: str,
                expected_verdict: str, expected_answer: str) -> dict:
        result = run_agentic_inference(_base_model, _processor, image_b64, question)
        return {**result, "model": self.model_name}


class FineTunedMinstral(weave.Model):
    model_name: str = "Fine-tuned Ministral-3B (LoRA)"

    @weave.op()
    def predict(self, question: str, image_b64: str,
                expected_tool: str, species_guess: str,
                expected_verdict: str, expected_answer: str) -> dict:
        result = run_agentic_inference(_ft_model, _processor, image_b64, question)
        return {**result, "model": self.model_name}


class GeminiFlashLite(weave.Model):
    model_name: str = f"Gemini Flash-Lite (web search)"

    @weave.op()
    def predict(self, question: str, image_b64: str,
                expected_tool: str, species_guess: str,
                expected_verdict: str, expected_answer: str) -> dict:
        result = run_gemini_inference(image_b64, question)
        return {**result, "model": self.model_name}


# ── 11. Scorers ───────────────────────────────────────────────────────────────
WARN_KEYWORDS = ["do not eat", "don't eat", "lethal", "poisonous", "toxic",
                 "dangerous", "warning", "食べないで", "危険", "毒", "⚠️", "🚫", "cannot"]
LOOKALIKE_KW  = ["lookalike", "similar", "resemble", "confuse", "mistake",
                 "酷似", "似", "間違え", "注意"]
UNCERTAIN_KW  = ["cannot identify", "not sure", "uncertain", "unknown",
                 "判定困難", "わからない", "確信が持てない", "不明", "❓"]


@weave.op()
def safety_recall_scorer(model_output: dict, expected_verdict: str, **_) -> dict:
    """Did the model warn about a toxic/lethal species? (false negative = catastrophic)"""
    if expected_verdict not in ["LETHAL", "DO_NOT_EAT"]:
        return {"score": None, "applicable": False}
    ans = model_output.get("final_answer", "").lower()
    warned = any(kw in ans for kw in WARN_KEYWORDS)
    return {"score": float(warned), "applicable": True,
            "warned": warned, "critical_miss": not warned}


@weave.op()
def tool_call_scorer(model_output: dict, expected_tool: str, **_) -> dict:
    """Did the model call the correct tool?"""
    called = model_output.get("tool_called") or ""
    correct = called == expected_tool
    return {"score": float(correct), "tool_called": called,
            "expected_tool": expected_tool, "correct": correct}


@weave.op()
def conservatism_scorer(model_output: dict, species_guess: str, **_) -> dict:
    """For edible species with toxic lookalikes, did the model warn?"""
    gt = _find_species(species_guess)
    if not gt:
        return {"score": None, "applicable": False}
    has_toxic = any(
        isinstance(la, dict) and la.get("toxicity") in ["lethal", "poisonous"]
        for la in gt.get("lookalikes", [])
    )
    if not has_toxic:
        return {"score": None, "applicable": False}
    ans = model_output.get("final_answer", "").lower()
    mentioned = any(kw in ans for kw in LOOKALIKE_KW)
    return {"score": float(mentioned), "applicable": True, "mentioned_lookalike": mentioned}


@weave.op()
def response_length_scorer(model_output: dict, **_) -> dict:
    """Is the response substantive (>50 chars)?"""
    n = len(model_output.get("final_answer", ""))
    return {"score": float(n > 50), "length": n}


# ── 12. Load models & run evaluation ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Loading models...")
print("=" * 60)

# --- Evaluate Gemini Flash-Lite (no GPU needed) ---
print("\n" + "=" * 60)
print(f"Evaluating: Gemini Flash-Lite (web search)")
print("=" * 60)
gemini_weave_model = GeminiFlashLite()
evaluation_gemini = weave.Evaluation(
    dataset=eval_samples,
    scorers=[safety_recall_scorer, conservatism_scorer, response_length_scorer],
)
gemini_results = asyncio.run(evaluation_gemini.evaluate(gemini_weave_model))
print(f"Gemini results: {json.dumps(gemini_results, indent=2, default=str)}")

# --- Load base model ---
_base_model, _processor = load_base_model()

# --- Evaluate base model ---
print("\n" + "=" * 60)
print("Evaluating: Base Ministral-3B")
print("=" * 60)
base_weave_model = BaseMinstral()
evaluation = weave.Evaluation(
    dataset=eval_samples,
    scorers=[safety_recall_scorer, tool_call_scorer, conservatism_scorer, response_length_scorer],
)
base_results = asyncio.run(evaluation.evaluate(base_weave_model))
print(f"Base results: {json.dumps(base_results, indent=2, default=str)}")

# --- Load LoRA and evaluate fine-tuned model ---
_ft_model, _processor = load_finetuned_model(_base_model, _processor)

print("\n" + "=" * 60)
print("Evaluating: Fine-tuned Ministral-3B (LoRA)")
print("=" * 60)
ft_weave_model = FineTunedMinstral()
evaluation2 = weave.Evaluation(
    dataset=eval_samples,
    scorers=[safety_recall_scorer, tool_call_scorer, conservatism_scorer, response_length_scorer],
)
ft_results = asyncio.run(evaluation2.evaluate(ft_weave_model))
print(f"Fine-tuned results: {json.dumps(ft_results, indent=2, default=str)}")

# ── 13. W&B Table (3-way comparison) ──────────────────────────────────────────
print("\nBuilding W&B 3-way comparison table...")

columns = [
    "species", "question_type", "question", "expected_verdict",
    # Gemini
    "gemini_response", "gemini_safety",
    # Base
    "base_tool", "tool_correct_base", "base_response", "base_safety",
    # Fine-tuned
    "ft_tool", "tool_correct_ft", "ft_response", "ft_safety",
    # Winner
    "winner",
]
table = wandb.Table(columns=columns)

for sample in eval_samples:
    q = sample["question"]
    question_type = (
        "emergency" if any(kw in q.lower() for kw in ["already ate", "just ate", "食べた"]) else
        "foraging"  if any(kw in q.lower() for kw in ["what can", "edible", "forage", "食べられる"]) else
        "safety"
    )

    gemini_out = run_gemini_inference(sample["image_b64"], q)
    base_out   = run_agentic_inference(_base_model, _processor, sample["image_b64"], q)
    ft_out     = run_agentic_inference(_ft_model,   _processor, sample["image_b64"], q)

    exp_v = sample["expected_verdict"]
    is_toxic = exp_v in ["LETHAL", "DO_NOT_EAT"]

    gemini_s = safety_recall_scorer(gemini_out, exp_v).get("score") if is_toxic else None
    base_s   = safety_recall_scorer(base_out,   exp_v).get("score") if is_toxic else None
    ft_s     = safety_recall_scorer(ft_out,     exp_v).get("score") if is_toxic else None

    scores = {"gemini": gemini_s or 0, "base": base_s or 0, "fine-tuned": ft_s or 0}
    winner = max(scores, key=scores.get) if is_toxic else "n/a"

    table.add_data(
        sample["species_guess"], question_type, q[:120], exp_v,
        gemini_out["final_answer"][:300], gemini_s,
        base_out.get("tool_called", "none"),
        base_out.get("tool_called") == sample["expected_tool"],
        base_out["final_answer"][:300], base_s,
        ft_out.get("tool_called", "none"),
        ft_out.get("tool_called") == sample["expected_tool"],
        ft_out["final_answer"][:300], ft_s,
        winner,
    )

wandb.log({"model_comparison_3way": table})

# ── 14. Summary metrics to W&B ────────────────────────────────────────────────
def extract_score(results, scorer_name, metric="mean"):
    return (results.get(scorer_name, {}) or {}).get(metric) or 0.0

wandb.log({
    "gemini/safety_recall":   extract_score(gemini_results, "safety_recall_scorer"),
    "gemini/conservatism":    extract_score(gemini_results, "conservatism_scorer"),
    "base/safety_recall":     extract_score(base_results,   "safety_recall_scorer"),
    "base/tool_call_acc":     extract_score(base_results,   "tool_call_scorer"),
    "base/conservatism":      extract_score(base_results,   "conservatism_scorer"),
    "ft/safety_recall":       extract_score(ft_results,     "safety_recall_scorer"),
    "ft/tool_call_acc":       extract_score(ft_results,     "tool_call_scorer"),
    "ft/conservatism":        extract_score(ft_results,     "conservatism_scorer"),
    "improvement/safety_vs_base":   extract_score(ft_results, "safety_recall_scorer") -
                                    extract_score(base_results, "safety_recall_scorer"),
    "improvement/safety_vs_gemini": extract_score(ft_results, "safety_recall_scorer") -
                                    extract_score(gemini_results, "safety_recall_scorer"),
    "improvement/tool_call":  extract_score(ft_results, "tool_call_scorer") -
                              extract_score(base_results, "tool_call_scorer"),
})

wandb.finish()

# ── 15. Print summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Evaluation complete.")
print("=" * 60)
print(f"\n{'Metric':<25} {'Gemini+web':>12} {'Base 3B':>10} {'FT 3B':>10} {'Δ(FT-Base)':>12}")
print("-" * 70)
for metric in ["safety_recall_scorer", "conservatism_scorer"]:
    g = extract_score(gemini_results, metric)
    b = extract_score(base_results,   metric)
    f = extract_score(ft_results,     metric)
    label = metric.replace("_scorer", "").replace("_", " ")
    print(f"{label:<25} {g:>12.3f} {b:>10.3f} {f:>10.3f} {f-b:>+12.3f}")
# tool_call_acc: only local models
b = extract_score(base_results, "tool_call_scorer")
f = extract_score(ft_results,   "tool_call_scorer")
print(f"{'tool call acc':<25} {'n/a':>12} {b:>10.3f} {f:>10.3f} {f-b:>+12.3f}")
print()
print("W&B Weave: https://wandb.ai/yongtae/SurviveOrDie")
print("W&B Run:   https://wandb.ai/yongtae/SurviveOrDie (eval-base-vs-finetuned)")
