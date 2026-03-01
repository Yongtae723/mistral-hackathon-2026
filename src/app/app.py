# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "gradio>=5.0.0",
#   "torch>=2.1.0",
#   "transformers @ git+https://github.com/huggingface/transformers.git@bf3f0ae70d0e902efab4b8517fce88f6697636ce",
#   "peft>=0.12.0",
#   "pillow>=10.0.0",
#   "bitsandbytes>=0.43.0",
#   "accelerate>=0.34.0",
#   "sentencepiece",
#   "protobuf",
#   "google-generativeai>=0.8.0",
#   "huggingface_hub>=0.34.0",
# ]
# ///
"""
SurviveOrDie — Demo App
3-way comparison: Gemini Flash-Lite vs Base Ministral-3B vs Fine-tuned Ministral-3B
"""
import os, json, base64, io, time, re
import torch
import gradio as gr
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
import google.generativeai as genai
from mistralai import Mistral

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL_ID       = os.environ.get("MODEL_REPO_ID",   "mistralai/Ministral-3-3B-Instruct-2512-BF16")
LORA_REPO_ID        = os.environ.get("LORA_REPO_ID",    "yongtae-jp/AiOrDie-Ministral-3B-LoRA")
DATASET_REPO        = os.environ.get("DATASET_REPO",    "yongtae-jp/survive-or-die")
GEMINI_MODEL        = os.environ.get("GEMINI_MODEL",    "gemini-2.5-flash-lite")
HF_TOKEN            = os.environ.get("HF_TOKEN")
GEMINI_KEY          = os.environ.get("GEMINI_API_KEY")
MISTRAL_KEY         = os.environ.get("MISTRAL_API_KEY")
MISTRAL_BASE_MODEL  = os.environ.get("MISTRAL_BASE_MODEL",  "ministral-3b-latest")
MISTRAL_LARGE_MODEL = os.environ.get("MISTRAL_LARGE_MODEL", "mistral-large-latest")

# Auth
if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN, add_to_git_credential=False)
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
_mistral_client = Mistral(api_key=MISTRAL_KEY) if MISTRAL_KEY else None

# ── Species DB ────────────────────────────────────────────────────────────────
_SPECIES_DB: dict = {}
_IMG_ROOTS = ["images", "data/images", "../images"]

def _load_species_db():
    global _SPECIES_DB
    # Check local paths first (useful during local development)
    for p in ["species_db.json", "data/species_db.json", "../data/species_db.json"]:
        if os.path.exists(p):
            with open(p) as f:
                _SPECIES_DB = json.load(f)
            print(f"✓ species_db loaded ({len(_SPECIES_DB)} species) from {p}")
            return
    # Download from HF Hub dataset
    try:
        from huggingface_hub import hf_hub_download
        db_path = hf_hub_download(repo_id=DATASET_REPO, filename="species_db.json",
                                   repo_type="dataset", token=HF_TOKEN)
        with open(db_path) as f:
            _SPECIES_DB = json.load(f)
        print(f"✓ species_db loaded from HF Hub ({len(_SPECIES_DB)} species)")
    except Exception as e:
        print(f"⚠️ species_db not loaded: {e}")

_load_species_db()


def _find_species_image(scientific_name: str) -> str | None:
    """Return path to first available image for a species, or None."""
    folder_name = scientific_name.replace(" ", "_")
    for root in _IMG_ROOTS:
        for category in ["mushroom", "plant", ""]:
            base = os.path.join(root, category, folder_name) if category else os.path.join(root, folder_name)
            if os.path.isdir(base):
                imgs = sorted(f for f in os.listdir(base) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
                if imgs:
                    return os.path.join(base, imgs[0])
    return None


def _lookup_species(species_guess: str) -> dict | None:
    """Fuzzy match species_guess against DB. Returns species dict or None."""
    if not species_guess or not _SPECIES_DB:
        return None
    guess_l = species_guess.lower().strip()
    if species_guess in _SPECIES_DB:
        return _SPECIES_DB[species_guess]
    for key, data in _SPECIES_DB.items():
        if key.lower() == guess_l:
            return data
    for key, data in _SPECIES_DB.items():
        if guess_l in key.lower():
            return data
        cn_en = data.get("common_name_en", "").lower()
        cn_ja = data.get("common_name_ja", "").lower()
        if guess_l in cn_en or guess_l in cn_ja:
            return data
    genus = guess_l.split()[0] if " " in guess_l else guess_l
    for key, data in _SPECIES_DB.items():
        if key.lower().startswith(genus):
            return data
    return None


_EDIBILITY_BADGE = {
    "edible":       "🟢 **EDIBLE**",
    "conditional":  "🟡 **CONDITIONAL** *(requires preparation)*",
    "inedible":     "🟠 **INEDIBLE** *(not toxic, but don't eat)*",
    "poisonous":    "🔴 **POISONOUS**",
    "deadly":       "☠️ **DEADLY**",
    "unknown":      "⚪ **UNKNOWN**",
}

_MONTH_ABBR = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def _species_info_card(species_data: dict) -> str:
    """Render a rich markdown card for a species."""
    sci   = species_data.get("scientific_name", "Unknown")
    en    = species_data.get("common_name_en", "")
    ja    = species_data.get("common_name_ja", "")
    fam   = species_data.get("family", "")
    edib  = species_data.get("edibility", "unknown").lower()
    badge = _EDIBILITY_BADGE.get(edib, f"⚪ {edib.upper()}")

    months = species_data.get("season_months", [])
    season_str = " ".join(
        f"**{_MONTH_ABBR[m-1]}**" if m in months else _MONTH_ABBR[m-1]
        for m in range(1, 13)
    )

    habitats = species_data.get("habitat", [])
    habitat_str = ", ".join(habitats[:4]) if habitats else "—"

    alt = species_data.get("altitude_range_m", [])
    lat = species_data.get("distribution_lat_range", [])
    range_str = ""
    if alt and len(alt) == 2:
        range_str += f"Altitude: {alt[0]}–{alt[1]}m"
    if lat and len(lat) == 2:
        range_str += f"  ·  Latitude: {lat[0]}°–{lat[1]}°"

    features    = (species_data.get("key_features") or "")[:200]
    lookalikes  = species_data.get("lookalikes") or []
    toxins      = species_data.get("toxins") or []
    first_aid   = (species_data.get("first_aid") or "")[:200]

    lookalike_lines = []
    for lk in lookalikes[:3]:
        rel = lk.get("relationship", lk.get("relation", ""))
        danger = "⚠️ " if "dangerous" in rel.lower() or "confused" in rel.lower() else ""
        lookalike_lines.append(
            f"- {danger}**{lk.get('common_name', lk.get('species',''))}** "
            f"*({lk.get('species','')})*"
        )

    lines = [
        f"## 🍄 *{sci}*",
        f"**{en}**" + (f"  ·  {ja}" if ja else ""),
        f"*{fam}*" if fam else "",
        "",
        f"### Edibility  {badge}",
        "",
        f"| | |",
        f"|---|---|",
        f"| 🗓️ Season | {season_str} |",
        f"| 🌲 Habitat | {habitat_str} |",
        f"| 📍 Range | {range_str} |",
        "",
    ]
    if features:
        lines += [f"### Key Features", features, ""]
    if lookalike_lines:
        lines += [f"### ⚠️ Lookalikes", *lookalike_lines, ""]
    if toxins:
        lines += [f"### ☠️ Toxins", ", ".join(toxins), ""]
    if first_aid and edib in ("poisonous", "deadly"):
        lines += [f"### 🚨 First Aid", first_aid, ""]

    return "\n".join(lines)


# ── Tool definitions ───────────────────────────────────────────────────────────
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
        "description": "Search for edible and dangerous species near a given location.",
        "parameters": {"type": "object", "required": ["latitude", "month", "altitude_m", "environment"],
                       "properties": {
                           "latitude":    {"type": "number"},
                           "month":       {"type": "integer"},
                           "altitude_m":  {"type": "integer"},
                           "environment": {"type": "string"},
                       }},
    }},
]

# ── Model loading ──────────────────────────────────────────────────────────────
# Single PeftModel shared by base (adapter disabled) and fine-tuned (adapter enabled)
print("Loading models... (this takes ~1 minute)")
_processor  = None
_model      = None   # PeftModel: disable_adapter_layers() → base, enable → FT
_load_error = None

def _load_models():
    global _processor, _model, _load_error
    try:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,  # float16 for compatibility
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForImageTextToText.from_pretrained(
            BASE_MODEL_ID, device_map="auto",
            quantization_config=quant,
            attn_implementation="sdpa",
            torch_dtype=torch.float16,
        )
        _processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
        print("✓ Base model loaded")

        _model = PeftModel.from_pretrained(base, LORA_REPO_ID, token=HF_TOKEN)
        _model.eval()
        print(f"✓ LoRA adapter loaded: {LORA_REPO_ID}")
    except Exception as e:
        _load_error = str(e)
        print(f"✗ Model load error: {e}")

if torch.cuda.is_available():
    _load_models()
else:
    _load_error = "No GPU available. Local models disabled."
    print(f"⚠️  {_load_error}")


# ── Tool call parser ───────────────────────────────────────────────────────────
def _parse_tool_calls(text: str) -> list[dict]:
    """
    Parse Mistral's native tool call format:
      [TOOL_CALLS]func_name[ARGS]{"key": "val"}
    Returns list of tool_call dicts with id/type/function fields.
    """
    results = []
    for m in re.finditer(
        r'\[TOOL_CALLS\]([\w_]+)\[ARGS\](\{.*?\})(?=\s*(?:</s>|$|\[TOOL_CALLS\]))',
        text, re.DOTALL
    ):
        name = m.group(1)
        try:
            args = json.loads(m.group(2))
        except Exception:
            args = {}
        results.append({
            "id":       f"call_{name}_{len(results)}",
            "type":     "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
        })
    if results:
        return results

    # Fallback: JSON array format [{"name": ..., "arguments": ...}]
    m = re.search(r'\[TOOL_CALLS\]\s*(\[.*?\])', text, re.DOTALL)
    if m:
        try:
            raw = json.loads(m.group(1))
            return [
                {
                    "id":       f"call_{i}",
                    "type":     "function",
                    "function": {"name": x["name"], "arguments": json.dumps(x.get("arguments", {}), ensure_ascii=False)},
                }
                for i, x in enumerate(raw) if "name" in x
            ]
        except Exception:
            pass
    return []


# ── Inference helpers ──────────────────────────────────────────────────────────
_JSON_SCHEMA = """\
Respond ONLY with a JSON object in this exact format (no markdown, no extra text):
{
  "species": "<scientific name only, e.g. 'Amanita muscaria'. Use 'unknown' if unsure>",
  "danger_level": "<one of: LETHAL | POISONOUS | CONDITIONAL | SAFE | UNKNOWN>",
  "reply": "<your safety advice to the user in 2-3 sentences>"
}"""

GEMINI_SYSTEM = f"""You are a wilderness survival expert.
Given an image of a plant or mushroom and a user question, identify the species and assess safety.
{_JSON_SCHEMA}"""

_BASE_PROMPT_SUFFIX = f"\n\n{_JSON_SCHEMA}"

def _to_str(val) -> str:
    """Ensure val is a plain string."""
    if val is None:       return ""
    if isinstance(val, str): return val
    if isinstance(val, dict):
        for k in ("text", "content", "message", "advice", "reply", "warning"):
            if k in val and isinstance(val[k], str):
                return val[k]
        return json.dumps(val, ensure_ascii=False)
    return str(val)

def _extract_json_output(parsed: dict, raw: str) -> tuple[str | None, str, str]:
    """Extract (species, danger_level, reply) from structured JSON output."""
    reply = _to_str(
        parsed.get("reply") or parsed.get("warning") or parsed.get("message")
        or parsed.get("advice") or parsed.get("response") or parsed.get("answer") or raw
    )
    danger_level = _to_str(parsed.get("danger_level") or parsed.get("safety") or "UNKNOWN").upper()
    raw_sp = _to_str(parsed.get("species") or "")
    m = re.search(r'([A-Z][a-z]+(?:\s+[a-z]+)?)', raw_sp)
    species = m.group(1) if m else (raw_sp.strip() or None)
    if species and len(species) > 50:
        species = None
    return species, danger_level, reply

def _build_tool_result(tool_called: str, tool_args: dict, context: dict) -> str:
    """Build tool result matching training data format."""
    if tool_called == "species_db_lookup":
        species_data = _lookup_species(tool_args.get("species_guess", ""))
        if not species_data:
            return json.dumps({"error": "not_found", "queried": tool_args.get("species_guess", "")})
        season_months = species_data.get("season_months") or list(range(1, 13))
        alt_range     = species_data.get("altitude_range_m") or [0, 9000]
        habitat       = species_data.get("habitat") or []
        return json.dumps({
            "species":        species_data.get("scientific_name", tool_args.get("species_guess")),
            "edibility":      species_data.get("edibility"),
            "toxins":         species_data.get("toxins") or [],
            "season_match":   context.get("month", 6) in season_months,
            "altitude_match": alt_range[0] <= context.get("altitude", 0) <= alt_range[1],
            "habitat_match":  (context["env"] in habitat) if (habitat and context.get("env")) else True,
        })

    if tool_called == "emergency_protocol":
        species_data = _lookup_species(tool_args.get("species_guess", ""))
        if not species_data:
            return json.dumps({"severity": "UNKNOWN", "species": tool_args.get("species_guess", "unknown"), "toxins": []})
        return json.dumps({
            "severity": species_data.get("survival_verdict", "UNKNOWN"),
            "species":  species_data.get("scientific_name", tool_args.get("species_guess")),
            "toxins":   species_data.get("toxins", []),
        })

    if tool_called == "nearby_species_search":
        lat   = tool_args.get("latitude",    context.get("lat", 35.0))
        month = tool_args.get("month",       context.get("month", 6))
        alt   = tool_args.get("altitude_m",  context.get("altitude", 0))
        env   = tool_args.get("environment", context.get("env", ""))
        edible, dangerous = [], []
        for sci_name, sd in _SPECIES_DB.items():
            lat_r  = sd.get("distribution_lat_range") or [0, 90]
            months = sd.get("season_months") or list(range(1, 13))
            alt_r  = sd.get("altitude_range_m") or [0, 9000]
            hab    = sd.get("habitat") or []
            if not (lat_r[0] <= lat <= lat_r[1]):          continue
            if month not in months:                        continue
            if not (alt_r[0] <= alt <= alt_r[1]):          continue
            edibility = sd.get("edibility", "unknown")
            if edibility in ["edible", "conditional"]:
                edible.append({"species": sci_name, "edibility": edibility})
            elif edibility in ["lethal", "poisonous", "toxic"]:
                dangerous.append({"species": sci_name, "danger_level": sd.get("survival_verdict", "LETHAL")})
        return json.dumps({"edible_count": len(edible), "dangerous_count": len(dangerous),
                           "edible_species": edible, "dangerous_nearby": dangerous})

    return json.dumps({"error": f"Unknown tool: {tool_called}"})


def _inputs_to_device(inputs: dict, device) -> dict:
    """Move inputs to device, cast floating tensors to float16."""
    return {
        k: (v.to(device).to(torch.float16) if v.is_floating_point() else v.to(device))
        for k, v in inputs.items()
    }


@torch.no_grad()
def _run_ft(image: Image.Image, question: str, context: dict) -> tuple[str, dict, str]:
    """Fine-tuned model: 2-step agentic inference with tool calling."""
    if _model is None or _processor is None:
        return "—", {}, f"⚠️ {_load_error or 'Model not loaded'}"

    torch.cuda.empty_cache()
    _model.enable_adapter_layers()

    user_msg = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}

    # Step 1: generate tool call
    prompt1 = _processor.apply_chat_template(
        [user_msg], tools=TOOL_DEFINITIONS, add_generation_prompt=True, tokenize=False,
    )
    inputs1 = _processor(text=prompt1, images=[image], return_tensors="pt")
    inputs1 = _inputs_to_device(inputs1, _model.device)

    out1 = _model.generate(**inputs1, max_new_tokens=128, do_sample=False, use_cache=True)
    step1_text = _processor.tokenizer.decode(
        out1[0][inputs1["input_ids"].shape[1]:], skip_special_tokens=False
    )
    del inputs1, out1; torch.cuda.empty_cache()

    tool_calls = _parse_tool_calls(step1_text)
    if not tool_calls:
        return "none", {}, step1_text.replace("</s>", "").strip()

    tc = tool_calls[0]
    tool_name = tc["function"]["name"]
    tool_args = json.loads(tc["function"]["arguments"])
    tool_result = _build_tool_result(tool_name, tool_args, context)

    # Step 2: generate final answer
    messages2 = [
        user_msg,
        {"role": "assistant", "tool_calls": tool_calls},
        {"role": "tool", "content": tool_result},
    ]
    prompt2 = _processor.apply_chat_template(
        messages2, tools=TOOL_DEFINITIONS, add_generation_prompt=True, tokenize=False,
    )
    inputs2 = _processor(text=prompt2, images=[image], return_tensors="pt")
    inputs2 = _inputs_to_device(inputs2, _model.device)

    out2 = _model.generate(**inputs2, max_new_tokens=512, do_sample=False, use_cache=True)
    answer = _processor.tokenizer.decode(
        out2[0][inputs2["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    del inputs2, out2; torch.cuda.empty_cache()

    return tool_name, tool_args, answer


def _run_base(image: Image.Image, question: str) -> tuple[str, dict, str]:
    """Ministral-3B via API (text only)."""
    if not _mistral_client:
        return "—", {}, "⚠️ MISTRAL_API_KEY not set"
    try:
        resp = _mistral_client.chat.complete(
            model=MISTRAL_BASE_MODEL,
            messages=[{"role": "user", "content": question}],
        )
        return "none", {}, resp.choices[0].message.content.strip()
    except Exception as e:
        return "error", {}, f"⚠️ {e}"


def _run_mistral_large(image: Image.Image, question: str) -> tuple[str, dict, str]:
    """Mistral Large via API with vision."""
    if not _mistral_client:
        return "—", {}, "⚠️ MISTRAL_API_KEY not set"
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    try:
        resp = _mistral_client.chat.complete(
            model=MISTRAL_LARGE_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": question},
            ]}],
        )
        return "built-in knowledge", {}, resp.choices[0].message.content.strip()
    except Exception as e:
        return "error", {}, f"⚠️ {e}"


def _run_gemini(image: Image.Image, question: str) -> tuple[str, dict, str]:
    if not GEMINI_KEY:
        return "—", {}, "⚠️ GEMINI_API_KEY not set"
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    img_bytes = buf.getvalue()
    g = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=GEMINI_SYSTEM,
        generation_config=genai.types.GenerationConfig(response_mime_type="application/json"),
    )
    try:
        resp = g.generate_content([{"mime_type": "image/jpeg", "data": img_bytes}, question])
        parsed = json.loads(resp.text)
        species, danger_level, reply = _extract_json_output(parsed, resp.text)
        # Pack structured fields into args dict for downstream use
        return "built-in knowledge", {"danger_level": danger_level, "species": species}, reply
    except Exception as e:
        return "error", {}, f"⚠️ {e}"


def _fmt_tool(name: str, args: dict) -> str:
    icons = {
        "species_db_lookup":     "🔍",
        "emergency_protocol":    "🚨",
        "nearby_species_search": "🗺️",
        "built-in knowledge":    "🧠",
        "none":                  "⚠️",
        "error":                 "❌",
        "—":                     "—",
    }
    icon  = icons.get(name, "🔧")
    label = f"`{name}`" if name not in ("—", "none", "error", "built-in knowledge") else name
    if not args:
        return f"{icon} **{label}**"
    arg_lines = [f"  - `{k}`: **{v}**" for k, v in args.items()]
    return f"{icon} **{label}**\n" + "\n".join(arg_lines)


def _safety_badge(answer: str, danger_level: str | None = None) -> str:
    # Structured danger_level takes priority (avoids false positives from lookalike mentions)
    if danger_level and danger_level not in ("UNKNOWN", ""):
        dl = danger_level.upper()
        if dl in ("LETHAL", "POISONOUS"):   return "🔴 DANGER WARNING"
        if dl == "CONDITIONAL":             return "🟡 CONDITIONAL"
        if dl == "SAFE":                    return "🟢 SAFE"
    # Keyword fallback (for FT / base model free-text answers)
    a = answer.lower()
    if any(k in a for k in ["lethal", "do not eat", "don't eat", "☠️", "🚫", "119", "911",
                             "call emergency", "emergency services"]):
        return "🔴 DANGER WARNING"
    if any(k in a for k in ["safe to eat", "is edible", "can be eaten", "安全", "食べられる"]):
        return "🟢 SAFE"
    return "🟡 UNCERTAIN"


# ── Context builder ────────────────────────────────────────────────────────────
def _build_context_str(lat, lon, altitude, month, environment) -> str:
    if not any([lat, lon, altitude, month, environment]):
        return ""
    import datetime
    month_name = datetime.date(2000, int(month), 1).strftime("%B") if month else "unknown"
    parts = []
    if lat and lon: parts.append(f"lat {lat:.2f}°, lon {lon:.2f}°")
    elif lat:       parts.append(f"latitude {lat:.2f}°")
    if altitude:    parts.append(f"altitude {int(altitude)}m")
    if month:       parts.append(f"month: {month_name}")
    if environment: parts.append(f"environment: {environment}")
    return "\n\n📍 *Location context: " + ", ".join(parts) + "*"


# ── Gradio inference function ──────────────────────────────────────────────────
def compare(image, question, lat, lon, altitude, month):
    if not question.strip():
        question = "Is this safe to eat in a wilderness survival situation?"

    ctx_str  = _build_context_str(lat, lon, altitude, month, None)
    full_q   = question + ctx_str
    context  = {"lat": lat or 35.0, "altitude": altitude or 0, "month": int(month or 6), "env": ""}

    if image is None:
        pil_image = Image.new("RGB", (64, 64), (255, 255, 255))
    else:
        pil_image = Image.fromarray(image).convert("RGB") if not isinstance(image, Image.Image) else image

    t0 = time.time()
    base_tool, base_args, base_ans = _run_base(pil_image, full_q)
    t_base = time.time() - t0

    t1 = time.time()
    ft_tool, ft_args, ft_ans = _run_ft(pil_image, full_q, context)
    t_ft = time.time() - t1

    def _card(tool, args, ans, elapsed):
        badge = _safety_badge(ans, args.get("danger_level") if isinstance(args, dict) else None)
        lines = [badge, "", ans, ""]
        if tool not in ("none", "—", "error"):
            args_display = {k: v for k, v in (args or {}).items()
                            if k not in ("danger_level", "species")}
            lines += [
                "---",
                f"🔧 **Tool:** `{tool}`",
                f"```json\n{json.dumps(args_display, indent=2, ensure_ascii=False)}\n```",
            ]
        lines.append(f"*{elapsed:.1f}s*")
        return "\n".join(lines)

    species_card  = ""
    species_image = None
    if ft_tool == "species_db_lookup" and ft_args.get("species_guess"):
        data = _lookup_species(ft_args["species_guess"])
        if data:
            species_card  = _species_info_card(data)
            species_image = _find_species_image(data.get("scientific_name", ""))

    return (
        _card(ft_tool,   ft_args,   ft_ans,   t_ft),
        _card(base_tool, base_args, base_ans, t_base),
        species_image,
        species_card,
    )


# ── Gradio UI ──────────────────────────────────────────────────────────────────
EXAMPLE_QUESTIONS = [
    "Is this safe to eat? I'm lost in the mountains.",
    "I already ate this an hour ago. What should I do?",
    "What edible plants or mushrooms can I find around here?",
    "Is this mushroom safe? I'm very hungry and have no other food.",
]

with gr.Blocks(title="SurviveOrDie") as demo:
    gr.Markdown("# 🌿 SurviveOrDie — Wilderness Survival Assistant")

    with gr.Tabs():
        with gr.Tab("Analyze"):
            # ── Top: 2-column results ──────────────────────────────────────
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🌿 Fine-tuned Ministral-3B\n`Offline` · **our model**")
                    ft_out = gr.Markdown("")
                with gr.Column():
                    gr.Markdown("### 🤖 Base Ministral-3B\n`API` · no fine-tuning")
                    base_out = gr.Markdown("")

            with gr.Row():
                with gr.Column(scale=1):
                    species_image_out = gr.Image(label="Species Reference", height=200)
                with gr.Column(scale=2):
                    species_card_out = gr.Markdown("")

            gr.Markdown("---")

            # ── Bottom: input ──────────────────────────────────────────────
            with gr.Row():
                image_input = gr.Image(label="📷 Photo", type="numpy", height=260)
                with gr.Column():
                    question_input = gr.Textbox(
                        label="Question",
                        placeholder="Is this safe to eat?",
                        lines=3,
                    )
                    gr.Examples(
                        examples=EXAMPLE_QUESTIONS,
                        inputs=question_input,
                        label="Examples",
                    )
                    submit_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")

        with gr.Tab("📍 Location"):
            gr.Markdown("*In a real app these are auto-filled from GPS + barometer.*")
            with gr.Row():
                lat_input      = gr.Number(label="緯度 Lat",  value=35.0)
                lon_input      = gr.Number(label="経度 Lon",  value=139.69)
                altitude_input = gr.Number(label="標高 m",    value=300)
            with gr.Row():
                month_input = gr.Slider(label="Month", minimum=1, maximum=12, step=1, value=9)

    submit_btn.click(
        fn=compare,
        inputs=[image_input, question_input, lat_input, lon_input,
                altitude_input, month_input],
        outputs=[ft_out, base_out, species_image_out, species_card_out],
    )

if __name__ == "__main__":
    demo.launch(share=True)
