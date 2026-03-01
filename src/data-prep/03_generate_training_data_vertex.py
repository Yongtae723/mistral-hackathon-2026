"""
Generate training data using rule engine + Vertex AI Gemini.
Flow: species_db + images → rule-based verdict + LLM response → train.jsonl
"""
import json
import os
import base64
import random
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import httpx
from tqdm import tqdm


# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DATA_DIR = Path("data")
IMAGES_DIR = Path("images")
OUTPUT_DIR = Path("data")

# Context generation parameters
ENVIRONMENTS = [
    "broadleaf_forest", "conifer_forest", "mixed_forest",
    "grassland", "riverside", "mountain_trail", "meadow"
]

SUBSTRATES_MUSHROOM = ["on_ground", "on_wood", "on_tree", "in_grass", "on_leaves"]

# Land regions for realistic coordinate generation (all confirmed on land)
LAND_REGIONS = [
    {"name": "Japan (Honshu)",        "lat": [34, 41],  "lon": [132, 141], "emergency": "119", "lang": "ja"},
    {"name": "Japan (Hokkaido)",       "lat": [42, 45],  "lon": [141, 145], "emergency": "119", "lang": "ja"},
    {"name": "Japan (Kyushu/Shikoku)", "lat": [31, 34],  "lon": [130, 134], "emergency": "119", "lang": "ja"},
    {"name": "Japan (Okinawa)",        "lat": [24, 28],  "lon": [122, 132], "emergency": "119", "lang": "ja"},
    {"name": "Korea",                  "lat": [34, 38],  "lon": [126, 130], "emergency": "119", "lang": "ko"},
    {"name": "China (northeast)",      "lat": [40, 50],  "lon": [118, 130], "emergency": "120", "lang": "zh"},
    {"name": "China (central)",        "lat": [28, 40],  "lon": [108, 122], "emergency": "120", "lang": "zh"},
    {"name": "China (south)",          "lat": [22, 28],  "lon": [108, 120], "emergency": "120", "lang": "zh"},
    {"name": "Central Europe",         "lat": [46, 54],  "lon": [10, 25],   "emergency": "112", "lang": "en"},
    {"name": "Western Europe",         "lat": [43, 52],  "lon": [-5, 10],   "emergency": "112", "lang": "en"},
    {"name": "Scandinavia",            "lat": [56, 68],  "lon": [8, 28],    "emergency": "112", "lang": "en"},
    {"name": "Eastern Europe",         "lat": [48, 58],  "lon": [24, 40],   "emergency": "112", "lang": "en"},
    {"name": "Mediterranean",          "lat": [36, 46],  "lon": [5, 28],    "emergency": "112", "lang": "en"},
    {"name": "Russia (west Siberia)",  "lat": [52, 62],  "lon": [60, 82],   "emergency": "103", "lang": "en"},
    {"name": "North America (NE)",     "lat": [42, 50],  "lon": [-80, -65], "emergency": "911", "lang": "en"},
    {"name": "North America (NW)",     "lat": [45, 55],  "lon": [-125,-110],"emergency": "911", "lang": "en"},
    {"name": "North America (SE)",     "lat": [30, 42],  "lon": [-90, -74], "emergency": "911", "lang": "en"},
    {"name": "Pacific Northwest (US)", "lat": [44, 49],  "lon": [-124,-115],"emergency": "911", "lang": "en"},
    {"name": "Rocky Mountains",        "lat": [36, 48],  "lon": [-112,-103],"emergency": "911", "lang": "en"},
    {"name": "Himalayas (foothills)",  "lat": [26, 32],  "lon": [80, 96],   "emergency": "112", "lang": "en"},
    {"name": "Taiwan",                 "lat": [22, 25],  "lon": [120, 122], "emergency": "119", "lang": "zh"},
]

QUESTIONS = {
    "safety": [
        "Can I eat this?",
        "Is this safe to eat?",
        "これ食べられますか？",
        "Is this poisonous?",
        "Should I eat this?",
        "このきのこは毒がありますか？"
    ],
    "cooking": [
        "How should I cook this?",
        "How do I prepare this safely?",
        "調理法を教えてください",
        "What's the best way to cook this?",
        "Can I eat this raw?",
        "どうやって食べればいいですか？"
    ],
    "identification": [
        "What is this?",
        "Can you identify this mushroom?",
        "この植物は何ですか？",
        "What species is this?",
        "Is this edible or poisonous?",
        "名前は何ですか？"
    ],
    "situation": [
        "I'm lost in the mountains and hungry. Can I eat this?",
        "My child found this. Is it dangerous?",
        "I already ate some of this. What should I do?",
        "遭難して3日目です。これは食べても大丈夫？",
        "Emergency situation - no food for 2 days. Safe to eat?",
        "家族がこれを食べました。大丈夫ですか？"
    ],
    "foraging": [
        "What can I find to eat around here?",
        "What's edible in this environment?",
        "この辺で食べられるものは何がありますか？",
        "Any safe mushrooms or plants I can forage here?",
        "What edible species are in season at this location?",
        "近くに食べられるものはありますか？"
    ]
}

LLM_CONCURRENCY = 80   # concurrent Gemini calls (300 causes DNS/resource exhaustion)

# Selected species only (mushrooms, one per safety category)
SELECTED_SPECIES = [
    "Amanita phalloides",    # LETHAL
    "Amanita muscaria",      # DO_NOT_EAT (clearly toxic)
    "Flammulina velutipes",  # CONFUSING  (edible but lethal lookalike in wild)
    "Lentinula edodes",      # CONDITIONAL_SAFE
    "Hericium erinaceus",    # SAFE
]

# Pattern distribution weights
PATTERN_WEIGHTS = {
    "high_confidence": 50,
    "medium_confidence": 20,
    "low_confidence": 10,
    "emergency": 10,
    "foraging": 10
}
PATTERNS = list(PATTERN_WEIGHTS.keys())
PATTERN_WEIGHT_VALUES = [PATTERN_WEIGHTS[p] for p in PATTERNS]

# System prompt for LLM response generation
RESPONSE_SYSTEM_PROMPT = """You are a mountain survival AI assistant. Generate natural, helpful responses for users asking about mushrooms/plants.

LANGUAGE: Always respond in English, regardless of the language of the user's question.

IMPORTANT: You are generating the assistant's response AFTER a tool call has returned simplified data.
The tool return contains ONLY: species name, edibility, toxins list, and boolean context matches (season_match, altitude_match, habitat_match).
Your response must be based ONLY on these fields. Do NOT reference information not present in the tool return (e.g., specific lookalike species names, symptoms, first aid details, habitat lists).

TONE GUIDELINES:
- Emergency situations: Urgent, clear, prioritizing safety
- Casual questions: Friendly, informative

RESPONSE STRUCTURE (follow this format):
1. Clear verdict header (use emoji: 🚫 for lethal, ⚠️ for caution, ✅ for safe)
2. Species identification (scientific name)
3. Safety explanation based on edibility and toxins from tool return
4. Context match warnings (if any match booleans are false)
5. General safety advice

SAFETY RULES (NON-NEGOTIABLE):
- NEVER recommend eating anything uncertain
- If lethal: Maximum urgency
- If any context match is false: Warn about reduced confidence
- If already consumed: Emergency instructions

Return the response text directly (no JSON wrapper).

DIVERSITY RULE: Every response must use unique wording, varied sentence structure, and a different opening phrase. Never repeat the same response text for different observations."""


def get_gemini_api_key() -> str:
    """Get Gemini API key from env."""
    key = os.getenv("GEMINI_API_KEY", GEMINI_API_KEY)
    if not key:
        raise ValueError("GEMINI_API_KEY not set. Run: export GEMINI_API_KEY=your_key")
    return key


def safe_range(value, default: List) -> List:
    """Return value if it's a valid [min, max] list, otherwise return default."""
    if not isinstance(value, list) or len(value) < 2:
        return default
    if any(v is None for v in value[:2]):
        return default
    return value


def safe_list(value, default: List) -> List:
    """Return value if it's a non-empty list without None, otherwise return default."""
    if not isinstance(value, list) or len(value) == 0:
        return default
    cleaned = [v for v in value if v is not None]
    return cleaned if cleaned else default


def is_confusing_species(species_data: Dict) -> bool:
    """Detect edible species that have dangerous lookalikes → always advise DO NOT EAT wild."""
    edibility = species_data.get("edibility", "")
    verdict = species_data.get("survival_verdict", "")
    if edibility not in ["edible", "edible_cooked", "edible_processed", "conditional"]:
        return False
    if verdict != "DO_NOT_EAT":
        return False
    return any(
        isinstance(la, dict) and la.get("toxicity") in ["lethal", "poisonous"]
        for la in species_data.get("lookalikes", [])
    )


def is_emergency_ingestion(question: str) -> bool:
    """Detect emergency ingestion situations."""
    keywords = [
        "already ate", "just ate", "already eaten", "just eaten",
        "食べました", "食べてしまった", "食べてしまいました",
        "家族がこれを食べました", "i ate", "we ate", "ate some"
    ]
    q_lower = question.lower()
    return any(kw in q_lower for kw in keywords)


def compute_context_match(species_data: Dict, context: Dict) -> Dict:
    """Calculate match degree between context and species_db data."""
    season_months = safe_list(species_data.get("season_months"), list(range(1, 13)))
    lat_range = safe_range(species_data.get("distribution_lat_range"), [0, 90])
    alt_range = safe_range(species_data.get("altitude_range_m"), [0, 9000])
    habitat = safe_list(species_data.get("habitat"), ENVIRONMENTS)

    season_match = context["month"] in season_months
    latitude_match = lat_range[0] <= context["lat"] <= lat_range[1]
    altitude_match = alt_range[0] <= context["altitude"] <= alt_range[1]
    habitat_match = context["env"] in habitat if isinstance(habitat, list) else True

    mismatch_count = sum([
        not season_match, not latitude_match,
        not altitude_match, not habitat_match
    ])

    return {
        "season_match": season_match,
        "latitude_match": latitude_match,
        "altitude_match": altitude_match,
        "habitat_match": habitat_match,
        "mismatch_count": mismatch_count
    }


def decide_pattern(species_data: Dict, context: Dict) -> str:
    """Deterministically decide pattern based on context match."""
    if is_emergency_ingestion(context.get("question", "")):
        return "emergency"

    context_match = compute_context_match(species_data, context)
    mismatch_count = context_match["mismatch_count"]

    if mismatch_count >= 2:
        return "low_confidence"
    elif mismatch_count == 1:
        return "medium_confidence"
    else:
        return "high_confidence"


def find_nearby_species(context: Dict, species_db: Dict) -> Dict:
    """Filter species_db by context for foraging queries."""
    lat = context["lat"]
    month = context["month"]
    altitude = context["altitude"]
    env = context["env"]

    edible_species = []
    dangerous_nearby = []

    for sci_name, sd in species_db.items():
        lat_range = safe_range(sd.get("distribution_lat_range"), [0, 90])
        season_months = safe_list(sd.get("season_months"), list(range(1, 13)))
        alt_range = safe_range(sd.get("altitude_range_m"), [0, 9000])
        habitat = safe_list(sd.get("habitat"), [])

        if not (lat_range[0] <= lat <= lat_range[1]):
            continue
        if month not in season_months:
            continue
        if not (alt_range[0] <= altitude <= alt_range[1]):
            continue
        if habitat and env not in habitat:
            continue

        edibility = sd.get("edibility", "unknown")
        survival_verdict = sd.get("survival_verdict", "UNKNOWN_DO_NOT_EAT")

        if edibility in ["edible", "conditional"]:
            edible_species.append({
                "species": sci_name,
                "edibility": edibility,
            })
        elif edibility in ["lethal", "poisonous", "toxic"] or survival_verdict == "LETHAL":
            dangerous_nearby.append({
                "species": sci_name,
                "danger_level": survival_verdict,
            })

    return {
        "edible_count": len(edible_species),
        "dangerous_count": len(dangerous_nearby),
        "edible_species": edible_species,
        "dangerous_nearby": dangerous_nearby,
    }


class RuleEngine:
    """Rule-based verdict determination engine."""

    @staticmethod
    def determine_verdict(species_data: Dict, context: Dict) -> Dict:
        """Determine survival verdict using rule-based logic."""
        edibility = species_data.get("edibility", "unknown")
        toxicity_verdict = species_data.get("survival_verdict", "UNKNOWN_DO_NOT_EAT")
        lookalikes = species_data.get("lookalikes", [])

        # Rule 1: Lethal species → absolute refusal
        if toxicity_verdict == "LETHAL" or edibility == "lethal":
            return {
                "verdict": "LETHAL_DO_NOT_EAT",
                "confidence": "certain",
                "reasoning": f"Contains lethal toxins. {species_data.get('toxicity_detail', 'Immediate medical attention required.')}"
            }

        # Rule 2: Has toxic lookalike
        for la in lookalikes:
            if la.get("toxicity") in ["lethal", "poisonous"]:
                return {
                    "verdict": "DO_NOT_EAT",
                    "confidence": "high",
                    "reasoning": f"Resembles {la['species']} ({la.get('common_name', '')}) which is {la.get('toxicity')}. {la.get('distinguishing', 'Cannot distinguish from photo alone.')}"
                }

        # Rule 3: Poisonous species
        if edibility in ["poisonous", "toxic"]:
            return {
                "verdict": "DO_NOT_EAT",
                "confidence": "high",
                "reasoning": f"Poisonous. {species_data.get('toxicity_detail', 'Do not consume.')}"
            }

        # Rule 4: Edible species → safe
        if edibility == "edible":
            return {
                "verdict": "SAFE",
                "confidence": "high",
                "reasoning": f"Edible species. {species_data.get('survival_note', 'Safe to eat when correctly identified.')}"
            }

        # Rule 5: Conditional (requires preparation) → caution
        if edibility == "conditional":
            return {
                "verdict": "CONDITIONAL_SAFE",
                "confidence": "medium",
                "reasoning": f"Edible only with proper preparation. {species_data.get('preparation_required', 'Must be cooked thoroughly before eating.')}"
            }

        # Rule 6: Unknown or insufficient data → refuse
        return {
            "verdict": "UNKNOWN_DO_NOT_EAT",
            "confidence": "low",
            "reasoning": "Insufficient data or uncertain identification. When in doubt, do not eat."
        }


class ContextGenerator:
    """Generates realistic contexts for training samples."""

    @staticmethod
    def generate_context(species_data: Dict, target_pattern: str = None) -> Dict:
        """Generate a context, optionally targeting a specific pattern."""
        lat_range = safe_range(species_data.get("distribution_lat_range"), [30, 50])
        base_season = safe_list(species_data.get("season_months"), [4, 5, 6, 9, 10, 11])
        base_alt_range = safe_range(species_data.get("altitude_range_m"), [0, 2000])
        base_habitat = safe_list(species_data.get("habitat"), ENVIRONMENTS)

        # Pick a land region that overlaps with species' lat range
        matching_regions = [
            r for r in LAND_REGIONS
            if r["lat"][0] <= lat_range[1] and r["lat"][1] >= lat_range[0]
        ]
        region = random.choice(matching_regions if matching_regions else LAND_REGIONS)
        # Clamp lat to intersection of species range and region range
        lat_min = max(lat_range[0], region["lat"][0])
        lat_max = min(lat_range[1], region["lat"][1])
        if lat_min > lat_max:
            lat_min, lat_max = region["lat"]
        lat = round(random.uniform(lat_min, lat_max), 2)
        lon = round(random.uniform(*region["lon"]), 2)

        month = random.choice(base_season)
        altitude = random.randint(*base_alt_range)
        env = random.choice(base_habitat) if isinstance(base_habitat, list) else base_habitat

        category = species_data.get("category", "mushroom")
        substrate = ""
        if category == "mushroom":
            sub = species_data.get("substrate", "on_ground")
            substrate = sub if isinstance(sub, str) else random.choice(SUBSTRATES_MUSHROOM)

        if target_pattern == "emergency":
            # Matching context with emergency question
            emergency_questions = [q for q in QUESTIONS["situation"] if is_emergency_ingestion(q)]
            question = random.choice(emergency_questions) if emergency_questions else "I already ate some of this. What should I do?"
            question_type = "situation"

        elif target_pattern == "foraging":
            question = random.choice(QUESTIONS["foraging"])
            question_type = "foraging"

        elif target_pattern == "medium_confidence":
            # Mismatch exactly 1 parameter
            mismatch_type = random.choice(["season", "altitude", "habitat"])
            if mismatch_type == "season":
                off_months = [m for m in range(1, 13) if m not in base_season]
                if off_months:
                    month = random.choice(off_months)
            elif mismatch_type == "altitude":
                if base_alt_range[0] > 200:
                    altitude = random.randint(max(0, base_alt_range[0] - 800), base_alt_range[0] - 1)
                else:
                    altitude = random.randint(base_alt_range[1] + 100, base_alt_range[1] + 1000)
            elif mismatch_type == "habitat":
                if isinstance(base_habitat, list):
                    other_envs = [e for e in ENVIRONMENTS if e not in base_habitat]
                    if other_envs:
                        env = random.choice(other_envs)
            qt = random.choice(["safety", "cooking", "identification", "situation"])
            valid_qs = [q for q in QUESTIONS[qt] if not is_emergency_ingestion(q)]
            question = random.choice(valid_qs if valid_qs else QUESTIONS["safety"])
            question_type = qt

        elif target_pattern == "low_confidence":
            # Mismatch 2+ parameters
            off_months = [m for m in range(1, 13) if m not in base_season]
            if off_months:
                month = random.choice(off_months)
            altitude = base_alt_range[1] + random.randint(500, 1500)
            if isinstance(base_habitat, list):
                other_envs = [e for e in ENVIRONMENTS if e not in base_habitat]
                if other_envs:
                    env = random.choice(other_envs)
            qt = random.choice(["safety", "cooking", "identification", "situation"])
            valid_qs = [q for q in QUESTIONS[qt] if not is_emergency_ingestion(q)]
            question = random.choice(valid_qs if valid_qs else QUESTIONS["safety"])
            question_type = qt

        else:  # high_confidence or None
            qt = random.choice(["safety", "cooking", "identification", "situation"])
            valid_qs = [q for q in QUESTIONS[qt] if not is_emergency_ingestion(q)]
            question = random.choice(valid_qs if valid_qs else QUESTIONS["safety"])
            question_type = qt

        context_str = (
            f"lat: {lat}, lon: {lon}, region: {region['name']}, "
            f"month: {month}, alt: {altitude}m, env: {env}"
        )
        if substrate:
            context_str += f", growing: {substrate}"
        context_str += f"\n{question}"

        return {
            "context_string": context_str,
            "lat": lat,
            "lon": lon,
            "region": region["name"],
            "emergency_number": region["emergency"],
            "month": month,
            "altitude": altitude,
            "env": env,
            "substrate": substrate,
            "question": question,
            "question_type": question_type
        }


def image_to_data_uri(image_path: Path) -> str:
    """Convert image file to base64 data URI."""
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    ext = image_path.suffix[1:]
    return f"data:image/{ext};base64,{data}"


async def generate_response_text_vertex(
    species_data: Dict,
    verdict: Dict,
    context: Dict,
    pattern: str = "high_confidence",
    client: Optional[httpx.AsyncClient] = None,
) -> str:
    """Generate natural response text using Gemini."""

    question_type = context.get("question_type", "safety")

    context_match = compute_context_match(species_data, context)

    species_info = f"""TOOL RETURN (this is what the model will see from species_db_lookup):
{json.dumps({"species": species_data['scientific_name'], "edibility": species_data.get('edibility', 'unknown'), "toxins": species_data.get('toxins', []), "season_match": context_match['season_match'], "altitude_match": context_match['altitude_match'], "habitat_match": context_match['habitat_match']}, ensure_ascii=False)}

USER CONTEXT:
{context['context_string']}

REGION INFO:
- Region: {context.get('region', 'Unknown')}
- Local emergency number: {context.get('emergency_number', '112')}

QUESTION TYPE: {question_type}
CONFIDENCE PATTERN: {pattern}"""

    if is_confusing_species(species_data):
        instruction = f"""Generate a survival advice response. This species ({species_data.get('common_name_en', '')}) may be edible but has dangerous lookalikes.
Based on the tool return showing edibility and toxin info, your response MUST:
1. Warn that wild-picked specimens should NOT be eaten without expert verification
2. Note that dangerous lookalike species exist
3. Conclude with: DO NOT EAT unless 100% certain of identification
Base your response ONLY on the tool return fields shown above."""

    elif pattern == "medium_confidence":
        mismatches = []
        if not context_match["season_match"]:
            mismatches.append("season_match=false")
        if not context_match["altitude_match"]:
            mismatches.append("altitude_match=false")
        if not context_match["habitat_match"]:
            mismatches.append("habitat_match=false")
        mismatch_str = ", ".join(mismatches) if mismatches else "environmental conditions differ"
        instruction = f"""Generate a survival advice response. The tool returned these mismatches: {mismatch_str}.
Reflect this uncertainty in your response — mention that identification confidence is reduced due to these context mismatches.
Still provide safety information but express appropriate uncertainty. Do NOT recommend eating anything uncertain.
Base your response ONLY on the tool return fields shown above."""

    elif pattern == "low_confidence":
        instruction = """Generate a survival advice response. MULTIPLE context mismatches detected (multiple of season_match, altitude_match, habitat_match are false).
Your response MUST strongly warn that this identification cannot be confirmed.
Use clear language like "I cannot confirm this identification" and "This species is not expected in these conditions".
Err heavily on the side of caution. DO NOT suggest the species is safe.
Base your response ONLY on the tool return fields shown above."""

    else:
        instruction = f"""Generate a natural, helpful survival advice response with high confidence.
Follow the response structure in the system prompt. Adapt tone to question type: {question_type}.
Use a unique opening phrase and vary your sentence structure — do not repeat the same wording as other responses.
Base your response ONLY on the tool return fields shown above (species, edibility, toxins, match booleans)."""

    prompt = f"""{species_info}

INSTRUCTION:
{instruction}

Generate the response now:"""

    try:
        api_key = get_gemini_api_key()
    except ValueError as e:
        print(f"  ⚠️  {e}")
        return generate_template_response(species_data, verdict, context, pattern)

    url = GEMINI_BASE_URL.format(model=GEMINI_MODEL) + f"?key={api_key}"
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": f"{RESPONSE_SYSTEM_PROMPT}\n\n{prompt}"}]
        }],
        "generationConfig": {
            "maxOutputTokens": 1500,
            "temperature": 0.8,
            "topP": 0.95,
        },
    }

    # Use shared client if provided, otherwise create a one-off client
    async def _do_request(c: httpx.AsyncClient) -> str:
        for attempt in range(5):
            try:
                response = await c.post(url, json=payload,
                                        headers={"Content-Type": "application/json"})
                if response.status_code == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
                response.raise_for_status()
                return response.json()["candidates"][0]["content"]["parts"][0]["text"]
            except Exception as e:
                if attempt == 4:
                    print(f"  ⚠️  Gemini failed after {attempt+1} attempts, using template: {e}")
                    return generate_template_response(species_data, verdict, context, pattern)
                await asyncio.sleep(2 ** attempt)
        return generate_template_response(species_data, verdict, context, pattern)

    if client is not None:
        return await _do_request(client)
    else:
        async with httpx.AsyncClient(timeout=120.0) as c:
            return await _do_request(c)


def generate_template_response(
    species_data: Dict,
    verdict: Dict,
    context: Dict,
    pattern: str = "high_confidence"
) -> str:
    """Fallback template-based response generation (uses only simplified tool return info)."""

    v = verdict["verdict"]
    sci_name = species_data["scientific_name"]
    edibility = species_data.get("edibility", "unknown")
    toxins = species_data.get("toxins", [])

    header_map = {
        "LETHAL_DO_NOT_EAT": "🚫 LETHAL — DO NOT EAT",
        "DO_NOT_EAT": "⚠️ DO NOT EAT",
        "CONDITIONAL_SAFE": "⚠️ EDIBLE WITH CAUTION",
        "SAFE": "✅ SAFE TO EAT",
        "UNKNOWN_DO_NOT_EAT": "❓ UNIDENTIFIED — DO NOT EAT"
    }

    response = f"""{header_map.get(v, '❓ UNKNOWN')}

Species: {sci_name}
Edibility: {edibility}"""

    if toxins:
        response += f"\nToxins: {', '.join(toxins)}"

    context_match = compute_context_match(species_data, context)

    if pattern == "medium_confidence":
        mismatches = []
        if not context_match["season_match"]:
            mismatches.append("season")
        if not context_match["altitude_match"]:
            mismatches.append("altitude")
        if not context_match["habitat_match"]:
            mismatches.append("habitat")
        if mismatches:
            response += f"\n\n⚠️ Context mismatch ({', '.join(mismatches)}). Identification confidence is reduced."

    elif pattern == "low_confidence":
        response += "\n\n🚫 WARNING: Multiple context mismatches detected. Cannot confirm identification. DO NOT consume any unverified species."

    return response


def create_training_sample(
    species_data: Dict,
    image_path: Path,
    context: Dict,
    verdict: Dict,
    response_text: str,
    pattern: str = "high_confidence"
) -> Dict:
    """Create one training sample in Mistral SFT format."""
    image_uri = image_to_data_uri(image_path)

    confidence_map = {
        "high_confidence": "high",
        "medium_confidence": "medium",
        "low_confidence": "low",
        "emergency": "low"
    }
    confidence = confidence_map.get(pattern, "high")
    context_match = compute_context_match(species_data, context)

    # Step 1: User sends image + context
    user_message = {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_uri}},
            {"type": "text", "text": context["context_string"]}
        ]
    }

    # Step 2: Assistant makes function call
    assistant_tool_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"call_{species_data['scientific_name'].replace(' ', '_')}_{random.randint(1000, 9999)}",
                "type": "function",
                "function": {
                    "name": "species_db_lookup",
                    "arguments": json.dumps({
                        "species_guess": species_data["scientific_name"],
                        "category": species_data["category"],
                        "confidence": confidence
                    })
                }
            }
        ]
    }

    # Step 3: Tool returns simplified confidence criteria
    db_response_data = {
        "species": species_data["scientific_name"],
        "edibility": species_data.get("edibility"),
        "toxins": species_data.get("toxins", []),
        "season_match": context_match["season_match"],
        "altitude_match": context_match["altitude_match"],
        "habitat_match": context_match["habitat_match"],
    }

    tool_response = {
        "role": "tool",
        "name": "species_db_lookup",
        "content": json.dumps(db_response_data, ensure_ascii=False)
    }

    # Step 4: Assistant generates final report
    final_response = {
        "role": "assistant",
        "content": response_text
    }

    return {
        "messages": [user_message, assistant_tool_call, tool_response, final_response]
    }


def create_emergency_sample(
    species_data: Dict,
    image_path: Path,
    context: Dict,
    verdict: Dict
) -> Dict:
    """Create emergency pattern training sample."""
    image_uri = image_to_data_uri(image_path)

    # Step 1: User sends image + context (with "already ate" question)
    user_message = {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_uri}},
            {"type": "text", "text": context["context_string"]}
        ]
    }

    # Step 2: Assistant calls emergency_protocol
    assistant_tool_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"call_emergency_{species_data['scientific_name'].replace(' ', '_')}_{random.randint(1000, 9999)}",
                "type": "function",
                "function": {
                    "name": "emergency_protocol",
                    "arguments": json.dumps({
                        "species_guess": species_data["scientific_name"],
                        "category": species_data["category"],
                        "time_since_ingestion": "unknown"
                    })
                }
            }
        ]
    }

    # Step 3: Tool returns simplified emergency data
    toxins = species_data.get("toxins", [])
    severity = verdict.get("verdict", "LETHAL_DO_NOT_EAT")

    emergency_response_data = {
        "severity": severity,
        "species": species_data["scientific_name"],
        "toxins": toxins,
    }

    tool_response = {
        "role": "tool",
        "name": "emergency_protocol",
        "content": json.dumps(emergency_response_data, ensure_ascii=False)
    }

    # Step 4: Assistant outputs JSON emergency card
    final_response = {
        "role": "assistant",
        "content": json.dumps(emergency_response_data, ensure_ascii=False, indent=2)
    }

    return {
        "messages": [user_message, assistant_tool_call, tool_response, final_response]
    }


def create_foraging_sample(
    image_path: Path,
    context: Dict,
    nearby_species_result: Dict
) -> Dict:
    """Create foraging pattern training sample."""
    image_uri = image_to_data_uri(image_path)

    # Step 1: User sends env image + context with foraging question
    user_message = {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_uri}},
            {"type": "text", "text": context["context_string"]}
        ]
    }

    # Step 2: Assistant calls nearby_species_search
    assistant_tool_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"call_foraging_{random.randint(1000, 9999)}",
                "type": "function",
                "function": {
                    "name": "nearby_species_search",
                    "arguments": json.dumps({
                        "latitude": context["lat"],
                        "month": context["month"],
                        "altitude_m": context["altitude"],
                        "environment": context["env"]
                    })
                }
            }
        ]
    }

    # Step 3: Tool returns nearby species list
    tool_response = {
        "role": "tool",
        "name": "nearby_species_search",
        "content": json.dumps(nearby_species_result, ensure_ascii=False)
    }

    # Step 4: Assistant outputs JSON species card list
    final_response = {
        "role": "assistant",
        "content": json.dumps(nearby_species_result, ensure_ascii=False, indent=2)
    }

    return {
        "messages": [user_message, assistant_tool_call, tool_response, final_response]
    }


def create_unknown_sample(image_path: Path, context: Dict) -> Dict:
    """Create training sample for unknown species → cannot identify → DO NOT EAT."""
    image_uri = image_to_data_uri(image_path)

    user_message = {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": image_uri}},
            {"type": "text", "text": context["context_string"]}
        ]
    }

    assistant_tool_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": f"call_unknown_{random.randint(1000, 9999)}",
            "type": "function",
            "function": {
                "name": "species_db_lookup",
                "arguments": json.dumps({
                    "species_guess": "unknown mushroom",
                    "category": "mushroom",
                    "confidence": "low"
                })
            }
        }]
    }

    tool_response = {
        "role": "tool",
        "name": "species_db_lookup",
        "content": json.dumps({
            "found": False,
            "message": "Species not found in database. Cannot identify."
        })
    }

    final_response = {
        "role": "assistant",
        "content": (
            "❓ CANNOT IDENTIFY — DO NOT EAT\n\n"
            "This species is not in my database and cannot be identified from this image.\n\n"
            "DO NOT EAT any mushroom you cannot identify with 100% certainty. "
            "Many deadly mushrooms resemble edible ones. "
            "When in doubt, do not eat."
        )
    }

    return {"messages": [user_message, assistant_tool_call, tool_response, final_response]}


async def generate_training_data(
    species_db: Dict,
    images_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Dict[str, List[Dict]]:
    """Generate all training data samples with pattern distribution control."""

    print("\n" + "=" * 60)
    print("Generating training data samples...")
    print(f"LLM concurrency: {LLM_CONCURRENCY}")
    print("=" * 60)

    # Fix random seed so job list is identical across runs → cache hits work
    random.seed(42)

    rule_engine = RuleEngine()
    context_gen = ContextGenerator()

    # ── Phase 1: plan all jobs ────────────────────────────────
    # Each job: {split, species_data, img, context, pattern, verdict}
    all_jobs: List[Dict] = []

    print("\nPhase 1: Planning jobs...")
    print(f"Selected species: {SELECTED_SPECIES}")
    for sci_name, species_data in tqdm(species_db.items(), desc="Species", unit="sp"):
        if sci_name not in SELECTED_SPECIES:
            continue
        category = species_data.get("category", "unknown")
        species_dir = images_dir / category / sci_name.replace(" ", "_")

        if not species_dir.exists():
            continue

        images = sorted([
            f for f in species_dir.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])
        if len(images) < 2:
            continue

        n_train = max(1, int(len(images) * train_ratio))
        n_val = max(1, int(len(images) * val_ratio))
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        for img in train_imgs:
            for _ in range(3):
                pattern = random.choices(PATTERNS, weights=PATTERN_WEIGHT_VALUES, k=1)[0]
                ctx = context_gen.generate_context(species_data, target_pattern=pattern)
                verdict = rule_engine.determine_verdict(species_data, ctx) if pattern != "foraging" else {}
                all_jobs.append({"split": "train", "species_data": species_data,
                                  "img": img, "context": ctx, "pattern": pattern, "verdict": verdict})

        for img in val_imgs:
            ctx = context_gen.generate_context(species_data)
            verdict = rule_engine.determine_verdict(species_data, ctx)
            all_jobs.append({"split": "val", "species_data": species_data,
                              "img": img, "context": ctx, "pattern": "high_confidence", "verdict": verdict})

        for img in test_imgs:
            ctx = context_gen.generate_context(species_data)
            verdict = rule_engine.determine_verdict(species_data, ctx)
            all_jobs.append({"split": "test", "species_data": species_data,
                              "img": img, "context": ctx, "pattern": "high_confidence", "verdict": verdict})

    # ── Unknown pool ──────────────────────────────────────────
    unknown_dir = images_dir.parent / "images" / "unknown" / "mushroom"
    if not unknown_dir.exists():
        unknown_dir = Path("images") / "unknown" / "mushroom"

    if unknown_dir.exists():
        unknown_images = sorted([
            f for f in unknown_dir.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])
        unknown_images = unknown_images[:len(unknown_images) // 2]  # use half
        n_train = max(1, int(len(unknown_images) * train_ratio))
        n_val = max(1, int(len(unknown_images) * val_ratio))
        u_train = unknown_images[:n_train]
        u_val   = unknown_images[n_train:n_train + n_val]
        u_test  = unknown_images[n_train + n_val:]

        dummy_species = {"scientific_name": "unknown", "category": "mushroom"}
        for img in u_train:
            ctx = context_gen.generate_context(dummy_species)
            all_jobs.append({"split": "train", "pattern": "unknown", "img": img, "context": ctx,
                              "species_data": dummy_species, "verdict": {}})
        for img in u_val:
            ctx = context_gen.generate_context(dummy_species)
            all_jobs.append({"split": "val", "pattern": "unknown", "img": img, "context": ctx,
                              "species_data": dummy_species, "verdict": {}})
        for img in u_test:
            ctx = context_gen.generate_context(dummy_species)
            all_jobs.append({"split": "test", "pattern": "unknown", "img": img, "context": ctx,
                              "species_data": dummy_species, "verdict": {}})
        print(f"Unknown pool: {len(unknown_images)} images → train:{len(u_train)} val:{len(u_val)} test:{len(u_test)}")

    llm_indices = [i for i, j in enumerate(all_jobs) if j["pattern"] not in ("emergency", "foraging", "unknown")]
    print(f"Total jobs: {len(all_jobs)}  |  LLM calls needed: {len(llm_indices)}")

    # ── Phase 2: parallel Gemini calls (with disk cache) ─────
    CACHE_PATH = OUTPUT_DIR / "responses_cache.json"
    responses: Dict[int, str] = {}

    # Resume from cache if available
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            cached = json.load(f)
        responses = {int(k): v for k, v in cached.items()}
        remaining = [i for i in llm_indices if i not in responses]
        print(f"\nPhase 2: Resuming from cache ({len(responses)} done, {len(remaining)} remaining)...")
    else:
        remaining = llm_indices
        print(f"\nPhase 2: Generating Gemini responses ({len(remaining)} calls)...")

    semaphore = asyncio.Semaphore(LLM_CONCURRENCY)
    CACHE_SAVE_INTERVAL = 500  # save to disk every N completions
    save_counter = 0

    def _save_cache():
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in responses.items()}, f, ensure_ascii=False)

    # Shared httpx client with connection pool limits to avoid resource exhaustion
    limits = httpx.Limits(max_connections=LLM_CONCURRENCY + 20,
                          max_keepalive_connections=LLM_CONCURRENCY)

    async with httpx.AsyncClient(timeout=120.0, limits=limits) as shared_client:
        async def call_one(idx: int, job: Dict, bar: tqdm) -> Tuple[int, str]:
            nonlocal save_counter
            async with semaphore:
                r = await generate_response_text_vertex(
                    job["species_data"], job["verdict"], job["context"],
                    job["pattern"], client=shared_client
                )
                responses[idx] = r
                bar.update(1)
                save_counter += 1
                if save_counter % CACHE_SAVE_INTERVAL == 0:
                    _save_cache()
                return idx, r

        with tqdm(total=len(remaining), desc="Gemini", unit="call") as bar:
            tasks = [call_one(i, all_jobs[i], bar) for i in remaining]
            await asyncio.gather(*tasks, return_exceptions=True)

    # Final save
    _save_cache()
    print(f"  💾 Responses cached → {CACHE_PATH} ({len(responses)} total)")

    # ── Phase 3: build samples ────────────────────────────────
    print("\nPhase 3: Building samples...")
    train_data: List[Dict] = []
    val_data: List[Dict] = []
    test_data: List[Dict] = []
    pattern_counts = {p: 0 for p in PATTERNS + ["unknown"]}

    for i, job in enumerate(tqdm(all_jobs, desc="Samples", unit="sample")):
        pattern = job["pattern"]
        species_data = job["species_data"]
        img = job["img"]
        ctx = job["context"]
        verdict = job["verdict"]

        if pattern == "emergency":
            sample = create_emergency_sample(species_data, img, ctx, verdict)
        elif pattern == "foraging":
            nearby = find_nearby_species(ctx, species_db)
            sample = create_foraging_sample(img, ctx, nearby)
        elif pattern == "unknown":
            sample = create_unknown_sample(img, ctx)
        else:
            response_text = responses.get(
                i, generate_template_response(species_data, verdict, ctx, pattern)
            )
            sample = create_training_sample(species_data, img, ctx, verdict, response_text, pattern)

        if job["split"] == "train":
            train_data.append(sample)
            pattern_counts[pattern] += 1
        elif job["split"] == "val":
            val_data.append(sample)
        else:
            test_data.append(sample)

    # Save to JSONL files
    save_jsonl(train_data, OUTPUT_DIR / "train.jsonl")
    save_jsonl(val_data, OUTPUT_DIR / "val.jsonl")
    save_jsonl(test_data, OUTPUT_DIR / "test.jsonl")

    print(f"\n{'=' * 60}")
    print("Training data generation complete!")
    print(f"  Train samples : {len(train_data)}")
    print(f"  Val samples   : {len(val_data)}")
    print(f"  Test samples  : {len(test_data)}")
    print("\nPattern distribution (train):")
    for p, count in pattern_counts.items():
        pct = count / len(train_data) * 100 if train_data else 0
        print(f"  {p}: {count} ({pct:.1f}%)")
    print(f"{'=' * 60}")

    return {"train": train_data, "val": val_data, "test": test_data}


def save_jsonl(samples: List[Dict], filepath: Path):
    """Save samples to JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


async def main():
    """Main execution."""
    print("SurviveOrDie - Training Data Generator (Vertex AI Gemini)")
    print("=" * 60)

    # Load species database
    db_path = DATA_DIR / "species_db.json"
    if not db_path.exists():
        print(f"✗ {db_path} not found!")
        print("   Run generate_species_db_vertex.py first.")
        return

    with open(db_path, "r", encoding="utf-8") as f:
        species_db = json.load(f)

    print(f"Loaded {len(species_db)} species from database")

    if not IMAGES_DIR.exists():
        print(f"✗ {IMAGES_DIR} not found!")
        print("   Run download_from_inaturalist.py first.")
        return

    # Generate training data
    await generate_training_data(species_db, IMAGES_DIR)

    print("\nNext steps:")
    print("1. Review generated data (train.jsonl, val.jsonl, test.jsonl)")
    print("2. Upload to Google Colab for fine-tuning with Unsloth")


if __name__ == "__main__":
    asyncio.run(main())
