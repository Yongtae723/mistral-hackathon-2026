"""
Generate training data using rule engine + templates (no LLM required for testing).
Flow: species_db + images → rule-based verdict + template response → train.jsonl
"""
import json
import base64
import random
from pathlib import Path
from typing import Dict, List


# Configuration
DATA_DIR = Path("data")
IMAGES_DIR = Path("images")
OUTPUT_DIR = Path("data")

# Context generation parameters
ENVIRONMENTS = [
    "broadleaf_forest", "conifer_forest", "mixed_forest",
    "grassland", "riverside", "mountain_trail", "meadow"
]

SUBSTRATES_MUSHROOM = ["on_ground", "on_wood", "on_tree", "in_grass", "on_leaves"]

QUESTIONS = {
    "safety": [
        "Can I eat this?",
        "Is this safe to eat?",
        "Is this poisonous?",
    ],
    "cooking": [
        "How should I cook this?",
        "How do I prepare this safely?",
        "What's the best way to eat this?",
    ],
    "identification": [
        "What is this?",
        "Can you identify this plant?",
        "What species is this?",
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
    season_months = species_data.get("season_months", list(range(1, 13)))
    lat_range = species_data.get("distribution_lat_range", [0, 90])
    alt_range = species_data.get("altitude_range_m", [0, 9000])
    habitat = species_data.get("habitat", ENVIRONMENTS)

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
        lat_range = sd.get("distribution_lat_range", [0, 90])
        season_months = sd.get("season_months", list(range(1, 13)))
        alt_range = sd.get("altitude_range_m", [0, 9000])
        habitat = sd.get("habitat", [])

        if not (lat_range[0] <= lat <= lat_range[1]):
            continue
        if month not in season_months:
            continue
        if not (alt_range[0] <= altitude <= alt_range[1]):
            continue
        if isinstance(habitat, list) and env not in habitat:
            continue

        edibility = sd.get("edibility", "unknown")
        survival_verdict = sd.get("survival_verdict", "UNKNOWN_DO_NOT_EAT")

        if edibility in ["edible", "conditional"]:
            edible_species.append({
                "scientific_name": sci_name,
                "common_name_en": sd.get("common_name_en", ""),
                "common_name_ja": sd.get("common_name_ja", ""),
                "edibility": edibility,
                "where_to_look": f"in {env}",
                "key_features": sd.get("key_features", ""),
                "season_peak": True,
                "altitude_match": True
            })
        elif edibility in ["lethal", "poisonous", "toxic"] or survival_verdict == "LETHAL":
            dangerous_nearby.append({
                "scientific_name": sci_name,
                "common_name_en": sd.get("common_name_en", ""),
                "danger_level": survival_verdict,
                "active_this_season": True,
                "warning": f"Present in {env} this season. Do not consume."
            })

    return {
        "location": {
            "lat": lat,
            "month": month,
            "altitude_m": altitude,
            "env": env
        },
        "edible_species": edible_species,
        "dangerous_nearby": dangerous_nearby,
        "total_edible_found": len(edible_species)
    }


class RuleEngine:
    """Rule-based verdict determination engine."""

    @staticmethod
    def determine_verdict(species_data: Dict, context: Dict) -> Dict:
        """
        Determine survival verdict using rule-based logic.
        Returns verdict + reasoning.
        """
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
        lat_range = species_data.get("distribution_lat_range", [30, 50])
        base_season = species_data.get("season_months", [4, 5, 6, 9, 10, 11])
        base_alt_range = species_data.get("altitude_range_m", [0, 2000])
        base_habitat = species_data.get("habitat", ENVIRONMENTS)

        # Default: matching context
        lat = round(random.uniform(*lat_range), 2)
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

        context_str = f"lat: {lat}, month: {month}, alt: {altitude}m, env: {env}"
        if substrate:
            context_str += f", growing: {substrate}"
        context_str += f"\n{question}"

        return {
            "context_string": context_str,
            "lat": lat,
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
    ext = image_path.suffix[1:]  # Remove the dot
    return f"data:image/{ext};base64,{data}"


def generate_template_response(
    species_data: Dict,
    verdict: Dict,
    context: Dict,
    pattern: str = "high_confidence"
) -> str:
    """Template-based response generation with pattern-aware confidence."""

    v = verdict["verdict"]
    name_en = species_data.get("common_name_en", "Unknown")
    name_ja = species_data.get("common_name_ja", "")
    sci_name = species_data["scientific_name"]

    header_map = {
        "LETHAL_DO_NOT_EAT": "🚫 LETHAL — DO NOT EAT",
        "DO_NOT_EAT": "⚠️ DO NOT EAT",
        "CONDITIONAL_SAFE": "⚠️ EDIBLE WITH CAUTION",
        "SAFE": "✅ SAFE TO EAT",
        "UNKNOWN_DO_NOT_EAT": "❓ UNIDENTIFIED — DO NOT EAT"
    }

    response = f"""{header_map.get(v, '❓ UNKNOWN')}

Species: {name_en} ({name_ja}) — {sci_name}

{verdict['reasoning']}"""

    # Add lookalike warning
    lookalikes = species_data.get("lookalikes", [])
    if lookalikes:
        response += "\n\nSimilar species to watch for:"
        for la in lookalikes:
            response += f"\n- {la['species']} ({la.get('common_name', '')}): {la.get('distinguishing', '')}"

    # Add first aid if toxic
    if v in ["LETHAL_DO_NOT_EAT", "DO_NOT_EAT"]:
        first_aid = species_data.get("first_aid", "")
        if first_aid:
            response += f"\n\nIf ingested: {first_aid}"

    # Add preparation for conditional
    if v == "CONDITIONAL_SAFE":
        prep = species_data.get("preparation_required", "Cook thoroughly before eating.")
        response += f"\n\nPreparation: {prep}"

    # Add context-aware note
    question = context.get("question", "").lower()
    if context.get("question_type") == "situation":
        if "lost" in question or "遭難" in question:
            response += "\n\nSurvival tip: Prioritize finding water and shelter. Eating unidentified plants/mushrooms is more dangerous than going hungry for several days."
        if "child" in question or "子供" in question or "家族" in question:
            response += "\n\n⚠️ Children are more sensitive to toxins. If a child may have consumed this, seek immediate medical attention."

    # Pattern-specific confidence notes
    context_match = compute_context_match(species_data, context)
    if pattern == "medium_confidence":
        mismatches = []
        if not context_match["season_match"]:
            mismatches.append(f"month {context['month']} is outside typical season {species_data.get('season_months', [])}")
        if not context_match["altitude_match"]:
            mismatches.append(f"altitude {context['altitude']}m is outside expected range {species_data.get('altitude_range_m', [])}")
        if not context_match["habitat_match"]:
            mismatches.append(f"habitat '{context['env']}' doesn't match expected {species_data.get('habitat', [])}")
        if mismatches:
            response += f"\n\n⚠️ Context mismatch: {'; '.join(mismatches)}. This species is not typically found under these conditions — identification confidence is reduced."

    elif pattern == "low_confidence":
        response += "\n\n🚫 WARNING: Multiple context mismatches detected (season, altitude, and/or habitat). This species is highly unlikely at this location and time. Cannot confirm identification. DO NOT consume any unverified species."

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

    # Step 3: Tool returns DB data (with context_match)
    db_response_data = {
        "species": species_data["scientific_name"],
        "common_name_en": species_data.get("common_name_en"),
        "common_name_ja": species_data.get("common_name_ja"),
        "edibility": species_data.get("edibility"),
        "toxins": species_data.get("toxins", []),
        "toxicity_detail": species_data.get("toxicity_detail", ""),
        "symptoms": species_data.get("symptoms", ""),
        "lookalikes": species_data.get("lookalikes", []),
        "habitat": species_data.get("habitat", []),
        "season_months": species_data.get("season_months", []),
        "first_aid": species_data.get("first_aid", ""),
        "survival_verdict": species_data.get("survival_verdict", ""),
        "survival_note": species_data.get("survival_note", ""),
        "context_match": context_match
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

    # Step 3: Tool returns emergency protocol
    toxins = species_data.get("toxins", [])
    symptoms_raw = species_data.get("symptoms", "Symptoms may be delayed. Seek medical help immediately.")
    severity = verdict.get("verdict", "LETHAL_DO_NOT_EAT")

    emergency_response_data = {
        "severity": severity,
        "species": species_data["scientific_name"],
        "toxins": toxins,
        "actions": [
            {
                "step": 1,
                "action": "Call emergency services NOW",
                "numbers": {"JP": "119", "US": "911", "EU": "112"},
                "urgency": "IMMEDIATE"
            },
            {"step": 2, "action": "Do NOT induce vomiting", "urgency": "CRITICAL"},
            {"step": 3, "action": "Note exact time of ingestion", "urgency": "HIGH"},
            {"step": 4, "action": "Save a sample of what was eaten", "urgency": "HIGH"},
            {"step": 5, "action": "Go to nearest hospital immediately", "urgency": "IMMEDIATE"}
        ],
        "symptoms_onset": symptoms_raw,
        "critical_warning": "Feeling fine does NOT mean you are safe. Toxin symptoms can be significantly delayed.",
        "tell_doctor": f"Possible {', '.join(toxins) if toxins else 'unknown toxin'} ingestion"
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


def generate_training_data(
    species_db: Dict,
    images_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Dict[str, List[Dict]]:
    """
    Generate all training data samples with pattern distribution control.
    Returns train, val, test splits.
    """

    print("\n" + "=" * 60)
    print("Generating training data samples...")
    print("=" * 60)

    rule_engine = RuleEngine()
    context_gen = ContextGenerator()

    training_data = []
    val_data = []
    test_data = []
    pattern_counts = {p: 0 for p in PATTERNS}

    for sci_name, species_data in species_db.items():
        category = species_data.get("category", "unknown")
        species_dir = images_dir / category / sci_name.replace(" ", "_")

        if not species_dir.exists():
            print(f"  ⚠️  No images for {sci_name}")
            continue

        # Get image files
        images = sorted([
            f
            for f in species_dir.iterdir()
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ])

        if len(images) < 2:
            print(f"  ⚠️  Too few images for {sci_name}: {len(images)}")
            continue

        print(f"\nProcessing {sci_name}: {len(images)} images")

        # Split images
        n_train = max(1, int(len(images) * train_ratio))
        n_val = max(1, int(len(images) * val_ratio))

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        # Generate samples for train (2 variations per image)
        for img in train_imgs:
            for _ in range(2):
                pattern = random.choices(PATTERNS, weights=PATTERN_WEIGHT_VALUES, k=1)[0]
                ctx = context_gen.generate_context(species_data, target_pattern=pattern)

                if pattern == "emergency":
                    verdict = rule_engine.determine_verdict(species_data, ctx)
                    sample = create_emergency_sample(species_data, img, ctx, verdict)
                elif pattern == "foraging":
                    nearby = find_nearby_species(ctx, species_db)
                    sample = create_foraging_sample(img, ctx, nearby)
                else:
                    verdict = rule_engine.determine_verdict(species_data, ctx)
                    response_text = generate_template_response(
                        species_data, verdict, ctx, pattern
                    )
                    sample = create_training_sample(
                        species_data, img, ctx, verdict, response_text, pattern
                    )

                training_data.append(sample)
                pattern_counts[pattern] += 1

        # Generate samples for val (1 variation per image, default pattern)
        for img in val_imgs:
            ctx = context_gen.generate_context(species_data)
            verdict = rule_engine.determine_verdict(species_data, ctx)
            response_text = generate_template_response(species_data, verdict, ctx)
            sample = create_training_sample(
                species_data, img, ctx, verdict, response_text
            )
            val_data.append(sample)

        # Generate samples for test (1 variation per image, default pattern)
        for img in test_imgs:
            ctx = context_gen.generate_context(species_data)
            verdict = rule_engine.determine_verdict(species_data, ctx)
            response_text = generate_template_response(species_data, verdict, ctx)
            sample = create_training_sample(
                species_data, img, ctx, verdict, response_text
            )
            test_data.append(sample)

    # Save to JSONL files
    save_jsonl(training_data, OUTPUT_DIR / "train.jsonl")
    save_jsonl(val_data, OUTPUT_DIR / "val.jsonl")
    save_jsonl(test_data, OUTPUT_DIR / "test.jsonl")

    print(f"\n{'=' * 60}")
    print("Training data generation complete!")
    print(f"  Train samples: {len(training_data)}")
    print(f"  Val samples:   {len(val_data)}")
    print(f"  Test samples:  {len(test_data)}")
    print("\nPattern distribution (train):")
    for p, count in pattern_counts.items():
        pct = count / len(training_data) * 100 if training_data else 0
        print(f"  {p}: {count} ({pct:.1f}%)")
    print(f"{'=' * 60}")

    return {"train": training_data, "val": val_data, "test": test_data}


def save_jsonl(samples: List[Dict], filepath: Path):
    """Save samples to JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main():
    """Main execution."""
    print("SurviveOrDie - Training Data Generator (Mock/Template)")
    print("=" * 60)

    # Load species database
    db_path = DATA_DIR / "species_db.json"
    if not db_path.exists():
        print(f"✗ {db_path} not found!")
        print("   Run generate_species_db_mock.py first.")
        return

    with open(db_path, "r", encoding="utf-8") as f:
        species_db = json.load(f)

    print(f"Loaded {len(species_db)} species from database")

    # Check for images
    if not IMAGES_DIR.exists():
        print(f"✗ {IMAGES_DIR} not found!")
        print("   Run download_from_inaturalist.py first.")
        return

    # Generate training data
    generate_training_data(species_db, IMAGES_DIR)

    print("\n✓ All data generation steps complete!")
    print("\nGenerated files:")
    print("  - data/train.jsonl")
    print("  - data/val.jsonl")
    print("  - data/test.jsonl")


if __name__ == "__main__":
    main()
