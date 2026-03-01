"""
Generate detailed species metadata using Vertex AI Gemini API.
Flow: species_list → LLM (Vertex AI Gemini) → species_db.json
"""
import json
import os
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
import google.auth
import google.auth.transport.requests as tr_requests
import base64


# Configuration
PROJECT_ID = "beatrust-devs"
LOCATION = "us-central1"
MODEL_ID = "gemini-2.5-flash"
DATA_DIR = Path("data")
OUTPUT_DIR = Path("data")

# System prompt for LLM
SYSTEM_PROMPT = """You are a mycology and botany expert specializing in toxic species identification and survival advice.

Your task: Generate detailed metadata for the given species.

CRITICAL SAFETY RULES:
1. If a species is toxic, ALWAYS emphasize danger
2. If a species has toxic lookalikes, ALWAYS warn about them
3. NEVER mark a toxic species as "safe" to eat
4. Survival verdict must be conservative - "when in doubt, don't eat"

Output format: Return ONLY valid JSON with this structure:
{
  "scientific_name": "...",
  "common_name_en": "...",
  "common_name_ja": "...",
  "family": "...",
  "edibility": "lethal|poisonous|edible|edible_cooked|edible_processed|unknown",
  "toxins": ["..."],
  "toxicity_detail": "...",
  "symptoms": "...",
  "first_aid": "...",
  "lookalikes": [
    {
      "species": "...",
      "common_name": "...",
      "relation": "toxic_lookalike|edible_lookalike|similar",
      "toxicity": "lethal|poisonous|edible",
      "distinguishing": "..."
    }
  ],
  "habitat": ["..."],
  "substrate": "...",
  "season_months": [1,2,...],
  "distribution_lat_range": [min, max],
  "altitude_range_m": [min, max],
  "key_features": "...",
  "size_cm": "...",
  "survival_verdict": "LETHAL|DO_NOT_EAT|CONDITIONAL_SAFE|SAFE|UNKNOWN_DO_NOT_EAT",
  "survival_note": "...",
  "edible_parts": ["..."],
  "preparation_required": "...",
  "seasonal_edibility": "..."
}

VERDICT RULES:
- LETHAL: Species contains lethal toxins (amatoxins, orellanine, etc.)
- DO_NOT_EAT: Poisonous, OR edible but has lethal lookalike that can't be distinguished from photo
- CONDITIONAL_SAFE: Edible but requires cooking/processing
- SAFE: Edible with low misidentification risk AND no toxic lookalikes
- UNKNOWN_DO_NOT_EAT: Insufficient data - default to DO_NOT_EAT

Be specific and accurate. If you don't know a field, use null or empty array."""


def get_vertex_ai_client():
    """Get authenticated Vertex AI client."""
    credentials, project = google.auth.default()
    auth_req = tr_requests.Request()
    credentials.refresh(auth_req)
    return credentials.token


async def call_vertex_ai_gemini(scientific_name: str, category: str) -> Optional[Dict]:
    """
    Call Vertex AI Gemini API to generate species metadata.
    """
    user_prompt = f"""Generate detailed survival metadata for this species:

Species: {scientific_name}
Category: {category}

Include:
1. Toxicity information (toxins, symptoms, first aid)
2. Lookalike species (especially toxic ones)
3. Habitat and distribution
4. Survival verdict (follow the rules in system prompt)
5. For edible species: preparation requirements
6. For plants: edible parts

Return only valid JSON."""

    # Get access token
    try:
        token = get_vertex_ai_client()
    except Exception as e:
        print(f"  ✗ Authentication error: {e}")
        print("    Please run: gcloud auth application-default login")
        return None

    # Vertex AI API endpoint
    url = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}:generateContent"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": f"{SYSTEM_PROMPT}\n\n{user_prompt}"}]
        }],
        "generationConfig": {
            "maxOutputTokens": 2000,
            "temperature": 0.2,
            "topP": 0.8
        }
    }

    try:
        import httpx
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()

            # Extract text from Gemini response
            text = result["candidates"][0]["content"]["parts"][0]["text"]

            # Parse JSON (handle markdown code blocks)
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            return json.loads(text)

    except Exception as e:
        print(f"  ✗ Vertex AI error for {scientific_name}: {e}")
        return None


def generate_metadata_for_species(species_list: List[Dict], start_index: int = 0) -> Dict:
    """
    Generate detailed metadata for all species using Vertex AI Gemini.
    Supports resumption from last successful index.
    """
    print("\n" + "=" * 60)
    print(f"Generating species metadata using Vertex AI Gemini ({MODEL_ID})")
    print(f"Project: {PROJECT_ID}, Location: {LOCATION}")
    print("=" * 60)

    species_db = {}
    failed = []

    # Load existing progress if any
    existing_db_path = DATA_DIR / "species_db_partial.json"
    if existing_db_path.exists():
        with open(existing_db_path, "r", encoding="utf-8") as f:
            species_db = json.load(f)
        print(f"Loaded {len(species_db)} existing entries from partial DB")

    for i, sp in enumerate(species_list[start_index:], start_index):
        scientific_name = sp["scientific_name"]

        # Skip if already in DB
        if scientific_name in species_db:
            print(f"\n[{i+1}/{len(species_list)}] ✓ Skipping {scientific_name} (already in DB)")
            continue

        print(f"\n[{i+1}/{len(species_list)}] Processing {scientific_name}...")

        # Call Vertex AI
        metadata = asyncio.run(call_vertex_ai_gemini(scientific_name, sp["category"]))

        if metadata:
            # Ensure required fields are present
            metadata["scientific_name"] = scientific_name
            metadata["category"] = sp.get("category", "unknown")

            # Use common names from species_list if LLM didn't provide
            if not metadata.get("common_name_en"):
                metadata["common_name_en"] = sp.get("common_name_en", "")
            if not metadata.get("common_name_ja"):
                metadata["common_name_ja"] = sp.get("common_name_ja", "")

            # Safety validation
            if metadata.get("survival_verdict") == "SAFE":
                has_toxic_lookalike = any(
                    la.get("toxicity") in ["lethal", "poisonous"]
                    for la in metadata.get("lookalikes", [])
                )
                if has_toxic_lookalike:
                    print(f"  ⚠️  Correcting SAFE to DO_NOT_EAT (has toxic lookalike)")
                    metadata["survival_verdict"] = "DO_NOT_EAT"
                    metadata["survival_note"] = metadata.get("survival_note", "") + " Has toxic lookalike(s)."

            species_db[scientific_name] = metadata
            print(f"  ✓ Generated metadata for {scientific_name}")
            print(f"    Verdict: {metadata.get('survival_verdict')}")
        else:
            failed.append(scientific_name)
            print(f"  ✗ Failed to generate metadata for {scientific_name}")

        # Save progress every 3 species
        if (i + 1) % 3 == 0:
            save_partial_db(species_db, failed)

        # Rate limiting
        time.sleep(1)

    # Save final DB
    save_final_db(species_db, failed)

    return {"species_db": species_db, "failed": failed}


def save_partial_db(species_db: Dict, failed: List[str]):
    """Save partial progress."""
    partial_path = DATA_DIR / "species_db_partial.json"
    with open(partial_path, "w", encoding="utf-8") as f:
        json.dump(species_db, f, ensure_ascii=False, indent=2)

    failed_path = DATA_DIR / "species_db_failed.json"
    with open(failed_path, "w", encoding="utf-8") as f:
        json.dump(failed, f, ensure_ascii=False, indent=2)

    print(f"  💾 Saved partial DB ({len(species_db)} species)")


def save_final_db(species_db: Dict, failed: List[str]):
    """Save final species database."""
    final_path = DATA_DIR / "species_db.json"
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(species_db, f, ensure_ascii=False, indent=2)

    failed_path = DATA_DIR / "species_db_failed.json"
    with open(failed_path, "w", encoding="utf-8") as f:
        json.dump(failed, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print("✓ Species DB generation complete!")
    print(f"  Total species: {len(species_db)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Output: {final_path}")
    print(f"{'=' * 60}")

    if failed:
        print(f"\nFailed species:")
        for name in failed:
            print(f"  - {name}")


def main():
    """Main execution."""
    print("SurviveOrDie - Species Metadata Generator (Vertex AI Gemini)")
    print("=" * 60)

    # Load species list
    species_list_path = DATA_DIR / "species_list.json"
    if not species_list_path.exists():
        print(f"✗ {species_list_path} not found!")
        print("   Run download_from_inaturalist.py first.")
        return

    with open(species_list_path, "r", encoding="utf-8") as f:
        species_list = json.load(f)

    print(f"Loaded {len(species_list)} species from list")

    # Check for resumption
    partial_db = DATA_DIR / "species_db_partial.json"
    start_index = 0
    if partial_db.exists():
        with open(partial_db, "r", encoding="utf-8") as f:
            partial = json.load(f)
            for i, sp in enumerate(species_list):
                if sp["scientific_name"] not in partial:
                    start_index = i
                    break
            print(f"Resuming from index {start_index}")
    else:
        print("Starting fresh")

    # Generate metadata
    generate_metadata_for_species(species_list, start_index)

    print("\nNext step: Run generate_training_data_vertex.py to create training samples")


if __name__ == "__main__":
    main()
