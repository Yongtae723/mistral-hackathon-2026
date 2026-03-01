"""
Generate detailed species metadata using Vertex AI Gemini API.
Flow: species_list → LLM (Vertex AI Gemini) → species_db.json
"""
import json
import os
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
import google.auth
import google.auth.transport.requests as tr_requests


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


BATCH_SIZE = 50  # concurrent Vertex AI calls


def get_vertex_ai_token() -> str:
    """Get authenticated Vertex AI access token."""
    credentials, _ = google.auth.default()
    auth_req = tr_requests.Request()
    credentials.refresh(auth_req)
    return credentials.token


async def call_vertex_ai_gemini(
    scientific_name: str,
    category: str,
    token: str,
    semaphore: asyncio.Semaphore,
) -> Optional[Dict]:
    """Call Vertex AI Gemini API to generate species metadata."""
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

    url = (
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}"
        f"/locations/{LOCATION}/publishers/google/models/{MODEL_ID}:generateContent"
    )

    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": f"{SYSTEM_PROMPT}\n\n{user_prompt}"}]
        }],
        "generationConfig": {
            "maxOutputTokens": 8192,
            "temperature": 0.2,
            "topP": 0.8,
        },
    }

    import httpx

    async with semaphore:
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                )
                response.raise_for_status()
                text = response.json()["candidates"][0]["content"]["parts"][0]["text"]

                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()

                return json.loads(text)

        except Exception as e:
            print(f"  ✗ Vertex AI error for {scientific_name}: {e}")
            return None


def _validate_metadata(metadata: Dict, sp: Dict) -> Dict:
    """Add required fields and apply safety validation."""
    scientific_name = sp["scientific_name"]
    metadata["scientific_name"] = scientific_name
    metadata["category"] = sp.get("category", "unknown")

    if not metadata.get("common_name_en"):
        metadata["common_name_en"] = sp.get("common_name_en", "")
    if not metadata.get("common_name_ja"):
        metadata["common_name_ja"] = sp.get("common_name_ja", "")

    if metadata.get("survival_verdict") == "SAFE":
        has_toxic_lookalike = any(
            la.get("toxicity") in ["lethal", "poisonous"]
            for la in metadata.get("lookalikes", [])
        )
        if has_toxic_lookalike:
            print(f"  ⚠️  Correcting SAFE→DO_NOT_EAT for {scientific_name} (has toxic lookalike)")
            metadata["survival_verdict"] = "DO_NOT_EAT"
            metadata["survival_note"] = metadata.get("survival_note", "") + " Has toxic lookalike(s)."

    return metadata


async def generate_metadata_for_species(species_list: List[Dict], start_index: int = 0) -> Dict:
    """
    Generate detailed metadata for all species using Vertex AI Gemini.
    Calls up to BATCH_SIZE species concurrently. Supports resumption.
    """
    print("\n" + "=" * 60)
    print(f"Generating species metadata using Vertex AI Gemini ({MODEL_ID})")
    print(f"Project: {PROJECT_ID}, Location: {LOCATION}")
    print(f"Concurrency: {BATCH_SIZE} parallel calls")
    print("=" * 60)

    species_db: Dict = {}
    failed: List[str] = []

    # Load existing progress
    existing_db_path = DATA_DIR / "species_db_partial.json"
    if existing_db_path.exists():
        with open(existing_db_path, "r", encoding="utf-8") as f:
            species_db = json.load(f)
        print(f"Loaded {len(species_db)} existing entries from partial DB")

    # Filter pending species
    pending = [
        sp for sp in species_list[start_index:]
        if sp["scientific_name"] not in species_db
    ]
    skipped = len(species_list[start_index:]) - len(pending)
    print(f"Pending: {len(pending)} | Skipped (already done): {skipped}")

    if not pending:
        print("Nothing to do.")
        save_final_db(species_db, failed)
        return {"species_db": species_db, "failed": failed}

    # Get token once (valid ~1 hour — plenty for 100 species at 50 concurrent)
    try:
        token = get_vertex_ai_token()
    except Exception as e:
        print(f"✗ Authentication error: {e}")
        print("  Please run: gcloud auth application-default login")
        return {"species_db": species_db, "failed": failed}

    semaphore = asyncio.Semaphore(BATCH_SIZE)

    # Process in batches of BATCH_SIZE, save after each batch
    total = len(pending)
    for batch_start in range(0, total, BATCH_SIZE):
        batch = pending[batch_start : batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"\n── Batch {batch_num}/{total_batches}: {len(batch)} species ──")
        for sp in batch:
            print(f"  → {sp['scientific_name']}")

        tasks = [
            call_vertex_ai_gemini(sp["scientific_name"], sp["category"], token, semaphore)
            for sp in batch
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for sp, result in zip(batch, results):
            name = sp["scientific_name"]
            if isinstance(result, Exception) or result is None:
                failed.append(name)
                print(f"  ✗ Failed: {name}")
            else:
                metadata = _validate_metadata(result, sp)
                species_db[name] = metadata
                print(f"  ✓ {name} — {metadata.get('survival_verdict')}")

        save_partial_db(species_db, failed)

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


async def main_async():
    """Main execution."""
    print("SurviveOrDie - Species Metadata Generator (Vertex AI Gemini)")
    print("=" * 60)

    # Load species list
    species_list_path = DATA_DIR / "species_list.json"
    if not species_list_path.exists():
        print(f"✗ {species_list_path} not found!")
        print("   Run 01_download_images.py first.")
        return

    with open(species_list_path, "r", encoding="utf-8") as f:
        species_list = json.load(f)

    print(f"Loaded {len(species_list)} species from list")

    # Check for resumption (find first species not yet in partial DB)
    partial_db_path = DATA_DIR / "species_db_partial.json"
    start_index = 0
    if partial_db_path.exists():
        with open(partial_db_path, "r", encoding="utf-8") as f:
            partial = json.load(f)
        for i, sp in enumerate(species_list):
            if sp["scientific_name"] not in partial:
                start_index = i
                break
        else:
            start_index = len(species_list)  # all done
        print(f"Resuming from index {start_index} ({len(partial)} already done)")
    else:
        print("Starting fresh")

    # Generate metadata (parallel)
    await generate_metadata_for_species(species_list, start_index)

    print("\nNext step: Run 03_generate_training_data_vertex.py to create training samples")


if __name__ == "__main__":
    asyncio.run(main_async())
