"""
Evaluate Gemini response quality vs mock template.
Picks representative samples from train.jsonl, generates Gemini responses,
prints side-by-side comparison.
"""
import json
import asyncio
import random
from pathlib import Path
from typing import Dict, List, Optional
import httpx
import google.auth
import google.auth.transport.requests as tr_requests


# Configuration
PROJECT_ID = "beatrust-devs"
LOCATION = "us-central1"
MODEL_ID = "gemini-2.5-flash"
TRAIN_JSONL = Path("data/train.jsonl")
SAMPLES_PER_PATTERN = 1  # How many samples to test per pattern

RESPONSE_SYSTEM_PROMPT = """You are a mountain survival AI assistant. Generate natural, helpful responses for users asking about mushrooms/plants.

TONE GUIDELINES:
- Emergency situations: Urgent, clear, prioritizing safety
- Casual questions: Friendly, informative
- Japanese queries: Natural Japanese with appropriate politeness

RESPONSE STRUCTURE:
1. Clear verdict header (use emoji: 🚫 for lethal, ⚠️ for caution, ✅ for safe)
2. Species identification (scientific + common names)
3. Safety explanation
4. Lookalike warnings (if applicable)
5. First aid or preparation instructions (if applicable)
6. Context-aware advice

SAFETY RULES (NON-NEGOTIABLE):
- NEVER recommend eating anything uncertain
- If lethal: Maximum urgency, recommend immediate hospital
- If already consumed: Emergency first aid instructions

Return the response text directly (no JSON wrapper)."""


def get_vertex_ai_token():
    credentials, _ = google.auth.default()
    auth_req = tr_requests.Request()
    credentials.refresh(auth_req)
    return credentials.token


def classify_sample(sample: Dict) -> str:
    """Detect pattern type from sample."""
    for m in sample["messages"]:
        if "tool_calls" in m:
            for tc in m["tool_calls"]:
                name = tc["function"]["name"]
                if name == "emergency_protocol":
                    return "emergency"
                if name == "nearby_species_search":
                    return "foraging"
                if name == "species_db_lookup":
                    args = json.loads(tc["function"]["arguments"])
                    conf = args.get("confidence", "high")
                    return f"{conf}_confidence"
    return "unknown"


def extract_sample_info(sample: Dict) -> Dict:
    """Extract key fields from a training sample."""
    info = {"pattern": classify_sample(sample)}

    for m in sample["messages"]:
        if m["role"] == "user":
            for c in m.get("content", []):
                if c.get("type") == "text":
                    info["user_text"] = c["text"]

        if m["role"] == "tool":
            content = json.loads(m["content"])
            info["tool_name"] = m.get("name", "")
            info["tool_response"] = content

        if m["role"] == "assistant" and "tool_calls" not in m:
            info["mock_response"] = m.get("content", "")

    return info


def build_gemini_prompt(info: Dict) -> Optional[str]:
    """Build prompt for Gemini based on sample info."""
    tool_name = info.get("tool_name", "")
    tool_resp = info.get("tool_response", {})
    user_text = info.get("user_text", "")
    pattern = info.get("pattern", "")

    # emergency / foraging → JSON output, no LLM needed
    if tool_name in ["emergency_protocol", "nearby_species_search"]:
        return None

    # species_db_lookup → generate text response
    if tool_name == "species_db_lookup":
        if not tool_resp.get("species"):
            return None

        context_match = tool_resp.get("context_match", {})
        mismatch_count = context_match.get("mismatch_count", 0)

        if pattern == "low_confidence":
            confidence_instruction = """IMPORTANT: Multiple context mismatches detected.
Clearly state you cannot confirm the identification.
Use phrases like "I cannot confirm this" and "Do not consume".
Strong warning tone required."""
        elif pattern == "medium_confidence":
            mismatches = []
            if not context_match.get("season_match", True):
                mismatches.append("season mismatch")
            if not context_match.get("altitude_match", True):
                mismatches.append("altitude mismatch")
            if not context_match.get("habitat_match", True):
                mismatches.append("habitat mismatch")
            confidence_instruction = f"""NOTE: Context partially mismatches ({', '.join(mismatches)}).
Express reduced confidence in your identification.
Mention the environmental mismatch in your response."""
        else:
            confidence_instruction = "Standard high-confidence response."

        return f"""Generate a survival advice response based on this data:

SPECIES DB RESULT:
{json.dumps(tool_resp, ensure_ascii=False, indent=2)}

USER CONTEXT:
{user_text}

CONFIDENCE INSTRUCTION:
{confidence_instruction}

Generate the response:"""

    return None


async def call_gemini(prompt: str) -> str:
    """Call Vertex AI Gemini."""
    try:
        token = get_vertex_ai_token()
    except Exception as e:
        return f"[AUTH ERROR: {e}]"

    url = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}:generateContent"

    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": f"{RESPONSE_SYSTEM_PROMPT}\n\n{prompt}"}]
        }],
        "generationConfig": {
            "maxOutputTokens": 1500,
            "temperature": 0.3,
        }
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload, headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            })
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"[API ERROR: {e}]"


async def main():
    print("=" * 60)
    print("Gemini Response Quality Evaluation")
    print("=" * 60)

    # Load train.jsonl
    samples = []
    with open(TRAIN_JSONL) as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"Loaded {len(samples)} samples from train.jsonl")

    # Group by pattern
    by_pattern: Dict[str, List] = {}
    for s in samples:
        p = classify_sample(s)
        by_pattern.setdefault(p, []).append(s)

    print("\nPattern distribution:")
    for p, lst in sorted(by_pattern.items()):
        print(f"  {p}: {len(lst)}")

    # Pick representative samples
    targets = []
    for pattern in ["high_confidence", "medium_confidence", "low_confidence"]:
        pool = by_pattern.get(pattern, [])
        if pool:
            targets.extend(random.sample(pool, min(SAMPLES_PER_PATTERN, len(pool))))

    print(f"\nTesting {len(targets)} samples with Gemini...\n")

    results = []
    for i, sample in enumerate(targets, 1):
        info = extract_sample_info(sample)
        pattern = info.get("pattern", "?")
        species = info.get("tool_response", {}).get("species", "unknown")

        print(f"[{i}/{len(targets)}] {pattern} — {species}")

        prompt = build_gemini_prompt(info)
        if not prompt:
            print("  → Skipped (no LLM needed)")
            continue

        gemini_response = await call_gemini(prompt)
        results.append({
            "pattern": pattern,
            "species": species,
            "user_text": info.get("user_text", ""),
            "mock_response": info.get("mock_response", ""),
            "gemini_response": gemini_response,
        })
        print("  → Done")
        await asyncio.sleep(1.0)

    # Print comparison report
    print("\n" + "=" * 60)
    print("COMPARISON REPORT")
    print("=" * 60)

    for r in results:
        print(f"\n{'─' * 60}")
        print(f"Pattern : {r['pattern']}")
        print(f"Species : {r['species']}")
        print(f"Question: {r['user_text'].split(chr(10))[-1]}")
        print(f"\n--- MOCK (template) ---")
        print(r["mock_response"])
        print(f"\n--- GEMINI ---")
        print(r["gemini_response"])

    # Save to file
    output_path = Path("data/eval_gemini_comparison.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n\n✓ Saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
