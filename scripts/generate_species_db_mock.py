"""
Generate mock species DB for testing (no LLM required).
This creates sample data to test the full pipeline.
"""
import json
from pathlib import Path

DATA_DIR = Path("data")

# Mock species database (sample data)
MOCK_SPECIES_DB = {
    "Aconitum columbianum": {
        "scientific_name": "Aconitum columbianum",
        "common_name_en": "Western Monkshood",
        "common_name_ja": "ウエスタンモンクフード",
        "category": "plant",
        "family": "Ranunculaceae",
        "edibility": "poisonous",
        "toxins": ["aconitine", "mesaconitine"],
        "toxicity_detail": "Contains deadly aconitine alkaloids. Even small amounts can be fatal within hours.",
        "symptoms": "Numbness, tingling, vomiting, diarrhea, irregular heartbeat, respiratory failure",
        "first_aid": "Seek immediate emergency care. Do NOT induce vomiting. Monitor breathing and heart rate.",
        "lookalikes": [
            {
                "species": "Aconitum lycoctonum",
                "common_name": "Wolf's Bane / レイジンソウ",
                "relation": "similar",
                "toxicity": "poisonous",
                "distinguishing": "A. columbianum has more densely flowered spikes and bluer flowers"
            }
        ],
        "habitat": ["meadows", "mountain_slopes", "stream_banks"],
        "substrate": "",
        "season_months": [6, 7, 8],
        "distribution_lat_range": [40, 55],
        "altitude_range_m": [1000, 3000],
        "key_features": "Blue to purple helmet-shaped flowers, deeply divided leaves",
        "size_cm": "30-90 cm tall",
        "survival_verdict": "DO_NOT_EAT",
        "survival_note": "All parts of Aconitum species are poisonous. Do not consume."
    },
    "Aconitum napellus": {
        "scientific_name": "Aconitum napellus",
        "common_name_en": "Monkshood / Wolfsbane",
        "common_name_ja": "トリカブト",
        "category": "plant",
        "family": "Ranunculaceae",
        "edibility": "poisonous",
        "toxins": ["aconitine", "hypaconitine"],
        "toxicity_detail": "One of the most toxic plants in Europe. Lethal dose is as low as 2-4 mg of aconitine.",
        "symptoms": "Burning sensation in mouth, vomiting, diarrhea, muscle weakness, cardiac arrest",
        "first_aid": "CALL EMERGENCY IMMEDIATELY. This is a life-threatening poisoning. Do not induce vomiting.",
        "lookalikes": [
            {
                "species": "Delphinium elatum",
                "common_name": "Larkspur / オオヒエンソウ",
                "relation": "similar",
                "toxicity": "poisonous",
                "distinguishing": "Delphinium has spurred flowers, Aconitum has helmet-shaped flowers"
            },
            {
                "species": "Consolida ajacis",
                "common_name": "Doubtful Knight's Spur",
                "relation": "similar",
                "toxicity": "poisonous",
                "distinguishing": "Different flower structure and growth habit"
            }
        ],
        "habitat": ["mountain_slopes", "woodland_edges", "meadows"],
        "substrate": "",
        "season_months": [6, 7, 8, 9],
        "distribution_lat_range": [45, 60],
        "altitude_range_m": [500, 2500],
        "key_features": "Dark blue to purple helmet-shaped flowers in dense spikes",
        "size_cm": "50-150 cm tall",
        "survival_verdict": "DO_NOT_EAT",
        "survival_note": "Lethal poison. Never consume any part of this plant."
    }
}

def main():
    """Generate mock species DB."""
    print("SurviveOrDie - Mock Species DB Generator (Testing)")
    print("=" * 60)

    # Save mock DB
    output_path = DATA_DIR / "species_db.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(MOCK_SPECIES_DB, f, ensure_ascii=False, indent=2)

    print(f"✓ Mock species DB saved to {output_path}")
    print(f"  Total species: {len(MOCK_SPECIES_DB)}")
    print("\nNext step: Run generate_training_data.py to create training samples")

if __name__ == "__main__":
    main()
