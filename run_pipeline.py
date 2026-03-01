"""
SurviveOrDie Data Pipeline Runner
Execute all data generation steps in sequence.
"""
import asyncio
import sys
import os
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from download_from_inaturalist import main as download_main
from generate_species_db_mock import main as mock_db_main
from generate_training_data_mock import main as mock_training_main


async def run_test_pipeline():
    """Run the test pipeline (no LLM API required)."""
    print("=" * 70)
    print(" SurviveOrDie: TEST Pipeline (No LLM API required)")
    print("=" * 70)

    steps = [
        ("Generate mock species DB", mock_db_main),
        ("Generate training data (templates)", mock_training_main),
    ]

    for i, (step_name, step_func) in enumerate(steps, 1):
        print(f"\n{'=' * 70}")
        print(f" STEP {i}/{len(steps)}: {step_name}")
        print(f"{'=' * 70}\n")

        if asyncio.iscoroutinefunction(step_func):
            await step_func()
        else:
            step_func()

        print(f"\n✓ Step {i} complete!")

    print(f"\n{'=' * 70}")
    print(" ✓ TEST PIPELINE COMPLETE!")
    print(f"{'=' * 70}")
    print("\nGenerated files:")
    print("  - data/species_list.json")
    print("  - data/species_db.json")
    print("  - data/train.jsonl")
    print("  - data/val.jsonl")
    print("  - data/test.jsonl")


async def run_production_pipeline():
    """Run the production pipeline (LLM API required)."""
    print("=" * 70)
    print(" SurviveOrDie: PRODUCTION Pipeline (LLM API required)")
    print("=" * 70)

    # Check for API key
    if not os.getenv("LLM_API_KEY"):
        print("\n⚠️  LLM_API_KEY not set!")
        print("   Set it with: export LLM_API_KEY='your-api-key'")
        print("\n   Or run test pipeline: python run_pipeline.py --test")
        return

    steps = [
        ("Download images from iNaturalist", download_main),
        # Add other production steps as needed
        # ("Generate species DB (LLM)", db_main),
        # ("Generate training data (LLM)", training_main),
    ]

    for i, (step_name, step_func) in enumerate(steps, 1):
        print(f"\n{'=' * 70}")
        print(f" STEP {i}/{len(steps)}: {step_name}")
        print(f"{'=' * 70}\n")

        if asyncio.iscoroutinefunction(step_func):
            await step_func()
        else:
            step_func()

        print(f"\n✓ Step {i} complete!")


def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        asyncio.run(run_test_pipeline())
    else:
        asyncio.run(run_production_pipeline())


if __name__ == "__main__":
    main()
