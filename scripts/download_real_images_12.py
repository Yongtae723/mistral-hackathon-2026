"""
Download real images from iNaturalist for the 12 species in our mock dataset.
"""
import httpx
import json
from pathlib import Path
from typing import Dict, List, Optional

# 12 species for fine-tuning
SPECIES_TO_DOWNLOAD = [
    "Amanita phalloides",
    "Amanita virosa",
    "Galerina marginata",
    "Cantharellus cibarius",
    "Boletus edulis",
    "Grifola frondosa",
    "Aconitum napellus",
    "Cicuta virosa",
    "Veratrum album",
    "Oenanthe javanica",
    "Allium victorialis",
    "Artemisia indica"
]

BASE_URL = "https://api.inaturalist.org/v1"
IMAGES_DIR = Path("images")
MAX_IMAGES_PER_SPECIES = 10
RATE_LIMIT_DELAY = 0.5


async def get_taxon_id_by_name(scientific_name: str) -> Optional[int]:
    """Get iNaturalist taxon ID."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{BASE_URL}/taxa", params={
                "q": scientific_name,
                "rank": "species",
                "per_page": 5
            })
            results = resp.json().get("results", [])
            for r in results:
                if r.get("name", "").lower() == scientific_name.lower():
                    return r["id"]
            if results:
                return results[0]["id"]
            return None
    except Exception as e:
        print(f"  Error getting taxon ID: {e}")
        return None


async def download_species_images(scientific_name: str, category: str) -> Optional[Dict]:
    """Download research-grade images."""
    taxon_id = await get_taxon_id_by_name(scientific_name)

    if not taxon_id:
        print(f"  ⚠️  Taxon not found for {scientific_name}")
        return None

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{BASE_URL}/observations", params={
                "taxon_id": taxon_id,
                "quality_grade": "research",
                "photos": "true",
                "per_page": MAX_IMAGES_PER_SPECIES,
                "order": "votes",
                "order_by": "votes"
            })

        observations = resp.json().get("results", [])
        if not observations:
            print(f"  ⚠️  No observations for {scientific_name}")
            return None

        # Remove old mock images
        species_dir = IMAGES_DIR / category / scientific_name.replace(" ", "_")
        if species_dir.exists():
            import shutil
            shutil.rmtree(species_dir)
        species_dir.mkdir(parents=True, exist_ok=True)

        downloaded_count = 0
        unique_photos = set()

        async with httpx.AsyncClient(timeout=30.0) as client:
            for obs in observations:
                for photo in obs.get("photos", []):
                    if photo["id"] in unique_photos:
                        continue
                    unique_photos.add(photo["id"])

                    url = photo["url"].replace("square", "medium")
                    try:
                        img_resp = await client.get(url)
                        if img_resp.status_code == 200:
                            filepath = species_dir / f"{obs['id']}_{photo['id']}.jpg"
                            with open(filepath, "wb") as f:
                                f.write(img_resp.content)
                            downloaded_count += 1
                    except Exception as e:
                        print(f"    Error: {e}")

                await asyncio.sleep(RATE_LIMIT_DELAY)

        print(f"  ✓ Downloaded {downloaded_count} images for {scientific_name}")
        return {"scientific_name": scientific_name, "category": category, "count": downloaded_count}

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None


import asyncio

async def main():
    """Download real images for 12 species."""
    print("=" * 60)
    print("Downloading real images from iNaturalist for 12 species")
    print("=" * 60)

    # Category mapping
    fungi = {"Amanita phalloides", "Amanita virosa", "Galerina marginata",
             "Cantharellus cibarius", "Boletus edulis", "Grifola frondosa"}
    plants = {"Aconitum napellus", "Cicuta virosa", "Veratrum album",
              "Oenanthe javanica", "Allium victorialis", "Artemisia indica"}

    results = []

    for i, sci_name in enumerate(SPECIES_TO_DOWNLOAD, 1):
        print(f"\n[{i}/12] {sci_name}")

        category = "mushroom" if sci_name in fungi else "plant"
        result = await download_species_images(sci_name, category)

        if result:
            results.append(result)

    total = sum(r["count"] for r in results if r)
    print(f"\n{'=' * 60}")
    print(f"✓ Download complete! Total images: {total}")
    print(f"{'=' * 60}")
    print("\nNext: Run generate_training_data_mock.py")


if __name__ == "__main__":
    asyncio.run(main())
