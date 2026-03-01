"""
Download mushroom & plant images from iNaturalist.
- 50 known mushroom species → images/mushroom/{species_name}/
- 50 known plant species   → images/plant/{species_name}/
- unknown mushroom pool    → images/unknown/mushroom/
- unknown plant pool       → images/unknown/plant/
"""
import asyncio
import json
import httpx
from pathlib import Path
from typing import Dict, List, Optional
from tqdm.asyncio import tqdm


BASE_URL = "https://api.inaturalist.org/v1"
IMAGES_DIR = Path("images")
MAX_IMAGES_PER_SPECIES = 30
MAX_IMAGES_UNKNOWN_PER_SPECIES = 20
RATE_LIMIT_DELAY = 0.5
SPECIES_CONCURRENCY = 10  # concurrent species downloads

# ---------------------------------------------------------------
# 50 known mushroom species
# ---------------------------------------------------------------
KNOWN_MUSHROOMS: List[Dict] = [
    # --- Lethal (10) ---
    {"name": "Amanita phalloides",          "edibility": "lethal"},   # タマゴテングタケ
    {"name": "Amanita virosa",              "edibility": "lethal"},   # シロタマゴテングタケ
    {"name": "Galerina marginata",          "edibility": "lethal"},   # 猛毒カキノタケ
    {"name": "Amanita subjunquillea",       "edibility": "lethal"},   # ヒメシロテングタケ
    {"name": "Lepiota brunneoincarnata",    "edibility": "lethal"},   # ニセクリイロカラカサタケ
    {"name": "Cortinarius rubellus",        "edibility": "lethal"},   # 致死ドククモタケ
    {"name": "Russula subnigricans",        "edibility": "lethal"},   # クロハツモドキ
    {"name": "Tricholoma equestre",         "edibility": "lethal"},   # キシメジ
    {"name": "Omphalotus japonicus",        "edibility": "lethal"},   # ツキヨタケ
    {"name": "Gyromitra esculenta",         "edibility": "lethal"},   # シャグマアミガサタケ

    # --- Poisonous / do_not_eat (10) ---
    {"name": "Amanita muscaria",            "edibility": "poisonous"}, # ベニテングタケ
    {"name": "Hypholoma fasciculare",       "edibility": "poisonous"}, # ニガクリタケ
    {"name": "Entoloma sinuatum",           "edibility": "poisonous"}, # オオクサウラベニタケ
    {"name": "Chlorophyllum molybdites",    "edibility": "poisonous"}, # ドクカラカサタケ
    {"name": "Paxillus involutus",          "edibility": "poisonous"}, # マキバタケ
    {"name": "Boletus satanas",             "edibility": "poisonous"}, # サタンスボレタス
    {"name": "Tricholoma pardinum",         "edibility": "poisonous"}, # トラハツ
    {"name": "Scleroderma citrinum",        "edibility": "poisonous"}, # キツネノチャブクロ
    {"name": "Inocybe rimosa",              "edibility": "poisonous"}, # ワライタケ類
    {"name": "Omphalotus illudens",         "edibility": "poisonous"}, # ジャックオーランタン

    # --- Edible (30) ---
    {"name": "Cantharellus cibarius",       "edibility": "edible"},   # アンズタケ
    {"name": "Boletus edulis",              "edibility": "edible"},   # ヤマドリタケ
    {"name": "Grifola frondosa",            "edibility": "edible"},   # マイタケ
    {"name": "Tricholoma matsutake",        "edibility": "edible"},   # マツタケ
    {"name": "Lentinula edodes",            "edibility": "edible"},   # シイタケ
    {"name": "Flammulina velutipes",        "edibility": "edible"},   # エノキタケ
    {"name": "Pleurotus ostreatus",         "edibility": "edible"},   # ヒラタケ
    {"name": "Hericium erinaceus",          "edibility": "edible"},   # ヤマブシタケ
    {"name": "Sparassis crispa",            "edibility": "edible"},   # ハナビラタケ
    {"name": "Morchella esculenta",         "edibility": "edible"},   # アミガサタケ
    {"name": "Craterellus cornucopioides",  "edibility": "edible"},   # クロラッパタケ
    {"name": "Laetiporus sulphureus",       "edibility": "edible"},   # タコウキン
    {"name": "Armillaria mellea",           "edibility": "edible"},   # ナラタケ
    {"name": "Lycoperdon perlatum",         "edibility": "edible"},   # ホコリタケ
    {"name": "Macrolepiota procera",        "edibility": "edible"},   # オオカラカサタケ
    {"name": "Agaricus campestris",         "edibility": "edible"},   # ハラタケ
    {"name": "Suillus luteus",              "edibility": "edible"},   # ヌメリイグチ
    {"name": "Lactarius deliciosus",        "edibility": "edible"},   # アカハツ
    {"name": "Auricularia auricula-judae",  "edibility": "edible"},   # キクラゲ
    {"name": "Tremella fuciformis",         "edibility": "edible"},   # シロキクラゲ
    {"name": "Pleurotus eryngii",           "edibility": "edible"},   # エリンギ
    {"name": "Pholiota nameko",             "edibility": "edible"},   # ナメコ
    {"name": "Russula virescens",           "edibility": "edible"},   # アオハツ
    {"name": "Calvatia gigantea",           "edibility": "edible"},   # オオホコリタケ
    {"name": "Fistulina hepatica",          "edibility": "edible"},   # ビーフステーキタケ
    {"name": "Ramaria flava",               "edibility": "edible"},   # キホウキタケ
    {"name": "Kuehneromyces mutabilis",     "edibility": "edible"},   # チャナメツムタケ
    {"name": "Agaricus bisporus",           "edibility": "edible"},   # マッシュルーム
    {"name": "Tricholoma terreum",          "edibility": "edible"},   # ハイイロシメジ
    {"name": "Clavariadelphus truncatus",   "edibility": "edible"},   # ヤマホウキタケ
]

# ---------------------------------------------------------------
# 50 known plant species
# ---------------------------------------------------------------
KNOWN_PLANTS: List[Dict] = [
    # --- Lethal (10) ---
    {"name": "Aconitum napellus",       "edibility": "lethal"},   # トリカブト
    {"name": "Cicuta virosa",           "edibility": "lethal"},   # ドクゼリ
    {"name": "Veratrum album",          "edibility": "lethal"},   # バイケイソウ
    {"name": "Conium maculatum",        "edibility": "lethal"},   # ドクニンジン
    {"name": "Datura stramonium",       "edibility": "lethal"},   # チョウセンアサガオ
    {"name": "Taxus cuspidata",         "edibility": "lethal"},   # イチイ
    {"name": "Digitalis purpurea",      "edibility": "lethal"},   # ジギタリス
    {"name": "Colchicum autumnale",     "edibility": "lethal"},   # イヌサフラン
    {"name": "Atropa belladonna",       "edibility": "lethal"},   # ベラドンナ
    {"name": "Nerium oleander",         "edibility": "lethal"},   # キョウチクトウ

    # --- Poisonous / do_not_eat (10) ---
    {"name": "Phytolacca americana",    "edibility": "poisonous"}, # ヨウシュヤマゴボウ
    {"name": "Convallaria majalis",     "edibility": "poisonous"}, # スズラン
    {"name": "Solanum nigrum",          "edibility": "poisonous"}, # イヌホオズキ
    {"name": "Solanum dulcamara",       "edibility": "poisonous"}, # ヒヨドリジョウゴ
    {"name": "Hyoscyamus niger",        "edibility": "poisonous"}, # ヒヨス
    {"name": "Ranunculus sceleratus",   "edibility": "poisonous"}, # タガラシ
    {"name": "Euphorbia helioscopia",   "edibility": "poisonous"}, # トウダイグサ
    {"name": "Sambucus nigra",          "edibility": "poisonous"}, # ニワトコ（生）
    {"name": "Arisaema serratum",       "edibility": "poisonous"}, # マムシグサ
    {"name": "Brugmansia suaveolens",   "edibility": "poisonous"}, # エンジェルトランペット

    # --- Edible (30) ---
    {"name": "Oenanthe javanica",       "edibility": "edible"},   # セリ
    {"name": "Allium victorialis",      "edibility": "edible"},   # ギョウジャニンニク
    {"name": "Artemisia indica",        "edibility": "edible"},   # ヨモギ
    {"name": "Taraxacum officinale",    "edibility": "edible"},   # タンポポ
    {"name": "Urtica dioica",           "edibility": "edible"},   # イラクサ
    {"name": "Rumex acetosa",           "edibility": "edible"},   # スイバ
    {"name": "Plantago asiatica",       "edibility": "edible"},   # オオバコ
    {"name": "Chenopodium album",       "edibility": "edible"},   # シロザ
    {"name": "Portulaca oleracea",      "edibility": "edible"},   # スベリヒユ
    {"name": "Nasturtium officinale",   "edibility": "edible"},   # クレソン
    {"name": "Cardamine flexuosa",      "edibility": "edible"},   # タネツケバナ
    {"name": "Fallopia japonica",       "edibility": "edible"},   # イタドリ
    {"name": "Oxalis acetosella",       "edibility": "edible"},   # カタバミ
    {"name": "Stellaria media",         "edibility": "edible"},   # ハコベ
    {"name": "Capsella bursa-pastoris", "edibility": "edible"},   # ナズナ
    {"name": "Pueraria montana",        "edibility": "edible"},   # クズ
    {"name": "Osmunda japonica",        "edibility": "edible"},   # ゼンマイ
    {"name": "Matteuccia struthiopteris","edibility": "edible"},  # コゴミ
    {"name": "Allium tuberosum",        "edibility": "edible"},   # ニラ
    {"name": "Houttuynia cordata",      "edibility": "edible"},   # ドクダミ
    {"name": "Rubus idaeus",            "edibility": "edible"},   # 木イチゴ
    {"name": "Vaccinium myrtillus",     "edibility": "edible"},   # ブルーベリー
    {"name": "Fragaria vesca",          "edibility": "edible"},   # ワイルドストロベリー
    {"name": "Rosa canina",             "edibility": "edible"},   # ノイバラ
    {"name": "Morus alba",              "edibility": "edible"},   # クワ
    {"name": "Actinidia arguta",        "edibility": "edible"},   # サルナシ
    {"name": "Phyllostachys edulis",    "edibility": "edible"},   # タケノコ
    {"name": "Glycine soja",            "edibility": "edible"},   # 野生大豆
    {"name": "Persicaria hydropiper",   "edibility": "edible"},   # ヤナギタデ
    {"name": "Pteridium aquilinum",     "edibility": "edible"},   # ワラビ（加熱済）
]

# ---------------------------------------------------------------
# Unknown pools — NOT in DB, for "match: null" training samples
# ---------------------------------------------------------------
UNKNOWN_MUSHROOMS: List[str] = [
    "Mycena haematopus",
    "Coprinus comatus",
    "Trametes versicolor",
    "Phallus impudicus",
    "Xylaria polymorpha",
    "Neoboletus luridiformis",
    "Laccaria amethystina",
    "Inonotus obliquus",
    "Stereum hirsutum",
    "Daedalea quercina",
]

UNKNOWN_PLANTS: List[str] = [
    "Lamium purpureum",
    "Veronica persica",
    "Geranium thunbergii",
    "Epilobium angustifolium",
    "Impatiens noli-tangere",
    "Lysimachia japonica",
    "Circaea lutetiana",
    "Potentilla reptans",
    "Vicia cracca",
    "Lathyrus pratensis",
]


async def get_taxon_id(scientific_name: str) -> Optional[int]:
    """Get iNaturalist taxon ID by scientific name."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
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
        print(f"  Error getting taxon ID for {scientific_name}: {e}")
        return None


async def download_images(
    scientific_name: str,
    save_dir: Path,
    max_count: int
) -> int:
    """Download research-grade images from iNaturalist."""
    taxon_id = await get_taxon_id(scientific_name)
    if not taxon_id:
        print(f"  ⚠️  Taxon not found: {scientific_name}")
        return 0

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{BASE_URL}/observations", params={
                "taxon_id": taxon_id,
                "quality_grade": "research",
                "photos": "true",
                "per_page": max_count,
                "order": "votes",
                "order_by": "votes"
            })
        observations = resp.json().get("results", [])
    except Exception as e:
        print(f"  ✗ Error fetching observations: {e}")
        return 0

    if not observations:
        print(f"  ⚠️  No observations: {scientific_name}")
        return 0

    save_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    seen_photos = set()

    async with httpx.AsyncClient(timeout=30.0) as client:
        for obs in observations:
            for photo in obs.get("photos", []):
                if photo["id"] in seen_photos:
                    continue
                seen_photos.add(photo["id"])

                url = photo["url"].replace("square", "medium")
                try:
                    img_resp = await client.get(url)
                    if img_resp.status_code == 200:
                        filepath = save_dir / f"{obs['id']}_{photo['id']}.jpg"
                        with open(filepath, "wb") as f:
                            f.write(img_resp.content)
                        downloaded += 1
                except Exception as e:
                    print(f"    Photo error: {e}")

            await asyncio.sleep(RATE_LIMIT_DELAY)

    return downloaded


async def _download_known_species(
    sp: Dict,
    category: str,
    semaphore: asyncio.Semaphore,
    bar: tqdm,
) -> int:
    """Download one known species (used in parallel gather)."""
    sci_name = sp["name"]
    save_dir = IMAGES_DIR / category / sci_name.replace(" ", "_")

    if save_dir.exists():
        existing = len(list(save_dir.glob("*.jpg")))
        if existing >= 10:
            bar.set_postfix_str(f"SKIP {sci_name}")
            bar.update(1)
            return 0

    async with semaphore:
        bar.set_postfix_str(sci_name[:35])
        count = await download_images(sci_name, save_dir, MAX_IMAGES_PER_SPECIES)
        bar.update(1)
        return count


async def _download_unknown_species(
    sci_name: str,
    save_dir: Path,
    semaphore: asyncio.Semaphore,
    bar: tqdm,
) -> int:
    """Download one unknown species into shared folder (used in parallel gather)."""
    async with semaphore:
        bar.set_postfix_str(sci_name[:35])
        count = await download_images(sci_name, save_dir, MAX_IMAGES_UNKNOWN_PER_SPECIES)
        bar.update(1)
        return count


async def download_category(
    species_list: List[Dict],
    category: str,
    label: str,
) -> int:
    """Download all known species for a category (parallel, SPECIES_CONCURRENCY at once)."""
    print(f"\n{'=' * 60}")
    print(f"{label} ({len(species_list)} species, concurrency={SPECIES_CONCURRENCY})")
    print("=" * 60)

    semaphore = asyncio.Semaphore(SPECIES_CONCURRENCY)
    with tqdm(total=len(species_list), desc=label, unit="sp") as bar:
        tasks = [
            _download_known_species(sp, category, semaphore, bar)
            for sp in species_list
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    total = sum(r for r in results if isinstance(r, int))
    errors = sum(1 for r in results if isinstance(r, Exception))
    print(f"  → {total} new images downloaded, {errors} errors")
    return total


async def download_unknown_pool(
    species_list: List[str],
    category: str,
    label: str,
) -> int:
    """Download unknown pool images into a single mixed folder (parallel)."""
    print(f"\n--- {label} ---")
    save_dir = IMAGES_DIR / "unknown" / category
    save_dir.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(SPECIES_CONCURRENCY)
    with tqdm(total=len(species_list), desc=label, unit="sp") as bar:
        tasks = [
            _download_unknown_species(sci_name, save_dir, semaphore, bar)
            for sci_name in species_list
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    total = sum(r for r in results if isinstance(r, int))
    print(f"  → {total} unknown images")
    return total


async def main():
    print("=" * 60)
    print("SurviveOrDie — Image Downloader")
    print(f"  Known mushrooms : {len(KNOWN_MUSHROOMS)} species × {MAX_IMAGES_PER_SPECIES} images")
    print(f"  Known plants    : {len(KNOWN_PLANTS)} species × {MAX_IMAGES_PER_SPECIES} images")
    print(f"  Unknown pool    : {len(UNKNOWN_MUSHROOMS) + len(UNKNOWN_PLANTS)} species (mixed)")
    print("=" * 60)

    # Known species
    m_total = await download_category(KNOWN_MUSHROOMS, "mushroom", "Known Mushrooms")
    p_total = await download_category(KNOWN_PLANTS, "plant", "Known Plants")

    # Unknown pools
    um_total = await download_unknown_pool(UNKNOWN_MUSHROOMS, "mushroom", "Unknown Mushroom Pool")
    up_total = await download_unknown_pool(UNKNOWN_PLANTS, "plant", "Unknown Plant Pool")

    # Write species_list.json for 02_generate_species_db_*.py
    species_list = []
    for sp in KNOWN_MUSHROOMS:
        species_list.append({
            "scientific_name": sp["name"],
            "category": "mushroom",
            "edibility": sp["edibility"]
        })
    for sp in KNOWN_PLANTS:
        species_list.append({
            "scientific_name": sp["name"],
            "category": "plant",
            "edibility": sp["edibility"]
        })

    species_list_path = Path("data/species_list.json")
    species_list_path.parent.mkdir(parents=True, exist_ok=True)
    with open(species_list_path, "w", encoding="utf-8") as f:
        json.dump(species_list, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print("✓ All downloads complete!")
    print(f"  Known mushrooms : {m_total} new images")
    print(f"  Known plants    : {p_total} new images")
    print(f"  Unknown pool    : {um_total + up_total} images")
    print(f"  species_list.json written: {len(species_list)} species")
    print("=" * 60)
    print("\nNext: Run 02_generate_species_db_*.py")


if __name__ == "__main__":
    asyncio.run(main())
