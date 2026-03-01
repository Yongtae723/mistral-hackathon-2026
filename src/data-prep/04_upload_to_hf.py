"""
Upload training dataset to HuggingFace Hub.
Uploads train/val/test.jsonl (with base64-embedded images) + species_db.json.
"""
import os
from pathlib import Path
from huggingface_hub import HfApi, login
from tqdm import tqdm

# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN", "")
REPO_ID = "yongtae-jp/survive-or-die"
DATA_DIR = Path("data")

FILES_TO_UPLOAD = [
    DATA_DIR / "train.jsonl",
    DATA_DIR / "val.jsonl",
    DATA_DIR / "test.jsonl",
    DATA_DIR / "species_db.json",
]


def main():
    if not HF_TOKEN:
        print("✗ HF_TOKEN not set. Run: export HF_TOKEN=your_token")
        return

    print(f"Uploading to: {REPO_ID}")
    print(f"Files:")
    for f in FILES_TO_UPLOAD:
        size = f"{f.stat().st_size / 1024**3:.2f} GB" if f.stat().st_size > 1e9 else f"{f.stat().st_size / 1024**2:.1f} MB"
        print(f"  {f.name}: {size}")

    login(token=HF_TOKEN)
    api = HfApi()

    # Ensure repo exists (create if not, no-op if exists)
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)

    print("\nUploading...")
    for local_path in FILES_TO_UPLOAD:
        if not local_path.exists():
            print(f"  ⚠️  Not found: {local_path}")
            continue
        print(f"  → {local_path.name}")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=local_path.name,
            repo_id=REPO_ID,
            repo_type="dataset",
        )
        print(f"  ✓ {local_path.name}")

    print(f"\n✓ Done: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
