"""
Upload dataset to HuggingFace Hub - simplified version.
"""
import os
import json
from pathlib import Path

# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_USERNAME = os.environ.get("HF_USERNAME", "")
DATASET_NAME = "AiOrDie-dataset"
REPO_ID = f"{HF_USERNAME}/{DATASET_NAME}"

if not HF_TOKEN:
    print("⚠️  HF_TOKEN not set!")
    exit(1)

if not HF_USERNAME:
    print("⚠️  HF_USERNAME not set!")
    exit(1)

print(f"Username: {HF_USERNAME}")
print(f"Dataset: {DATASET_NAME}")

# Use huggingface_hub library
from huggingface_hub import login, HfApi

# Login
login(token=HF_TOKEN)

api = HfApi(token=HF_TOKEN)

# Create repository (or update)
try:
    api.repo_info(REPO_ID, repo_type="dataset")
    print(f"✓ Repository exists: {REPO_ID}")
except:
    print(f"Creating repository: {REPO_ID}")
    api.create_repo(
        repo_id=DATASET_NAME,
        repo_type="dataset",
        private=False,
        token=HF_TOKEN
    )

# Files to upload
files_to_upload = [
    ("data/species_db.json", "species_db.json"),
    ("data/train_paths.jsonl", "train_paths.jsonl"),
    ("data/val_paths.jsonl", "val_paths.jsonl"),
    ("data/test_paths.jsonl", "test_paths.jsonl"),
]

print("\nUploading files to HuggingFace...")
for local_path, repo_path in files_to_upload:
    if os.path.exists(local_path):
        with open(local_path, "rb") as f:
            try:
                api.upload_file(
                    path_or_fileobj=f,
                    path_in_repo=repo_path,
                    repo_id=DATASET_NAME,
                    repo_type="dataset",
                    token=HF_TOKEN
                )
                print(f"  ✓ Uploaded: {repo_path}")
            except Exception as e:
                print(f"  ✗ Error uploading {repo_path}: {e}")
    else:
        print(f"  ⚠️  Not found: {local_path}")

# Create dataset card
dataset_card = """---
license: mit
task_categories:
- image-classification
- visual-question-answering
language:
- en
- ja
tags:
- survival
- plants
- mushrooms
- safety
---

# AiOrDie Dataset

Dataset for training AI models to identify toxic plants and mushrooms.

## Dataset Structure

```
data/
├── species_db.json          # Species metadata (toxicity, lookalikes, first aid)
├── train_paths.jsonl       # Training samples (chat format with image paths)
├── val_paths.jsonl         # Validation samples
└── test_paths.jsonl        # Test samples

images/
├── mushroom/               # Real mushroom images from iNaturalist
└── plant/                 # Real plant images from iNaturalist
```

## Format

Each sample is a chat conversation:
- User: Image + question
- Assistant: Function call to species_db
- Tool: DB response
- Assistant: Final survival advice

## Statistics

- 12 species (6 mushrooms, 6 plants)
- 296 real images from iNaturalist
- 498 total samples
  - Train: 404
  - Val: 39
  - Test: 55

## Categories

- LETHAL: Deadly species
- DO_NOT_EAT: Poisonous or has toxic lookalike
- SAFE: Edible with low risk

## License

MIT
"""

with open("data/README.md", "w", encoding="utf-8") as f:
    f.write(dataset_card)

print("\nUploading README...")
try:
    with open("data/README.md", "rb") as f:
        api.upload_file(
            path_or_fileobj=f,
            path_in_repo="README.md",
            repo_id=DATASET_NAME,
            repo_type="dataset",
            token=HF_TOKEN
        )
    print("  ✓ Uploaded: README.md")
except Exception as e:
    print(f"  ✗ Error uploading README: {e}")

print(f"\n✓ Upload complete!")
print(f"  URL: https://huggingface.co/datasets/{REPO_ID}")
