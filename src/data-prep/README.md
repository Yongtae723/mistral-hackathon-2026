# SurviveOrDie — Data Prep Pipeline

VLM fine-tuning用の学習データを生成するパイプライン。

---

## パイプライン概要

```
iNaturalist API
      ↓
01_download_images.py        # 画像取得
      ↓
02_generate_species_db_*.py  # 種データベース生成
      ↓
03_generate_training_data_*.py  # 学習データ生成 (train/val/test.jsonl)
      ↓
04_upload_to_hf.py           # Hugging Face にアップロード
```

`*_mock` = LLM不要（テンプレート、動作確認用）
`*_vertex` = Vertex AI Gemini 使用（本番用）

---

## 対象種（12種）

| 種名 | 通称 | 区分 |
|-----|------|------|
| Amanita phalloides | タマゴテングタケ / Death Cap | 致死 |
| Amanita virosa | シロタマゴテングタケ / Destroying Angel | 致死 |
| Galerina marginata | 猛毒カキノタケ / Deadly Galerina | 致死 |
| Aconitum napellus | トリカブト / Monkshood | 致死 |
| Cicuta virosa | ドクゼリ / Water Hemlock | 致死 |
| Veratrum album | バイケイソウ / White False Hellebore | 致死 |
| Cantharellus cibarius | アンズタケ / Chanterelle | 食用 |
| Boletus edulis | ヤマドリタケ / Porcini | 食用 |
| Grifola frondosa | マイタケ / Maitake | 食用 |
| Oenanthe javanica | セリ / Japanese Parsley | 食用 |
| Allium victorialis | ギョウジャニンニク / Victory Onion | 食用 |
| Artemisia indica | ヨモギ / Mugwort | 食用 |

画像数: 合計 **296枚**（種ごと 13〜46枚、平均25枚、iNaturalist から取得）

---

## 学習データの構造

### メッセージフォーマット（Mistral SFT）

各サンプルは 4ターンの messages 配列：

```
user    → 画像(base64) + コンテキスト文字列
assistant → tool_call（function calling）
tool    → DB lookup 結果（JSON）
assistant → 最終回答
```

コンテキスト文字列の例：
```
lat: 35.5, month: 10, alt: 800m, env: broadleaf_forest, growing: on_ground
この辺で食べられるものは何がありますか？
```

### 3種類のツール

| ツール | 用途 | 最終出力形式 |
|--------|------|------------|
| `species_db_lookup` | 種の識別・安全性判定 | テキスト |
| `emergency_protocol` | 摂取済み緊急対応 | JSON（UIで緊急カード表示） |
| `nearby_species_search` | 周辺の採取可能種を検索 | JSON（UIで種リスト表示） |

---

## 学習パターン（5種類）

モデルに状況判断力を持たせるため、コンテキストの一致度と質問内容によって5パターンを生成する。

| パターン | 検出条件 | tool_call | 目標比率 |
|---------|---------|-----------|---------|
| `high_confidence` | コンテキスト全一致 | species_db_lookup(confidence=high) | 50% |
| `medium_confidence` | 1項目不一致（季節 or 標高 or 生息地） | species_db_lookup(confidence=medium) | 20% |
| `low_confidence` | 2項目以上不一致 | species_db_lookup(confidence=low) | 10% |
| `emergency` | "already ate" / "食べました" 等を検出 | emergency_protocol | 10% |
| `foraging` | foraging質問（周辺に何がある？） | nearby_species_search | 10% |

**コンテキスト一致判定ロジック（`compute_context_match`）:**

```python
season_match   = context["month"] in species_data["season_months"]
latitude_match = lat_range[0] <= context["lat"] <= lat_range[1]
altitude_match = alt_range[0] <= context["altitude"] <= alt_range[1]
habitat_match  = context["env"] in species_data["habitat"]

mismatch_count = 上記 False の数
# 0 → high_confidence
# 1 → medium_confidence
# 2以上 → low_confidence
```

medium/low_confidence では、コンテキスト不一致情報が `tool response` の `context_match` フィールドに含まれ、最終回答でも不確実性・警告として反映される。

---

## train / val / test の分け方

**画像単位で分割**（種ごとに独立して分割）：

```python
train : val : test = 70% : 15% : 15%  (各種の画像に対して適用)
```

サンプル数（現在）：

| split | サンプル数 | tool内訳 |
|-------|----------|---------|
| train | 404 | species_db_lookup: 326, emergency: 33, foraging: 45 |
| val   | 39  | species_db_lookup: 39（high_confidence のみ） |
| test  | 55  | species_db_lookup: 55（high_confidence のみ） |

**val/test はシンプルに保つ方針**：パターンのバリエーションは train のみ。val/test は標準的な識別タスク（high_confidence）で過学習を検出する目的。

train は画像ごとに **2〜3バリエーション** を生成（異なるコンテキスト・質問・パターン）。

---

## 実行方法

プロジェクトルートから実行する。

### Step 1: 画像ダウンロード

```bash
python src/data-prep/01_download_images.py
# → images/{category}/{scientific_name}/*.jpg
```

すでに `images/` が存在する場合はスキップ可。

### Step 2: species DB 生成

```bash
# テスト用（LLM不要）
python src/data-prep/02_generate_species_db_mock.py
# → data/species_db.json

# 本番用（Vertex AI Gemini）
python src/data-prep/02_generate_species_db_vertex.py
# → data/species_db.json
```

`species_db.json` には各種の `season_months`, `distribution_lat_range`, `altitude_range_m`, `habitat`, `lookalikes` 等が含まれ、パターン判定の基準になる。

### Step 3: 学習データ生成

```bash
# テスト用（テンプレート回答、LLM不要）
python src/data-prep/03_generate_training_data_mock.py
# → data/train.jsonl, val.jsonl, test.jsonl

# 本番用（Gemini で自然文生成）
# 事前に: gcloud auth application-default login
python src/data-prep/03_generate_training_data_vertex.py
```

mock版の出力確認：

```bash
python3 -c "
import json
tools = {}
with open('data/train.jsonl') as f:
    for line in f:
        for m in json.loads(line)['messages']:
            for tc in m.get('tool_calls', []):
                n = tc['function']['name']
                tools[n] = tools.get(n, 0) + 1
print(tools)
# 期待: {'species_db_lookup': N, 'emergency_protocol': M, 'nearby_species_search': K}
"
```

### Step 4: Hugging Face アップロード

```bash
export HF_TOKEN=hf_xxxx
export HF_USERNAME=your_username
python src/data-prep/04_upload_to_hf.py
```

---

## 出力ファイル

| ファイル | 内容 | サイズ目安 |
|---------|------|---------|
| `data/species_db.json` | 12種のメタデータ（毒性・生息地・季節等） | 〜100KB |
| `data/train.jsonl` | 学習データ（base64画像埋め込み） | 〜70MB |
| `data/val.jsonl` | 検証データ | 〜6MB |
| `data/test.jsonl` | テストデータ | 〜9MB |

画像は全て base64 埋め込み（`data:image/jpg;base64,...`）。Mistral fine-tuning API はURLまたはbase64が必要なため。

---

## 環境

```bash
# 依存関係
pip install httpx google-auth huggingface_hub

# Vertex AI 認証（03_vertex 実行時）
gcloud auth application-default login
```
