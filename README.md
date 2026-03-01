# AI Or Die — Wilderness Survival AI

**Offline · Agentic · Context-Aware**

📊 [Presentation Slides](https://www.canva.com/design/DAHCq0nIero/iFNST_k0hWE6QHEH5lJHQA/edit?utm_content=DAHCq0nIero&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) · 🤗 [Live Demo](https://huggingface.co/spaces/yongtae-jp/survive-or-die-app)

Lost in the wilderness with no signal? One wrong mushroom can kill you in 6 hours. **AI Or Die** identifies potentially lethal wild species from a photo and issues a survival verdict — no internet required.

> Wi-Fi covers only ~20% of the Earth's land surface. Emergencies happen in the other 80%.

Built at **Mistral AI Hackathon 2025**.

---

## How It Works

```
Photo + Location Context  (image · GPS · season · environment)
              ↓
   Fine-tuned Ministral-3B  ←→  Local species_db (offline JSON)
         [TOOL_CALLS]
              ↓
   🟢 SAFE  /  🔴 DO NOT EAT  /  🚨 EMERGENCY
```

The model is trained **not to memorize species facts**, but to call the right tool, interpret the DB result, and issue the correct warning. Facts live in a structured local DB — zero hallucination.

---

## Key Design Choices

| Approach | Problem |
|----------|---------|
| Fine-tune for knowledge | Hallucination — model invents facts |
| Cloud API (GPT-4, Gemini) | Needs internet — fails in the wilderness |
| **Fine-tune for agentic behavior + local DB** | ✅ Reliable facts. ✅ Works offline. |

---

## Model

- **Base:** Ministral-3B
- **Method:** QLoRA (4-bit NF4), r=32, alpha=32
- **Trained on:** 1,394 agentic tool-calling conversations
- **HF repo:** [Yongtae723/survive-or-die-lora](https://huggingface.co/Yongtae723/survive-or-die-lora)

### Tools the model learns to call

```
species_db_lookup(species_guess, category, confidence)
emergency_protocol(species_guess, category, time_since_ingestion)
nearby_species_search(latitude, month, altitude_m, environment)
```

---

## Training Data Pipeline

```
iNaturalist (7,900+ images)     Google Gemini 2.5 Flash
        ↓                               ↓
  5 mushroom species              species_db.json
  (one per safety category)       (toxins · habitat · range)
        ↓
  Rule Engine (verdict logic) + LLM (natural language)
        ↓
  train.jsonl — 1,394 samples (1,172 / 108 / 114)
        ↓
  Ministral-3B Fine-tuning
```

**Species used for training:**

| Species | Category |
|---------|----------|
| *Amanita phalloides* | LETHAL (Death Cap) |
| *Amanita muscaria* | DO_NOT_EAT (Fly Agaric) |
| *Flammulina velutipes* | CONFUSING (Enoki — wild has lethal lookalikes) |
| *Lentinula edodes* | CONDITIONAL_SAFE (Shiitake) |
| *Hericium erinaceus* | SAFE (Lion's Mane) |

---

## Evaluation

3-way comparison (Gemini 2.5 Flash Lite vs Base Ministral-3B vs Fine-tuned) on 75 test samples via **W&B Weave**:

| Metric | Gemini Flash Lite | Base Ministral-3B | Fine-tuned |
|--------|:-----------------:|:-----------------:|:----------:|
| Safety recall (lethal cases) | moderate | low | **↑ highest** |
| Species exact match | moderate | low | **↑ highest** |
| Tool call accuracy | n/a | n/a | **~high** |

---

## Project Structure

```
├── src/
│   ├── app/                        # Gradio demo app (HF Spaces)
│   │   └── app.py
│   ├── data-prep/                  # Training data pipeline
│   │   ├── 01_download_images.py   # iNaturalist API → images
│   │   ├── 02_generate_species_db_vertex.py  # LLM → species_db.json
│   │   ├── 03_generate_training_data_vertex.py  # Rule engine + LLM → train.jsonl
│   │   └── 04_upload_to_hf.py
│   ├── train/
│   │   └── train.py                # QLoRA fine-tuning
│   └── eval/
│       └── eval.py                 # Weave 3-way evaluation
├── notebooks/
│   └── inference_colab.ipynb       # Colab inference + eval notebook
├── artifacts/                      # LoRA adapter config
├── architecture.svg                # Architecture diagram
├── training_data.svg               # Training pipeline diagram
├── slides.md                       # Marp presentation
├── wandb_report.md                 # W&B report
└── requirements.txt
```

---

## Safety Principle

**"If in doubt, don't eat."**

- Lethal species → absolute refusal, no exceptions
- Species with dangerous lookalikes → DO NOT EAT even if edible
- Unknown species → DO NOT EAT
- Emergency ingestion → skip identification, trigger emergency protocol immediately
