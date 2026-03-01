# AI Or Die — W&B Report

**Project:** SurviveOrDie / AI Or Die
**Author:** Yongtae
**Event:** Mistral AI Hackathon
**Date:** March 2026

---

## Overview

Lost in the wilderness, 12–48 hours from help, with a mushroom in front of you — that's the problem we set out to solve. **AI Or Die** is an offline-capable, agentic AI assistant that helps wilderness survivors identify potentially lethal species.

The core insight: **don't encode knowledge into model weights — keep it in a structured local DB**. Instead, fine-tune the model to *behave* correctly: call the right tool, interpret the result, and issue the right warning.

> Wi-Fi is available on only ~20% of the Earth's land surface. Emergencies happen in the other 80%.

---

## Architecture

```
Photo + Location Context  (image · GPS · season · environment)
              ↓
   Fine-tuned Ministral-3B  ←→  Local species_db (offline JSON)
              ↓
   🟢 SAFE  /  🔴 DO NOT EAT  /  🚨 EMERGENCY
```

The model is trained **not to memorize species facts**, but to:
1. Decide which tool to call (`species_db_lookup`, `emergency_protocol`, `nearby_species_search`)
2. Pass the right arguments (species guess, confidence, context)
3. Interpret the structured DB response into a human-readable safety verdict

---

## Training Setup

| Parameter | Value |
|-----------|-------|
| Base model | Ministral-3B |
| Method | QLoRA (4-bit NF4) |
| LoRA rank (r) | 32 |
| LoRA alpha | 32 |
| Dropout | 0.05 |
| Learning rate | 2e-4 |
| Max grad norm | 0.3 |
| Attention impl | `eager` |
| Training samples | 1,394 (1,172 train / 108 val / 114 test) |
| Species covered | 5 mushrooms × 5 safety categories |
| Images source | iNaturalist (7,900+ images) |

**Training data pipeline:**
iNaturalist images + species_db.json → Rule Engine (verdict logic) + Gemini 2.5 Flash (natural language) → agentic tool-calling conversations → `train.jsonl`

---

## Evaluation Results

3-way comparison on **75 test samples** using [W&B Weave](https://wandb.ai/weave):

| Metric | Gemini 2.5 Flash Lite | Base Ministral-3B | Fine-tuned Ministral-3B |
|--------|:---------------------:|:-----------------:|:-----------------------:|
| Safety recall (`warned`) | moderate | low | **↑ highest** |
| Species exact match | moderate | low | **↑ highest** |
| Species genus match | moderate | low | **↑ highest** |
| Tool call accuracy | n/a | n/a | **~high** |
| Latency | fast (API) | fast | moderate (local GPU) |

The radar chart (warned / exact_match / species classification / latency) shows the fine-tuned model dominating on safety and accuracy axes, with latency as the expected trade-off for running locally.

**Key finding:** The base model has no concept of "call a tool" or "express uncertainty." The fine-tuned model reliably fires the right tool and issues warnings when it should.

---

## What We'd Do Differently

### 1. Train with Task-Specific Metrics, Not Just Loss

The biggest gap in our training loop: **we only watched cross-entropy loss during training**, but the real performance signal is behavioral.

Loss going down does not necessarily mean the model is getting better at the actual task. We implemented a `QualityEvalCallback` that logged 4 metrics per epoch to W&B:

- `tool_fire_rate` — is the model calling the tool at all?
- `safety_verdict_accuracy` — when it does, is the verdict correct?
- `completeness` — is the response structurally complete?
- `response_quality` — overall coherence

**What we should have done:** Use these metrics as the *primary* training signal. Run a small held-out eval (50–100 samples) at the end of every epoch, and use `safety_verdict_accuracy` — especially on **lethal species cases** — as the early stopping criterion. A model that achieves low loss but misses the warning on *Amanita phalloides* is worse than useless.

This is the wilderness survival equivalent of RLHF reward shaping: the loss function doesn't know that getting a lethal species wrong is catastrophically worse than getting an edible one wrong.

```python
# What we had
trainer.train()  # watch loss on W&B

# What we should have done
class SafetyFirstCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        lethal_accuracy = eval_lethal_cases(model, lethal_test_set)
        wandb.log({"lethal_safety_accuracy": lethal_accuracy})
        if lethal_accuracy < 0.95:
            control.should_training_stop = False  # keep going
```

### 2. Data Augmentation

We trained on **5 mushroom species** due to Gemini API cost and time constraints at the hackathon. The final training set covers only the most representative safety categories:

| Species | Category |
|---------|----------|
| *Amanita phalloides* | LETHAL (Death Cap) |
| *Amanita muscaria* | DO_NOT_EAT (Fly Agaric) |
| *Flammulina velutipes* | CONFUSING (Enoki — wild has lethal lookalikes) |
| *Lentinula edodes* | CONDITIONAL_SAFE (Shiitake) |
| *Hericium erinaceus* | SAFE (Lion's Mane) |

**What needs to happen next:**

- **Expand to all 100 species** already in our species_db (18 lethal, 26 poisonous, 56 edible) — we have the pipeline, just need to run it
- **Add plant species** — currently only mushrooms were used in the final training run; 50 plant species are catalogued in species_db
- **Harder negatives** — more training examples of visually similar confusing pairs (e.g., *Flammulina velutipes* wild vs cultivated)
- **Image augmentation** — rotation, brightness shifts, partial occlusion to simulate real-world field conditions (bad lighting, partial view, blurry photo)
- **Multilingual contexts** — training data currently generates English responses, but user queries span Japanese, Korean, and Chinese (the GPS context reflects East Asian regions)
- **Unknown species training** — currently 20 "unknown" species are in the test pool; more "I cannot identify this → DO NOT EAT" examples would improve safety conservatism

---

## What Worked Well

- **Agentic fine-tuning pattern** — the model successfully learned 3 distinct behavioral modes (identify → lookup, emergency → skip to protocol, foraging → scan by location) from 1,394 training samples
- **Rule engine + LLM synthesis** — deterministic verdict logic ensures ground truth correctness, LLM only handles natural language packaging
- **enable/disable adapter layers** — using `enable_adapter_layers()` / `disable_adapter_layers()` instead of `merge_and_unload()` allows true A/B comparison between base and fine-tuned model in a single app session
- **Weave 3-way evaluation** — running Gemini, Base, and Fine-tuned through the same `weave.Evaluation` pipeline gives clean, comparable metrics on the same 75 test samples

---

## Next Steps

1. Expand training data to full 100 species + plant species
2. Implement safety-weighted loss: lethal misclassifications penalized 10×
3. Add warm accuracy metrics as primary early stopping criterion
4. Explore distillation from Mistral Large with vision → Ministral-3B for better visual grounding
5. Quantize to 4-bit GGUF for on-device deployment (no GPU required)

---

*Built at Mistral AI Hackathon · Model: Ministral-3B + QLoRA · Eval: W&B Weave*
