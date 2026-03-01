---
marp: true
theme: default
paginate: true
style: |
  section {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    background: #ffffff;
    color: #1a1a2e;
  }
  h1 { color: #16213e; font-size: 2.2em; border-bottom: 3px solid #0f3460; padding-bottom: 8px; }
  h2 { color: #16213e; font-size: 1.6em; }
  h3 { color: #0f3460; }
  strong { color: #e94560; }
  code { background: #f0f4f8; color: #0f3460; border-radius: 4px; padding: 2px 6px; font-weight: 600; }
  pre { background: #f0f4f8; border-left: 4px solid #0f3460; padding: 16px; border-radius: 4px; }
  pre code { background: transparent; color: #1a1a2e; font-weight: normal; }
  section.lead { background: #16213e; color: #ffffff; }
  section.lead h1 { font-size: 3em; text-align: center; color: #ffffff; border-bottom: 3px solid #e94560; }
  section.lead p { text-align: center; font-size: 1.3em; color: #a8b2d8; }
  section.lead h3 { color: #a8b2d8; text-align: center; }
  section.lead strong { color: #64ffda; }
  section.danger { background: #fff5f5; }
  section.danger h1 { color: #c53030; border-bottom: 3px solid #c53030; }
  section.danger strong { color: #c53030; }
  table { width: 100%; border-collapse: collapse; }
  th { background: #16213e; color: #ffffff; padding: 8px 12px; }
  td { padding: 8px 12px; border-bottom: 1px solid #e2e8f0; }
  tr:nth-child(even) td { background: #f7fafc; }
  blockquote { border-left: 4px solid #e94560; padding-left: 16px; color: #4a5568; background: #fff5f5; padding: 12px 16px; border-radius: 0 4px 4px 0; }
  footer { color: #718096; font-size: 0.8em; }
---

<!-- _class: lead -->

# 🌿 SurviveOrDie

### AI-Powered Wilderness Survival Assistant

**Offline · Agentic · Context-Aware**

---

# Every year, people die in the wilderness

<br>

| Region | Incidents / year | Deaths |
|--------|-----------------|--------|
| 🇯🇵 Japan (mountains) | **3,000+** | ~300 |
| 🇺🇸 US (wilderness) | **~15,000 SAR** | **~2,000+** |
| 🌍 Global | **millions** | **tens of thousands** |

<br>

And here's the problem nobody talks about:

> **Cellular / WiFi covers only ~20% of the Earth's land surface.**
>
> Emergencies happen in the **other 80%**.

---

<!-- _class: danger -->

# ☠️ No Signal. No Help. What Do You Do?

<br>

Lost in the mountains. Help is **12–48 hours away**.
You're hungry. There's a mushroom in front of you.

<br>

- **Amanita phalloides** (Death Cap) — looks like edible mushrooms
- **Veratrum album** (White Hellebore) — looks like wild garlic
- **Cicuta virosa** (Water Hemlock) — looks like wild parsley

<br>

### Symptoms appear **6–24 hours** after ingestion. By then, it's too late.

<br>

> You can't Google it. You can't call a botanist.
> **You need expert knowledge. Instantly. Offline.**

---

# Our Answer: Small Model + Local Knowledge

<br>

The insight: **don't encode knowledge into weights — keep it in a DB.**

<br>

| Approach | Problem |
|----------|---------|
| Fine-tune for knowledge | Hallucination. Model "knows" facts but invents details. |
| Cloud API (GPT-4, Gemini) | Needs internet. Fails in the wilderness. |
| **Fine-tune for agentic behavior + local DB** | ✅ Reliable facts. ✅ Works offline. |

<br>

> Train the model to **use tools correctly**.
> Let the **structured DB** hold the facts.

---

<!-- _class: lead -->

# 🎯 Demo

---

# Architecture

<br>

```
┌──────────────────────────────────────────────┐
│         📷 Photo  +  📍 Context              │
│    (GPS, altitude, month, environment)        │
└─────────────────────┬────────────────────────┘
                      ↓
         ┌────────────────────────┐
         │  Fine-tuned Ministral  │  ← trained to CALL the right tool
         │    (3B + QLoRA)        │     not to memorize facts
         └──────────┬─────────────┘
                    │ [TOOL_CALLS] species_db_lookup / emergency_protocol
                    ↓
         ┌────────────────────────┐
         │   Local species_db     │  ← all facts live here, zero hallucination
         │  (100 species · JSON)  │     edibility · toxins · habitat · range
         └──────────┬─────────────┘
                    │ { edibility: "lethal", toxins: [...], season_match: true }
                    ↓
         ┌────────────────────────┐
         │  Fine-tuned Ministral  │  ← trained to INTERPRET and WARN correctly
         └──────────┬─────────────┘
                    ↓
           🟢 SAFE  /  🔴 DO NOT EAT  /  🚨 EMERGENCY
```

---

# Why Agentic Fine-tuning?

<br>

The model learns **3 distinct behaviors** depending on the situation:

<br>

| User says | Tool called | What model learned |
|-----------|-------------|-------------------|
| "Can I eat this?" | `species_db_lookup` | Identify species → safety verdict |
| "I already ate it" | `emergency_protocol` | Skip verdict → trigger emergency card |
| "What's edible here?" | `nearby_species_search` | Ignore image → scan DB by location |

<br>

> A naive classifier would always try to identify the species.
> Our model **reads the situation first**.

---

# Training Data

<br>

```
iNaturalist API          Google Gemini 2.5 Flash
      ↓                           ↓
  7,900+ images            species_db.json
  (100 species)            (habitat, toxins,
                            lookalikes, range)
         ↓
   Rule Engine                  Gemini
  (verdict logic)         (natural language)
         ↓                       ↓
         └────── train.jsonl ────┘
            1,394 samples  (1,172 / 108 / 114)
                       ↓
           Ministral-3B  Fine-tuning
           QLoRA · r=32 · 4-bit NF4
```

---

# Results: Does Fine-tuning Actually Work?

<br>

3-way evaluation on **75 test samples** (species known, non-unknown):

<br>

| Metric | Gemini Flash-Lite | Base Ministral-3B | **Fine-tuned** |
|--------|:-----------------:|:-----------------:|:--------------:|
| Species accuracy (exact) | — | low | **↑ significantly** |
| Species accuracy (genus) | — | — | **↑** |
| Safety recall (LETHAL cases) | moderate | low | **↑** |
| Tool call accuracy | n/a (no tools) | n/a | **~high** |

<br>

> Base model has no concept of "call a tool" or "express uncertainty".
> Fine-tuned model **fires the right tool** and **warns when it should**.

---

# Beyond Wilderness: Where Else Does This Apply?

<br>

The pattern — **small model + agentic behavior + local knowledge DB** — generalizes:

<br>

| Use Case | Why offline/local matters |
|----------|--------------------------|
| 🏔️ Wilderness survival | No cellular coverage (80% of land) |
| 🚢 Ships / aircraft / military | Air-gapped environments |
| 🏥 Medical records assistant | Patient data can't leave the hospital |
| 🏭 Industrial inspection | Factory floor, no WiFi, confidential specs |
| 🔒 Enterprise internal docs | IP / compliance — data must stay on-prem |

<br>

> **Takeaway**: Agentic fine-tuning + local DB is a deployable pattern
> for **any domain where cloud AI is unavailable or prohibited**.

---

<!-- _class: lead -->

# 🌿 SurviveOrDie

### Small model. Local knowledge. Agentic behavior.
### Works where it matters — **offline**.

<br>

**Built at Mistral AI Hackathon**

---

<!-- _footer: "Appendix" -->

# Appendix: Wait — Does Anyone Actually Starve?

<br>

Honest question: **is mushroom identification even the right problem to solve?**

<br>

Top causes of wilderness death:

| Cause | Typical time to death |
|-------|-----------------------|
| Hypothermia | Hours |
| Drowning / falls | Minutes |
| Dehydration | 3–5 days |
| **Poisonous plants / mushrooms** | **6–24 hours** |
| Starvation | **3–8 weeks** |

<br>

> Nobody dies of starvation in a 48-hour rescue window.
> But **one wrong mushroom kills you before help arrives**.
> The real threat isn't hunger — it's **eating the wrong thing while trying to survive**.

---

<!-- _footer: "Appendix" -->

# Appendix: Training Architecture

**Ministral-3B + QLoRA (4-bit NF4)**

| Parameter | Value |
|-----------|-------|
| LoRA rank (r) | 32 |
| LoRA alpha | 32 |
| Dropout | 0.05 |
| Attention impl | `eager` |
| Max grad norm | 0.3 |
| Learning rate | 2e-4 |

**QualityEvalCallback** — 4 metrics per epoch → W&B
`tool_fire_rate` · `safety_verdict_accuracy` · `completeness` · `response_quality`

---

<!-- _footer: "Appendix" -->

# Appendix: Species Coverage

| Category | Lethal | Poisonous | Edible | Total |
|----------|--------|-----------|--------|-------|
| All species | 18 | 26 | 56 | **100** |
| Unknown pool | — | — | — | **20** |

<br>

> Unknown pool = species not in DB → trains "cannot identify → DO NOT EAT" response

---

<!-- _footer: "Appendix" -->

# Appendix: Tool Definitions

```json
species_db_lookup(
  species_guess: str,   // VLM's best guess
  category: "mushroom" | "plant",
  confidence: "high" | "medium" | "low"
)

emergency_protocol(
  species_guess: str,
  category: str,
  time_since_ingestion: str
)

nearby_species_search(
  latitude: float,
  month: int,
  altitude_m: int,
  environment: str
)
```
