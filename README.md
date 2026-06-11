[**中文**](README.cn.md) · **English**

# Singer · Classical Chinese Poetry LLM

> **Teaching AI to write real poetry** — not just stringing elegant words together, but faithfully observing tonal patterns (平仄), parallelism (对仗), and rhyme schemes that define classical Chinese verse.

Singer is a complete training and evaluation pipeline for **classical Chinese regulated verse** (律诗 and 绝句). Built on [Qwen3-4B-Instruct](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507) and distilled from tens of thousands of canonical poems, it undergoes two stages of refinement — **SFT (Supervised Fine-Tuning)** and **GRPO (Group Relative Policy Optimization)** — to produce a domain-specific model that composes verse on demand, following both inspiration and prosodic rules.

Whether you're an NLP researcher, a poetry enthusiast, or looking to integrate a "poetry-writing AI" into your application, Singer provides an **out-of-the-box pipeline** from data to model, training to evaluation.

---

## Why Singer?

| Capability | Description |
|-----------|-------------|
| **Prosody-First Training** | Training objectives center on tone patterns (平仄), rhyme, and parallelism derived from the *Pingshui Rhyme System* (《平水韵》), not generic text completion |
| **4-Dimension Expert Scoring** | Prosodic correctness (40), Parallelism & Structure (20), Language & Polish (20), Artistic Conception & Depth (20) — total score 100 |
| **Inspiration-Driven Composition** | Input what you see, hear, and feel; the model outputs a complete poem — mirroring the real creative process |
| **Complete Workflow** | Data cleaning → SFT → GRPO → batch evaluation → log analysis, every step included |
| **Consumer GPU Friendly** | Pre-tuned configurations for RTX 2080 Ti and RTX 4090; 4-bit LoRA for low-memory training |
| **High-Quality Corpus** | Built on the open-source [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry) project, curated from *Quan Tang Shi* (全唐诗) and other classical collections — 50,000 selected poems spanning heptasyllabic regulated verse (七律), pentasyllabic regulated verse (五律), heptasyllabic quatrains (七绝), and pentasyllabic quatrains (五绝) |

---

## How It Works

```mermaid
flowchart LR
    A[Classical Poetry Corpus] --> B[Data Cleaning Pipeline]
    B --> C[SFT Supervised Fine-Tuning]
    C --> D[GRPO Reinforcement Learning]
    D --> E[4-Dimension Evaluation]
    E --> F[Regulated Verse Model]

    B -.->|Spark Inspiration Field| C
    E -.->|DeepSeek Scoring| D
```

**Core idea**: First teach the model *how* to write poems with high-quality examples, then use LLM-as-judge scoring signals (as GRPO rewards) to guide the model toward greater prosodic accuracy and artistic quality.

---

## Quick Start

### 1. Environment Setup

```bash
git clone <your-repo-url>
cd chinese-poetry-llm

python -m venv .venv && source .venv/bin/activate

# Install PyTorch per your CUDA version: https://pytorch.org
pip install torch
pip install -r requirements.txt
```

Set environment variables (in your shell or a `.env` file):

```bash
export DEEPSEEK_API_KEY="your-deepseek-api-key"        # Scoring & GRPO reward
export HUGGINGFACE_API_KEY="your-huggingface-token"    # Base model download
```

### 2. Prepare Data

The training/evaluation CSVs require two columns:

- **`Spark`** — creative inspiration / context (model input)
- **`Content`** — the poem text, one line per sentence (training label)

| File | Purpose |
|------|---------|
| `data/dataset_sft.csv` | SFT training set |
| `data/dataset_grpo.csv` | GRPO training set |
| `data/dataset_test.csv` | Evaluation set |

The repo includes a sample at [`data/sample.csv`](data/sample.csv). To run a quick test, copy it three times:

```bash
cp data/sample.csv data/dataset_sft.csv
cp data/sample.csv data/dataset_grpo.csv
cp data/sample.csv data/dataset_test.csv
```

### 3. One-Command Training & Evaluation

Run the following from the **project root**:

```bash
# Stage 1: Supervised Fine-Tuning
python sft/train_sft.py

# Stage 2: GRPO Reinforcement Learning (requires DeepSeek API)
python grpo/train_GRPO.py

# Batch Evaluation
python eval/eval.py

# Analyze evaluation logs (score distribution visualizations, etc.)
python eval/log_analyzer.py
```

GRPO training artifacts are written to `grpo_outputs/` and `grpo_lora_adapters/` by default.

---

## Data Pipeline

Process raw classical poetry from the [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry) project into usable training sets by executing the scripts in `dataset/` in order:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `1_convert.py` | Normalize fields and encoding |
| 2 | `1_rough_genre.py` | Coarse classification (poetic form identification) |
| 3 | `2_strict_genre.py` | Fine classification and filtering |
| 4 | `3_score_and_spark.py` | LLM-based 4-dimension scoring + `Spark` generation |
| 5 | `4_analyze_and_select.py` | Score-based filtering and subset export |

**Selection Reference** (based on LLM scores):

| Threshold | Size | Recommended Use |
|-----------|------|-----------------|
| Score ≥ 90 | ~2,000+ poems (avg ~92) | Premium subset, high-quality SFT |
| Score ≥ 85 | ~34,000+ poems (avg ~87) | Large-scale training |

Final dataset genre distribution (50K curated set):

| Genre | Count | Percentage |
|-------|-------|------------|
| 七律 (Heptasyllabic Regulated Verse) | 23,877 | 47.8% |
| 五律 (Pentasyllabic Regulated Verse) | 14,317 | 28.6% |
| 七绝 (Heptasyllabic Quatrain) | 10,323 | 20.7% |
| 五绝 (Pentasyllabic Quatrain) | 1,483 | 3.0% |

Detailed statistics in [`dataset/analysis/genre_distribution.md`](dataset/analysis/genre_distribution.md).

---

## Scoring System

[`poetry_core/poetry_evaluator.py`](poetry_core/poetry_evaluator.py) calls the DeepSeek API to score poems from the perspective of a classical poetry expert:

| Dimension | Points | Criteria |
|-----------|--------|----------|
| Prosodic Correctness (格律规范性) | 40 | Tone patterns (平仄), rhyme, line count, character count |
| Parallelism & Structure (对仗与结构) | 20 | Couplet parallelism (颔联/颈联), structural flow (起承转合) |
| Language & Polish (语言与锤炼) | 20 | Word choice precision, poetic "eye" (诗眼), avoiding forced rhymes |
| Artistic Conception & Depth (意境与立意) | 20 | Imagery coherence, genuine emotion, depth of meaning |

During GRPO, the **total score serves as the core reward signal**, complemented by format rewards (line count, character count compliance), striking a balance between poems that *look like poetry* and poems that *are good poetry*.

---

## Project Structure

```
chinese-poetry-llm/
├── poetry_core/          # Shared core: data loading, generation, scoring, logging
│   ├── poetry_data_loader.py
│   ├── poetry_generator.py
│   ├── poetry_evaluator.py
│   └── poetry_logger.py
├── dataset/              # Data cleaning pipeline (steps 1–5)
│   └── analysis/
├── sft/                  # Supervised Fine-Tuning (Unsloth + LoRA)
├── grpo/                 # GRPO Reinforcement Learning (TRL + DeepSeek reward)
├── eval/                 # Batch evaluation and log analysis
├── data/                 # Data directory (includes sample.csv)
└── requirements.txt
```

---

## Configuration

Each stage has its own configuration file. Modify as needed:

| Option | Location | Description |
|--------|----------|-------------|
| `GPU_FLAG` | `sft/sft_config.py`, `grpo/grpo_config.py`, `eval/eval_config.py` | `RTX2080Ti` / `RTX4090` / `DeepSeek` (eval only) |
| `MODEL_NAME` | Environment variable | Default: `unsloth/Qwen3-4B-Instruct-2507` |
| `IDX_START` / `IDX_END` | Each config | Data slice range |
| Data paths | Environment variables | `SFT_DATA_PATH`, `GRPO_DATA_PATH`, `EVAL_DATA_PATH` |

Batch sizes, mixed precision, Flash Attention, and other parameters are pre-tuned for RTX 2080 Ti and RTX 4090 — just switch `GPU_FLAG`.

---

## Sample Output

Input (in the style of the `Spark` field):

> Strolling the outskirts on a spring day, I see crumbling walls intertwined with new greenery, hear birdsong in the distance, and think of war still raging and letters from home cut off — a mix of sorrow and hope.

Model output (one line per verse, no title or attribution):

```
国破山河在，城春草木深。
感时花溅泪，恨别鸟惊心。
烽火连三月，家书抵万金。
白头搔更短，浑欲不胜簪。
```

(The example above is taken from [`data/sample.csv`](data/sample.csv) for format reference.)

---

## Tech Stack

- **Base Model**: Qwen3-4B-Instruct (Unsloth-accelerated)
- **Fine-Tuning**: LoRA (rank 16) + 4-bit quantization
- **Reinforcement Learning**: TRL `GRPOTrainer`
- **Reward Model**: DeepSeek API 4-dimension scoring
- **Experiment Tracking**: SwanLab (optional)

---

## FAQ

**Q: Can I run this without a GPU?**  
For evaluation, set `GPU_FLAG` to `DeepSeek` in `eval/eval_config.py` to use the API for both generation and scoring. SFT and GRPO still require a CUDA GPU.

**Q: Minimum VRAM requirements?**  
RTX 2080 Ti (22 GB) has been verified to complete the full pipeline with 4-bit LoRA.

**Q: Is DeepSeek required?**  
Scoring and GRPO rewards currently use the DeepSeek API. The generation module also supports local model inference — you can swap out the API client in [`poetry_core/`](poetry_core/).

---

## License

MIT License

---

## Acknowledgements

- Training data derived from the [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry) open-source project — our thanks to its contributors for their long-standing dedication to digitizing classical Chinese poetry.
- Base model [Qwen3-4B-Instruct](https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507) and the Unsloth training acceleration tools.

---

<p align="center">
  <strong>Singer</strong> — Classical prosody, new voice in AI<br>
  <sub>If this project helps you, consider giving it a Star ⭐</sub>
</p>
