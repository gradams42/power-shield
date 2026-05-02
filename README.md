# Power Shield — AYLA Pipeline

**An Automated Yield-Loss Alerting (AYLA) Pipeline Using Explainable AI (XAI) and Machine Learning (ML) for Power Grid Attack Detection and LLM-Guided Remediation**

Research paper: [`paper/power_shield_paper.pdf`](paper/power_shield_paper.pdf)

---

## Overview

Power Shield is a four-stage SCADA security pipeline that:

1. **Classifies** incoming grid events as Attack or Natural using a LightGBM + XGBoost ensemble with isotonic calibration (91.88% accuracy, AUC-ROC 0.9665, MCC 0.8020)
2. **Attributes** the alert to specific sensor readings using SHAP TreeExplainer
3. **Retrieves** relevant remediation protocols from a local vector database (ChromaDB + cross-encoder reranking)
4. **Generates** a structured diagnosis and step-by-step remediation plan via a local LLM (Mistral-Nemo, served by Ollama)

The LLM stage is triggered only on flagged events — not on every log entry — making it practical for real-time operation with local hardware.

---

## Repository Structure

```
.
├── model_utils.py              # EnsembleCalibratedClassifier and related classes
├── preprocess_raw_data.py      # Combine CSVs → processed_power_data.pkl
├── train_classifier_v2.py      # Train the active binary ensemble model
├── train_cascaded_classifier.py # Train the optional per-scenario cascade classifier
├── build_rag_index.py          # Build the ChromaDB RAG index from manuals/
├── evaluate_system.py          # Run the full 200-sample LLM benchmark
├── grid_monitor.py             # Main pipeline (real-time event loop)
├── run_once.py                 # Run the pipeline on a single event (demo)
│
├── power_model_lgb.pkl         # Active trained model (29-feature ensemble, τ=0.560)
├── label_encoder.pkl           # LabelEncoder for Attack/Natural classes
│
├── manuals/
│   └── grid_protocols.txt      # Remediation protocol reference manual
├── chroma_db/                  # Pre-built ChromaDB vector index (ready to use)
│
├── results/
│   ├── Full200Run.txt              # Run A benchmark (original architecture, 84.5%)
│   ├── Full200Run_AfterUpdates.txt # Run B / AYLA benchmark (90.5%)
│   ├── benchmark_results.json      # Structured benchmark output
│   ├── benchmark_results200.json
│   ├── training_metrics_v2.json    # Active model training metrics
│   ├── training_metrics_v3.json    # v3 experiment (128-feat, no SHAP pruning)
│   ├── training_metrics_v4.json    # v4 experiment (Optuna — rejected)
│   ├── cascade_metrics.json        # Per-scenario cascade classifier metrics
│   └── roc_curve.png
│
└── paper/
    ├── power_shield_paper.tex
    ├── power_shield_paper.pdf
    └── references.bib
```

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) with `mistral-nemo:latest` pulled (`ollama pull mistral-nemo`)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Setup

The dataset and large generated files are not included in this repository. Follow these steps to reproduce the full environment from scratch, or skip to **Quick Start** if you only want to run the pre-trained pipeline.

### 1. Dataset

Download the MSU/ORNL Power System Attack Dataset and place the 15 CSV files in:

```
binaryAllNaturalPlusNormalVsAttacks/data1.csv  …  data15.csv
```

Dataset source: Mississippi State University / Oak Ridge National Laboratory  
(Publicly available from the [ICS Security datasets page](https://www.ece.msstate.edu/research/laboratory/cyber-physical-systems-security-lab/))

### 2. Preprocess

```bash
python3 preprocess_raw_data.py
```

Outputs: `processed_power_data.pkl`

### 3. Train the classifier (optional — pre-trained model included)

```bash
python3 train_classifier_v2.py
```

Outputs: `power_model_lgb.pkl`, `label_encoder.pkl`, `results/training_metrics_v2.json`

### 4. Build the RAG index (optional — pre-built index included)

```bash
python3 build_rag_index.py
```

Outputs: `chroma_db/`

---

## Quick Start

If you have Ollama running and the pre-trained model files (`power_model_lgb.pkl`, `label_encoder.pkl`, `chroma_db/`, `processed_power_data.pkl`):

**Run the pipeline on a single event:**

```bash
python3 run_once.py                  # random Attack sample
python3 run_once.py --index 12345    # specific dataset row
python3 run_once.py --natural        # random sample (may be Natural)
```

**Run the real-time event monitor:**

```bash
python3 grid_monitor.py
```

**Run the full 200-sample LLM benchmark:**

```bash
python3 evaluate_system.py
```

---

## Results

| Metric | Value |
|--------|-------|
| Accuracy | 91.88% |
| Attack Recall | 94.6% |
| AUC-ROC | 0.9665 |
| MCC | 0.8020 |
| LLM Protocol Attribution (200-sample) | **90.5%** (181/200) |
| RAG retrieval (correct protocol in context) | 100% (13/13 protocols) |

Full benchmark details and statistical analysis are in the paper.

---

## Hardware Notes

The pipeline was benchmarked on a laptop with an NVIDIA RTX 3060 (6GB VRAM), averaging **3.5 minutes per event** end-to-end. Because the LLM stage is triggered only on flagged events (not on every relay log), throughput scales with event rate rather than raw data volume. For production deployments, an RTX 4090 reduces latency to ~45–90 seconds per event; an A100 to ~20–40 seconds.

---

## Citation

If you use this work, please cite the accompanying paper:

```
@article{adams2026powershield,
  title   = {Power Shield: An Automated Yield-Loss Alerting (AYLA) Pipeline Using
             Explainable AI (XAI) and Machine Learning (ML) for Power Grid
             Attack Detection and LLM-Guided Remediation},
  author  = {Adams, Gabriel},
  year    = {2026}
}
```
