# Palantir Trofinetide (Daybue) — Longitudinal Phenotype Monitoring Pipeline

## Overview

This project implements an **LLM-powered longitudinal phenotype monitoring pipeline** for patients on **trofinetide (Daybue)** — a novel treatment for Rett syndrome — built on **Palantir Foundry**. Starting from a curated cohort of 115 patients identified by clinical nurses, the pipeline extracts medication-relevant clinical notes, identifies diarrhea phenotypes (a primary side effect) over time using a multi-stage LLM approach, and constructs a patient × time-point matrix to track symptom trajectories.

This is a first-of-its-kind analysis for trofinetide, a medication costing approximately $500K per patient. The pipeline currently achieves **65% accuracy** compared to nurse annotations, with ongoing work to improve extraction quality and expand to additional phenotypes (other side effects and improvements).

## Pipeline Architecture

### Stage 1 — Note Extraction & Filtering

**`transforms/1_preprocessing.py`**
Extracts clinical notes for the patient cohort, filtering for notes that mention Daybue/trofinetide from September 2022 onward. Includes note chunking for large documents (>200K characters) to stay within LLM context limits.

### Stage 2 — LLM Phenotype Extraction (4-step + 1 prior events)

The LLM pipeline (`prompts/phenotype_extraction_prompts.txt`) uses Claude Opus 4.6 across multiple stages:

1. **Summarizer** — Extracts only Daybue-related passages from each clinical note
2. **Extractor** — Identifies diarrhea phenotype with severity levels (1–4) and temporal information
3. **Validator** — Verifies extracted findings against the original note, correcting evidence quotes, severity levels, and dates
4. **Enrollment Checker** — Determines medication start/stop status per note to establish time zero

**+1 Prior Events Prompt** — Extracts chronic/recurrent diarrhea mentions from pre-enrollment notes (1 year lookback) to establish baseline

### Stage 3 — Postprocessing & Matrix Construction

| Step | File | Description |
|------|------|-------------|
| 2 | `2_postprocessing.py` | Parses LLM outputs (enrollment JSON, extractor JSON, validator JSON), merges with validator-overrides-extractor logic, buckets into 6-month intervals, and pivots to patient × time-point matrix |
| 3 | `3_prior_events.py` | Collects all clinical notes from 1 year before enrollment for baseline phenotype extraction |
| 4 | `4_complete_table.py` | Adds pre-enrollment diarrhea columns (-6 months, -12 months) from prior event extraction to the main matrix |
| 5 | `5_meta_variables.py` | Enriches the matrix with EPIC medication safety checks (administered meds, order meds, refill coverage) and provider information, producing parallel rows per patient for diarrhea, medication coverage, and refill status |
| 6 | `6_time_adjustment.py` | Corrects enrollment dates for patients with pre-enrollment refill evidence, shifting time-point windows accordingly |

## Diarrhea Severity Scale

| Level | Description |
|-------|-------------|
| 1 | Loose stool in diaper |
| 2 | Watery stool in diaper |
| 3 | Watery stool on clothes |
| 4 | Watery stool outside clothes |

## Output Structure

The final output is a patient × time-point matrix spanning -12 months to +30 months (in 6-month buckets) with three parallel rows per patient:

- **diarrhea** — Yes/No (with severity level) per time window
- **epic_med** — Yes/No whether EPIC medication records confirm Daybue coverage
- **refill** — Yes/No whether EPIC refill records exist for that window

Each includes provider attribution and a SUMMARY row with aggregate counts.

## Tech Stack

- **Platform:** Palantir Foundry
- **Language:** Python (PySpark)
- **Framework:** Foundry Transforms (`transforms.api`)
- **LLM:** Claude Opus 4.6 (via Palantir AIP)

## Project Structure

```
Palantir_Trofinetide/
├── transforms/
│   ├── 1_preprocessing.py           # Note extraction & medication mention filtering
│   ├── 2_postprocessing.py          # LLM output parsing & time-point matrix construction
│   ├── 3_prior_events.py            # Pre-enrollment note collection (1-year lookback)
│   ├── 4_complete_table.py          # Pre-enrollment diarrhea column integration
│   ├── 5_meta_variables.py          # EPIC medication safety check & provider enrichment
│   └── 6_time_adjustment.py         # Enrollment date correction based on refill evidence
└── prompts/
    └── phenotype_extraction_prompts.txt  # 4-step LLM pipeline + prior events prompt
```

## Current Performance & Future Work

- **Current accuracy:** 65% agreement with nurse annotations
- **Planned improvements:** Prompt refinement, additional validation stages, expanded phenotype coverage (beyond diarrhea to other side effects and clinical improvements)
- **Future phenotypes:** Appetite changes, sleep quality, seizure frequency, behavioral improvements, GI symptoms

## Usage

All transforms run within a **Palantir Foundry** environment as a sequential pipeline. The LLM extraction steps are handled upstream via Foundry's AIP integration.

> **Note:** Dataset RIDs and provider names in the source code are placeholders. Replace them with your own Foundry dataset RIDs before deploying.

## Data Privacy

This repository contains **code and prompts only** — no clinical notes, patient data, or PHI are included. All data access and LLM inference occurs within Foundry's governed environment.