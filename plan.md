# Radiology Reporting Harness — Phased Execution Plan (Prompt Baseline → RAG Fallback)

## Goal

Build a reproducible GenAI & Deterministic NLP system for the Natoe AI Dev radiology reporting challenge.

**Input per case:**
- `case_id`
- `modality`
- `body_part`
- `study_description`
- `patient_age_band`
- `patient_sex`
- `template_content`
- `dictation`

**Output per case:**
```text
FINDINGS:
<FINDINGS fields>

IMPRESSION:
<concise summary>
```

**Final submission:** CSV file named `submission.csv` containing exactly `case_id,report`.

---

## Phased Strategy Architecture

```text
                          ┌───────────────────────────┐
                          │   Input Test/Train Case   │
                          └─────────────┬─────────────┘
                                        │
                                        ▼
                      ┌───────────────────────────────────┐
                      │  STAGE 1: Prompt-Only Baseline    │
                      │  - Direct LLM Prompting           │
                      │  - Structured Edit Extraction     │
                      │  - Deterministic Template Editor  │
                      └─────────────────┬─────────────────┘
                                        │
                                        ▼
                      ┌───────────────────────────────────┐
                      │    Evaluate RES Score on CV       │
                      └─────────────────┬─────────────────┘
                                        │
                      Is RES Score Good Enough?
                      /                         \
                    YES                          NO
                    /                             \
                   ▼                               ▼
       ┌──────────────────────┐        ┌──────────────────────────────┐
       │ Proceed to Submission│        │ STAGE 2: Advanced RAG System │
       │ & Kaggle Notebook    │        │ - Train Case Retrieval       │
       └──────────────────────┘        │ - Dynamic Few-Shot Context   │
                                       │ - Hybrid BM25/Dense Search   │
                                       └──────────────┬───────────────┘
                                                      │
                                                      ▼
                                       ┌──────────────────────────────┐
                                       │ Re-evaluate RES Score        │
                                       └──────────────────────────────┘
```

---

## 1. STAGE 1 — Primary Prompt-Only LLM Pipeline (Simple & Fast)

1. **Direct Prompt Engineering**:
   - Provide `modality`, `body_part`, `study_description`, `patient_age_band`, `patient_sex`, `template_content`, and `dictation`.
   - Instruct the LLM to output field-level edit deltas in JSON format (`field_updates` and `impression`) or direct minimally-edited report.
   - Enforce explicit rules: preserve untouched normal statements, route dictated findings to exact matching anatomical headers, preserve negation, laterality, and numerical measurements.

2. **Deterministic Template Integration**:
   - Parse `template_content` into field keys.
   - Replace only modified fields identified by the LLM.
   - Lock unmodified template fields byte-for-byte to avoid edit penalties.

3. **Local RES Evaluation Benchmark**:
   - Run 5-Fold Cross-Validation on the 636 `train.csv` cases using local RES metric scorer (`evaluator.py`).
   - Benchmark prompt-only performance.
   - **Decision Gate**: If local RES score meets target performance, adopt Stage 1 for final test prediction. If RES score is inadequate or has high edit error rate, trigger Stage 2 (RAG).

---

## 2. STAGE 2 — Secondary Fallback: Advanced RAG System (Example-Based Few-Shot)

If Stage 1 prompt-only performance is insufficient, escalate to Stage 2:

1. **Training Case Indexing & Retrieval (`rag_retriever.py`)**:
   - Index the 636 `train.csv` reference cases.
   - Match test cases to training cases using:
     - Exact/High-Jaccard Template Signature Match (e.g. matching headers like `LUNGS:`, `PLEURA:`).
     - Modality + Body Part exact filter.
     - Lexical (BM25) & Dense Semantic Embeddings (`sentence-transformers`) of `dictation + template_content`.

2. **Dynamic In-Context Few-Shot Prompting (`structured_llm.py`)**:
   - Retrieve Top-K (K=3 to 5) most similar training cases.
   - Inject retrieved cases as dynamic dynamic examples showing how dictation findings were edited into matching template fields in past cases.

3. **Re-Evaluation & Ablation**:
   - Benchmark Stage 2 RAG on the local RES evaluator.
   - Compare Stage 1 vs Stage 2 RES scores. Select the winning model pipeline.

---

## Detailed Execution Phases

### PHASE 1 — Directory & Environment Setup
Setup project directory structure: `Code/`, `data/`, `outputs/`, `notebooks/`, `requirements.txt`.

### PHASE 2 — Data Exploration & Edit Analysis
Analyze `train.csv` and `test.csv`:
- Catalog template field headers (`LUNGS:`, `PLEURA:`, `BONES:`, `IMPRESSION:`).
- Analyze train/test template overlap and field edit distribution.
- Catalog clinical keywords (negations, laterality, measurements).

### PHASE 3 — Local RES Metric Evaluator & 5-Fold CV Setup
Implement competition metric RES ($0.65 \times F + 0.35 \times I$) with token weights (4.0 critical, 2.0 content, 0.25 function) and field-aware weighting. Create 5-Fold CV split.

### PHASE 4 — Implement Stage 1 Prompt-Only Baseline
- Build `prompt_llm.py` to generate structured JSON field deltas using direct prompt engineering.
- Apply `template_editor.py` to lock unchanged fields.
- Measure baseline 5-Fold CV RES score.

### PHASE 5 — Stage 1 Decision Gate & Analysis
Evaluate Stage 1 error modes:
- Did LLM misroute findings?
- Did it miss negations or laterality?
- Is the RES score satisfying?
  - **If YES**: Move directly to Phase 8 (Validation) and Phase 9 (Submission & Kaggle Notebook).
  - **If NO**: Proceed to Phase 6 (Stage 2 RAG).

### PHASE 6 — Implement Stage 2 Hybrid RAG Engine (Fallback)
- Build `rag_retriever.py` with template signature matching, BM25, and dense embeddings.
- Retrieve Top-K (K=3..5) dynamic few-shot training examples per case.

### PHASE 7 — Stage 2 Few-Shot Dynamic Prompting & Evaluation
- Prompt LLM with dynamic few-shot examples.
- Re-evaluate RES score on 5-Fold CV. Compare Stage 1 vs Stage 2 results.

### PHASE 8 — Clinical & Structural Validation Engine
Apply guardrails (`validator.py`):
- Negation verification ("no pleural effusion" vs "pleural effusion").
- Laterality verification ("right" vs "left").
- Measurement numeric check (preserve numbers and units).
- Field header structural lock.

### PHASE 9 — Test Prediction Generation (`submission.csv`)
- Run winning pipeline (Stage 1 or Stage 2) over 132 test cases in `test.csv`.
- Create `submission.csv` with exactly `case_id,report`.

### PHASE 10 — Kaggle Notebook Packaging & Submission
- Create self-contained Kaggle notebook `kaggle_pipeline.ipynb`.
- Share private notebook with `natoeaidev`.
- Include notebook URL in final submission description.

---

## Development Order

```text
1. Directory & Environment Setup (Phase 1)
2. Data Exploration & Edit Analysis (Phase 2)
3. Local RES Evaluator & 5-Fold CV Harness (Phase 3)
4. Implement Stage 1 Prompt-Only Baseline (Phase 4)
5. Evaluate Stage 1 Baseline & Decision Gate (Phase 5)
6. Implement Stage 2 RAG Engine if Stage 1 is insufficient (Phases 6 & 7)
7. Clinical & Structural Validation Engine (Phase 8)
8. Test Prediction Generation (Phase 9)
9. Kaggle Notebook Packaging & Submission (Phase 10)
```

## Current Status

> **PHASE 2 — Data Exploration & Edit Analysis**

## Central Engineering Principle

> **Try simple direct prompting first. Use deterministic code to preserve unchanged template statements. Escalate to dynamic RAG only if prompt-only RES evaluation warrants it.**
