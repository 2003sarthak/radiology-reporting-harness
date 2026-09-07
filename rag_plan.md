# Radiology Reporting Harness — Stage 2 RAG Implementation Plan

## Goal

Target top-tier leaderboard performance (**RES < 0.250**) by upgrading from Stage 1 heuristic baseline to an API-assisted, CPU-lightweight **Retrieval-Augmented Generation (RAG)** pipeline.

---

## 1. Core RAG Architecture Overview

```text
                                 Input Test Case
                                       │
                                       ▼
                         Metadata & Template Signatures
                                       │
                                       ▼
                       Stage 2 Hybrid RAG Retriever
                             (Code/rag_retriever.py)
                   ├── Exact Template Signature Grouping
                   ├── Metadata Filter (Modality + Body Part)
                   └── BM25 + TF-IDF Lexical Similarity Search
                                       │
                                       ▼
                       Top-K (K=3) Relevant Training Examples
                         (Historical Dictation → Report Deltas)
                                       │
                                       ▼
                       Dynamic Few-Shot Prompt Builder
                             (Code/rag_prompt_builder.py)
                                       │
                                       ▼
                            Gemini API / LLM Call
                           (Structured JSON Output)
                                       │
                                       ▼
                       Deterministic Template Editing Engine
                             (Code/template_editor.py)
                                       │
                                       ▼
                        Clinical & Negation Validator
                             (Code/validator.py)
                                       │
                                       ▼
                       Target Leaderboard Output (< 0.25 RES)
```

---

## 2. Technical Component Design

### Component 1: `Code/rag_retriever.py` (Hybrid CPU Retrieval Engine)
- **Template Signature Clustering**: Extract ordered field header signatures (e.g. `["BONES", "JOINTS", "SOFT TISSUES"]`) for all 636 training cases.
- **Lexical Indexing (BM25 & TF-IDF)**: Build a zero-cost, CPU-only index over training dictations using `rank_bm25` and `scikit-learn`.
- **Hierarchical Ranker**:
  1. *Hard Filter*: Match exact or high-Jaccard `template_signature` + `modality` + `body_part`.
  2. *Lexical Scoring*: Score BM25 similarity between input dictation and reference dictations.
  3. *Select Top-K*: Pick Top-3 highest scoring reference examples per test case.

### Component 2: `Code/rag_prompt_builder.py` (Dynamic Few-Shot Context)
- Takes input case + Top-3 retrieved training cases.
- Formats dynamic few-shot dynamic prompt showing:
  - Input template & dictation.
  - Ground truth reference field updates and impression.
  - Target case to edit.
- Instructs the LLM to output structured JSON:
  ```json
  {
    "field_updates": {
      "LUNGS": "Mild right basilar airspace opacity."
    },
    "impression": "Mild right basilar airspace opacity."
  }
  ```

### Component 3: `Code/rag_llm.py` (LLM Integration with API Key & Fallback)
- Connects to Gemini API (`gemini-2.5-flash`) using user `GEMINI_API_KEY`.
- Fallback: Uses Stage 1 heuristic engine if offline or no key provided.

### Component 4: Integration & Evaluation (`Code/pipeline.py`)
- Run 5-fold cross-validation across 636 `train.csv` cases.
- Measure RES score reduction from `0.6173` (Stage 1) down to target `< 0.250`.
- Generate updated `submission.csv` for the 132 `test.csv` cases.

---

## 3. Step-by-Step Implementation Roadmap

```text
Step 1: Install rank_bm25 dependency (py -3.13 -m pip install rank_bm25)
Step 2: Implement Code/rag_retriever.py (BM25 + Template Signature Search)
Step 3: Implement Code/rag_prompt_builder.py (Dynamic Few-Shot Dynamic Prompting)
Step 4: Implement Code/rag_llm.py (Gemini API Integration)
Step 5: Execute 5-Fold Cross-Validation & Benchmark RES Score
Step 6: Generate v2 submission.csv & Kaggle Pipeline Notebook
```

---

## 4. Resource Allocation & Verification

- **Compute**: 100% CPU (No GPU required, runs in ~2 seconds for retrieval).
- **API**: Gemini API key (`GEMINI_API_KEY`).
- **Target Metric**: RES < 0.250 (Leaderboard Top 10 level).
