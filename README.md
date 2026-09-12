# AppleSupport Automated Agent

An automated customer support triaging and reply drafting agent for `@AppleSupport` on Twitter/X, developed for the Hiver SDE Intern take-home assignment.

---

## 1. Project Overview
Customer support on public social media involves high volumes of short, informal, and emotionally charged inquiries. This system provides an end-to-end automated workflow to:
1. **Classify Intent:** Map incoming customer inquiries to an established 11-intent domain taxonomy.
2. **Ground Responses in Evidence:** Retrieve verified historical Apple Support resolutions via TF-IDF cosine similarity to eliminate hallucinations.
3. **Automate Escalation Triaging:** Distinguish between self-serve inquiries (`AUTO_HANDLE`) and sensitive issues requiring human tier-2 routing (`ESCALATE`).

---

## 2. System Architecture

```
[Inbound Tweet]
       │
       ▼
[1. Text Preprocessing & Sanitization]
       │
       ├──► [2. Intent Classifier] (TF-IDF + Logistic Regression / Rules)
       │         │
       │         ▼
       ├──► [3. RAG Retrieval Engine] (Historical AppleSupport Pair Index)
       │         │
       ▼         ▼
[4. Triaging & Decision Policy]
       │
       ├─────────────────────────┬─────────────────────────┐
       ▼                         ▼                         ▼
 [Low Confidence]       [Security / Hardware]       [High Confidence & Evidence]
       │                         │                         │
       └───────────┬─────────────┘                         │
                   ▼                                       ▼
            [ESCALATE] (34.5%)                     [AUTO_HANDLE] (65.5%)
                   │                                       │
                   ▼                                       ▼
        [Escalation Advisory]                  [Evidence-Grounded Reply]
```

### Core Components
- **`src/support_agent.py`**: End-to-end agent combining classification, retrieval, triaging policy, and reply drafting.
- **`src/intent_classifier.py`**: Machine learning intent classification pipeline.
- **`src/baseline2_rule_based.py`**: Deterministic keyword and regex matching baseline.
- **`src/evaluate_agent.py`**: Automated evaluation harness over the 200-example golden set.
- **`src/llm_judge.py`**: LLM-as-Judge evaluation harness (supports `--limit`, transient retries, and clean skipping).

---

## 3. Setup Instructions

### Environment
- Python 3.10+ (tested on Python 3.12/3.14 on Windows)
- Required packages:
  ```bash
  pip install pandas scikit-learn joblib requests numpy
  ```

### Repository Structure
```
hiver-sde-assignment/
├── data/
│   └── processed/
│       ├── golden_set_final.csv         # 200-row human-audited golden evaluation benchmark
│       ├── applesupport_pairs.csv       # Historical customer-support dialogue pairs
│       ├── retrieval_index.joblib       # Precomputed TF-IDF retrieval index
│       ├── baseline1_tfidf_logreg.joblib# Baseline 1 trained model
│       ├── evaluation_results.csv       # 200-row agent predictions & retrieval outputs
│       ├── evaluation_summary.json      # Structured evaluation metrics
│       ├── baseline2_results.csv        # Baseline 2 predictions
│       ├── baseline2_summary.json       # Baseline 2 metrics
│       ├── llm_judge_results.csv        # LLM judge output (empty if skipped)
│       └── llm_judge_summary.json       # LLM judge summary
├── docs/
│   └── evaluation_report.md             # Comprehensive final evaluation report (17 sections)
├── src/
│   ├── support_agent.py                 # Main agent implementation
│   ├── evaluate_agent.py                # Evaluation runner for agent
│   ├── baseline2_rule_based.py          # Deterministic baseline runner
│   ├── llm_judge.py                     # LLM-as-Judge evaluation harness
│   └── intent_classifier.py             # Classifier definitions
└── README.md
```

---

## 4. Execution Commands

### 1. Run the Support Agent
To test the agent interactively or process sample queries:
```bash
python src/support_agent.py
```

### 2. Run Main Evaluation
To evaluate the agent pipeline against the 200-example golden set (`data/processed/golden_set_final.csv`):
```bash
python src/evaluate_agent.py
```

### 3. Run Baseline 2 (Rule-Based Classifier)
To execute and evaluate the deterministic keyword baseline:
```bash
python src/baseline2_rule_based.py
```

### 4. Run LLM-as-Judge
- **Single-example test mode:**
  ```bash
  python src/llm_judge.py --limit 1
  ```
- **Full evaluation (requires valid `OPENAI_API_KEY`):**
  ```bash
  python src/llm_judge.py
  ```
  *Note: If `OPENAI_API_KEY` is not set or authentication fails (HTTP 401), the harness cleanly skips evaluation and records `null` / `N/A` rather than fabricating synthetic scores.*

---

## 5. Evaluation Results Summary

### Intent Classification Benchmark (Golden Set N = 200)
| Metric | Baseline 1 (TF-IDF + LogReg) | Baseline 2 (Deterministic Rules) | Delta (Baseline 2 - Baseline 1) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 16.50% | **17.00%** | +0.50 pp |
| **Macro Precision** | 17.56% | **17.64%** | +0.08 pp |
| **Macro Recall** | 18.42% | **19.19%** | +0.77 pp |
| **Macro F1** | 17.16% | **17.56%** | +0.40 pp |
| **Weighted F1** | 16.16% | **16.29%** | +0.13 pp |

### Triaging & Operational Metrics
- **Total Evaluated:** 200 / 200 (100% pipeline stability, 0 execution errors)
- **AUTO_HANDLE Decisions:** 131 (65.5%)
- **ESCALATE Decisions:** 69 (34.5%)
- **Escalation Evaluation:** `NOT_EVALUABLE` (ground truth contains 144 False, 56 unlabeled, 0 True)
- **Retrieval Coverage:** 100% (200/200 queries matched historical pairs)
- **Reply Coverage:** 100%
- **Retrieval-Backed Reply Rate:** 100%
- **Mean Top-1 Cosine Similarity:** 0.3977

### Human Review Evidence
- Second-pass human audit of 56 ambiguous examples accepted 52/56 proposed labels (92.9%) and overrode 4/56 (7.1%).
- *Terminology note:* This represents a directed second-pass audit of proposed labels, not independent double-blind inter-annotator agreement (Cohen's Kappa).

### LLM-as-Judge Status
- **Status:** `NOT RUN — API authentication unavailable` (HTTP 401 error during key validation).
- Metrics reported as `N/A` with zero fabricated scores.

---

## 6. Key Limitations
1. **Benchmark Scale:** The golden benchmark contains 200 hand-verified examples; each classification shifts accuracy by 0.50 pp.
2. **Class Imbalance:** Class support ranges from 36 (`billing_subscription_refund`) to 7 (`keyboard_text_glitch`).
3. **Escalation Ground Truth:** Lack of positive escalation ground truth renders statistical escalation metrics incalculable.
4. **Retrieval Evaluation:** Retrieval metrics measure lexical overlap (cosine similarity, Jaccard overlap) rather than annotated relevance.
5. **Verbatim Reply Drafting:** Replies reuse historical support text, which can occasionally risk device-version discrepancies.

For the full 17-section analysis, root causes of failure cases, decision log, and next-week roadmap, see [`docs/evaluation_report.md`](docs/evaluation_report.md).
