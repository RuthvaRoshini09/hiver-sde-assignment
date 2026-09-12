"""
src/baseline2_rule_based.py
----------------------------
Baseline #2: Deterministic Rule-Based Keyword Intent Classifier for AppleSupport.

Evaluates the rule-based heuristic intent classification against the 200-row
finalized golden evaluation set (data/processed/golden_set_final.csv).

Reuses the project's 10 core intent definitions + explicit out-of-scope intent
defined in src/intent_discovery.py.

Capabilities:
1. Loads finalized 200-row golden set (clean_message as input, final_intent as ground truth).
2. Performs deterministic rule-based keyword / regex matching in prioritized order.
3. Maps all unmatched inquiries to 'other_general_inquiry'.
4. Computes Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1,
   per-class metrics, and confusion matrix.
5. Saves results to data/processed/baseline2_results.csv.
6. Saves summary to data/processed/baseline2_summary.json.
7. Evaluates and compares against Baseline #1 (TF-IDF + Logistic Regression).

Usage:
    python src/baseline2_rule_based.py
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Setup project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safe UTF-8 console output for Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

# Re-use existing taxonomy definitions and rules from intent_discovery
from src.intent_discovery import (
    CORE_INTENT_DEFINITIONS,
    compile_regex_matchers,
    classify_intent_rule
)

# File paths
GOLDEN_SET_PATH = PROJECT_ROOT / "data" / "processed" / "golden_set_final.csv"
BASELINE1_MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "baseline1_tfidf_logreg.joblib"
OUTPUT_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "baseline2_results.csv"
OUTPUT_JSON_PATH = PROJECT_ROOT / "data" / "processed" / "baseline2_summary.json"

# Approved 11-intent label space
LABEL_SPACE = sorted(list(CORE_INTENT_DEFINITIONS.keys()) + ["other_general_inquiry"])

# Exact priority hierarchy across core intents (from src/intent_discovery.py)
PRIORITY_ORDER = [
    "billing_subscription_refund",
    "keyboard_text_glitch",
    "apple_id_account_access",
    "hardware_buttons_audio",
    "bluetooth_airpods_carplay",
    "network_wifi_cellular",
    "app_crash_download_error",
    "screen_touch_freeze",
    "battery_drain_charging",
    "os_update_installation"
]


def classify_with_rule_tracking(text: str, matchers: Dict[str, Any]) -> Tuple[str, str]:
    """
    Classifies a customer inquiry using deterministic prioritized regex rules.
    Returns (predicted_intent, matched_rule_description).
    Unmatched inquiries fall back to 'other_general_inquiry'.
    """
    text_lower = str(text).lower()

    for intent in PRIORITY_ORDER:
        matcher = matchers.get(intent)
        if matcher and matcher.search(text_lower):
            match = matcher.search(text_lower)
            matched_term = match.group(0) if match else "regex_match"
            return intent, f"{intent} [term: '{matched_term}']"

    return "other_general_inquiry", "fallback_unmatched"


def generate_markdown_confusion_matrix(cm: np.ndarray, labels: List[str]) -> str:
    """Formats confusion matrix into a clean, readable ASCII / Markdown table."""
    abbrevs = {
        "app_crash_download_error": "AppCrash",
        "apple_id_account_access": "AppleID",
        "battery_drain_charging": "Battery",
        "billing_subscription_refund": "Billing",
        "bluetooth_airpods_carplay": "BT/Pods",
        "hardware_buttons_audio": "Hardware",
        "keyboard_text_glitch": "Keyboard",
        "network_wifi_cellular": "Network",
        "os_update_installation": "OSUpdate",
        "other_general_inquiry": "Other",
        "screen_touch_freeze": "Screen",
    }
    short_headers = [abbrevs.get(l, l[:8]) for l in labels]

    header_line = "| Ground Truth \\ Pred | " + " | ".join(short_headers) + " | Total |"
    sep_line = "| :--- | " + " | ".join([":---:"] * len(labels)) + " | :---: |"

    rows = []
    for i, label in enumerate(labels):
        row_counts = [str(cm[i, j]) for j in range(len(labels))]
        total_row = int(np.sum(cm[i, :]))
        short_row_label = f"**{label}**"
        rows.append(f"| {short_row_label} | " + " | ".join(row_counts) + f" | {total_row} |")

    col_totals = [str(int(np.sum(cm[:, j]))) for j in range(len(labels))]
    total_all = int(np.sum(cm))
    col_total_row = f"| **Total Predicted** | " + " | ".join(col_totals) + f" | **{total_all}** |"

    return "\n".join([header_line, sep_line] + rows + [col_total_row])


def main():
    print("=" * 80)
    print("BASELINE #2: DETERMINISTIC RULE-BASED INTENT CLASSIFIER")
    print("=" * 80)
    print(f"Golden Set:        {GOLDEN_SET_PATH}")
    print(f"Results CSV:       {OUTPUT_CSV_PATH}")
    print(f"Summary JSON:      {OUTPUT_JSON_PATH}")
    print(f"Baseline #1 Model: {BASELINE1_MODEL_PATH}\n")

    # 1. Load Golden Set
    if not GOLDEN_SET_PATH.exists():
        raise FileNotFoundError(f"Golden set file not found: {GOLDEN_SET_PATH}")

    df_golden = pd.read_csv(GOLDEN_SET_PATH)
    total_examples = len(df_golden)
    print(f"Loaded {total_examples} finalized golden evaluation examples.")

    if "final_intent" not in df_golden.columns:
        raise KeyError(f"Expected 'final_intent' column in {GOLDEN_SET_PATH}")

    # 2. Compile matchers
    matchers = compile_regex_matchers()

    # 3. Classify each message with Baseline #2
    results = []
    for idx, row in df_golden.iterrows():
        tweet_id = row["tweet_id"]
        clean_msg = str(row["clean_message"])
        gt_intent = str(row["final_intent"])

        pred_intent, rule_detail = classify_with_rule_tracking(clean_msg, matchers)
        is_correct = bool(pred_intent == gt_intent)

        results.append({
            "tweet_id": tweet_id,
            "clean_message": clean_msg,
            "ground_truth_intent": gt_intent,
            "predicted_intent": pred_intent,
            "is_correct": is_correct,
            "rule_triggered": rule_detail
        })

    df_results = pd.DataFrame(results)

    # 4. Save results to CSV
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Saved Baseline #2 results to: {OUTPUT_CSV_PATH}")

    # 5. Compute Baseline #2 Metrics
    y_true = df_results["ground_truth_intent"].tolist()
    y_pred = df_results["predicted_intent"].tolist()

    all_labels = LABEL_SPACE

    accuracy_b2 = float(accuracy_score(y_true, y_pred))
    p_macro_b2, r_macro_b2, f1_macro_b2, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted_b2, r_weighted_b2, f1_weighted_b2, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    p_per, r_per, f1_per, s_per = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, zero_division=0
    )

    per_class_b2 = {}
    for i, label in enumerate(all_labels):
        per_class_b2[label] = {
            "precision": round(float(p_per[i]), 4),
            "recall": round(float(r_per[i]), 4),
            "f1_score": round(float(f1_per[i]), 4),
            "support": int(s_per[i])
        }

    cm_b2 = confusion_matrix(y_true, y_pred, labels=all_labels)

    # 6. Evaluate Baseline #1 (TF-IDF + LogReg) for direct comparison
    b1_comparison = {"status": "NOT_EVALUATED"}
    if BASELINE1_MODEL_PATH.exists():
        try:
            b1_artifact = joblib.load(BASELINE1_MODEL_PATH)
            vec = b1_artifact["vectorizer"]
            clf = b1_artifact["classifier"]

            messages = df_results["clean_message"].tolist()
            X_tfidf = vec.transform(messages)
            y_b1_pred = clf.predict(X_tfidf).tolist()

            accuracy_b1 = float(accuracy_score(y_true, y_b1_pred))
            p_macro_b1, r_macro_b1, f1_macro_b1, _ = precision_recall_fscore_support(
                y_true, y_b1_pred, average="macro", zero_division=0
            )
            p_weighted_b1, r_weighted_b1, f1_weighted_b1, _ = precision_recall_fscore_support(
                y_true, y_b1_pred, average="weighted", zero_division=0
            )

            p_per_b1, r_per_b1, f1_per_b1, s_per_b1 = precision_recall_fscore_support(
                y_true, y_b1_pred, labels=all_labels, zero_division=0
            )
            per_class_b1 = {}
            for i, label in enumerate(all_labels):
                per_class_b1[label] = {
                    "precision": round(float(p_per_b1[i]), 4),
                    "recall": round(float(r_per_b1[i]), 4),
                    "f1_score": round(float(f1_per_b1[i]), 4),
                    "support": int(s_per_b1[i])
                }

            b1_comparison = {
                "status": "SUCCESS",
                "model_type": "TF-IDF (1-2 ngrams) + Logistic Regression",
                "accuracy": round(accuracy_b1, 4),
                "macro_precision": round(float(p_macro_b1), 4),
                "macro_recall": round(float(r_macro_b1), 4),
                "macro_f1": round(float(f1_macro_b1), 4),
                "weighted_f1": round(float(f1_weighted_b1), 4),
                "per_class_metrics": per_class_b1,
                "delta_accuracy (B2 - B1)": round(accuracy_b2 - accuracy_b1, 4),
                "delta_macro_f1 (B2 - B1)": round(float(f1_macro_b2 - f1_macro_b1), 4),
                "delta_weighted_f1 (B2 - B1)": round(float(f1_weighted_b2 - f1_weighted_b1), 4)
            }
            print("Successfully evaluated Baseline #1 for side-by-side comparison.")
        except Exception as e:
            b1_comparison = {"status": f"ERROR: {str(e)}"}
            print(f"Warning: Could not evaluate Baseline #1: {e}")

    # 7. Save Summary JSON
    summary_data = {
        "model_name": "Baseline 2 (Deterministic Rule-Based Keyword Matching)",
        "golden_set_file": "data/processed/golden_set_final.csv",
        "dataset_size": total_examples,
        "taxonomy": {
            "num_classes": len(all_labels),
            "classes": all_labels,
            "fallback_class": "other_general_inquiry"
        },
        "metrics": {
            "accuracy": round(accuracy_b2, 4),
            "macro_precision": round(float(p_macro_b2), 4),
            "macro_recall": round(float(r_macro_b2), 4),
            "macro_f1": round(float(f1_macro_b2), 4),
            "weighted_f1": round(float(f1_weighted_b2), 4)
        },
        "per_class_metrics": per_class_b2,
        "confusion_matrix": {
            "labels": all_labels,
            "matrix": cm_b2.tolist()
        },
        "baseline1_comparison": b1_comparison
    }

    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved Baseline #2 summary to: {OUTPUT_JSON_PATH}")

    # 8. Print Terminal Summary
    print("\n" + "=" * 80)
    print("BASELINE #2 EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Dataset Size:             {total_examples} examples (golden_set_final.csv)")
    print(f"Classifier Type:          Deterministic Priority Keyword / Regex Matching")
    print(f"Fallback Intent:          other_general_inquiry (unmatched)")
    print("-" * 80)
    print(f"Accuracy:                 {accuracy_b2 * 100:.2f}%")
    print(f"Macro Precision:          {p_macro_b2 * 100:.2f}%")
    print(f"Macro Recall:             {r_macro_b2 * 100:.2f}%")
    print(f"Macro F1:                 {f1_macro_b2 * 100:.2f}%")
    print(f"Weighted F1:              {f1_weighted_b2 * 100:.2f}%")
    print("-" * 80)

    print("\nPER-CLASS METRICS (BASELINE #2):")
    print(f"{'Intent':<32} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>8}")
    print("-" * 74)
    for label in all_labels:
        stats = per_class_b2[label]
        print(f"{label:<32} {stats['precision']*100:>9.1f}% {stats['recall']*100:>9.1f}% {stats['f1_score']*100:>9.1f}% {stats['support']:>8}")

    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE COMPARISON: BASELINE #1 VS BASELINE #2")
    print("=" * 80)
    if b1_comparison.get("status") == "SUCCESS":
        print(f"{'Metric':<25} {'Baseline #1 (TF-IDF LogReg)':>30} {'Baseline #2 (Rule-Based)':>25}")
        print("-" * 80)
        print(f"{'Accuracy':<25} {b1_comparison['accuracy']*100:>29.2f}% {accuracy_b2*100:>24.2f}%")
        print(f"{'Macro Precision':<25} {b1_comparison['macro_precision']*100:>29.2f}% {p_macro_b2*100:>24.2f}%")
        print(f"{'Macro Recall':<25} {b1_comparison['macro_recall']*100:>29.2f}% {r_macro_b2*100:>24.2f}%")
        print(f"{'Macro F1':<25} {b1_comparison['macro_f1']*100:>29.2f}% {f1_macro_b2*100:>24.2f}%")
        print(f"{'Weighted F1':<25} {b1_comparison['weighted_f1']*100:>29.2f}% {f1_weighted_b2*100:>24.2f}%")
        print("-" * 80)
        print(f"Delta Accuracy (B2 - B1):    {b1_comparison['delta_accuracy (B2 - B1)'] * 100:+.2f}%")
        print(f"Delta Macro F1 (B2 - B1):    {b1_comparison['delta_macro_f1 (B2 - B1)'] * 100:+.2f}%")
        print(f"Delta Weighted F1 (B2 - B1): {b1_comparison['delta_weighted_f1 (B2 - B1)'] * 100:+.2f}%")
    else:
        print(f"Baseline #1 comparison status: {b1_comparison.get('status')}")
    print("=" * 80)


if __name__ == '__main__':
    main()
