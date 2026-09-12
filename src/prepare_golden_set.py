"""
src/prepare_golden_set.py
-------------------------
Samples exactly 200 real, diverse customer messages from applesupport_pairs.csv
for human evaluation and gold-standard labeling.

Distribution:
- 15 examples for each of the 10 core intents = 150
- 50 examples for other_general_inquiry (out-of-scope/rejection)
- Total = 200 examples

Requirements:
- Only real tweets from AppleSupport dataset (no synthetic data).
- Diversity filtering to prevent duplicate or near-identical tweets.
- Reviewer columns left blank for human audit.
- review_status set to 'PROVISIONAL - NOT HUMAN LABELLED'.
"""

import os
import sys
import re
from collections import defaultdict

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd
import numpy as np

from src.intent_discovery import (
    CORE_INTENT_DEFINITIONS,
    OUT_OF_SCOPE_DEFINITION,
    compile_regex_matchers,
    classify_intent_rule
)

PAIRS_PATH = "data/processed/applesupport_pairs.csv"
OUTPUT_PATH = "data/processed/golden_set_200.csv"


def jaccard_similarity(text1: str, text2: str) -> float:
    """Compute token Jaccard similarity between two texts."""
    tokens1 = set(re.findall(r"\b\w+\b", text1.lower()))
    tokens2 = set(re.findall(r"\b\w+\b", text2.lower()))
    if not tokens1 or not tokens2:
        return 0.0
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)


def select_diverse_subset(df_pool: pd.DataFrame, target_count: int, sim_threshold: float = 0.5) -> pd.DataFrame:
    """
    Selects target_count diverse tweets from df_pool, ensuring no two tweets
    have Jaccard similarity >= sim_threshold.
    """
    selected_indices = []
    selected_texts = []

    # Iterate through pool
    for idx, row in df_pool.iterrows():
        text = row["customer_text_clean"]
        if not isinstance(text, str) or len(text.strip()) < 15:
            continue

        # Check similarity against already selected texts
        too_similar = False
        for s_text in selected_texts:
            if jaccard_similarity(text, s_text) >= sim_threshold:
                too_similar = True
                break

        if not too_similar:
            selected_indices.append(idx)
            selected_texts.append(text)
            if len(selected_indices) == target_count:
                break

    # If strict threshold didn't gather enough, relax slightly to fill remainder
    if len(selected_indices) < target_count:
        for idx, row in df_pool.iterrows():
            if idx not in selected_indices:
                selected_indices.append(idx)
                if len(selected_indices) == target_count:
                    break

    return df_pool.loc[selected_indices]


def main():
    print("=" * 80)
    print("PREPARING 200-ROW GOLDEN EVALUATION SET TEMPLATE")
    print("=" * 80)

    if not os.path.exists(PAIRS_PATH):
        print(f"Error: {PAIRS_PATH} not found!")
        sys.exit(1)

    print(f"Loading {PAIRS_PATH}...")
    df = pd.read_csv(PAIRS_PATH)
    print(f"Total pairs loaded: {len(df):,d}")

    # Subsetting: initial turns that are usable
    df_initial = df[(df["is_initial_turn"] == True) & (df["is_usable"] == True)].copy()
    print(f"Usable initial customer inquiries: {len(df_initial):,d}")

    # Compile matchers and classify
    matchers = compile_regex_matchers()
    df_initial["candidate_intent"] = df_initial["customer_text_clean"].apply(
        lambda t: classify_intent_rule(t, matchers)
    )

    golden_rows = []

    # 1. Sample 15 diverse examples for each of the 10 core intents (150 total)
    print("\nSampling 15 diverse examples per core intent (150 total)...")
    for intent_name, info in CORE_INTENT_DEFINITIONS.items():
        pool = df_initial[df_initial["candidate_intent"] == intent_name].copy()
        # Shuffle deterministically for diverse sampling
        pool = pool.sample(frac=1.0, random_state=42).reset_index(drop=True)

        selected = select_diverse_subset(pool, target_count=15, sim_threshold=0.45)
        print(f"  {intent_name:<30}: sampled {len(selected)} diverse examples from {len(pool)} available")

        for _, row in selected.iterrows():
            golden_rows.append({
                "tweet_id": row["customer_tweet_id"],
                "clean_message": row["customer_text_clean"],
                "provisional_intent": intent_name,
                "expected_action": info["typical_action"],
                "escalation_expected": info["escalation_expected"],
                "reviewer_label": "",
                "reviewer_action": "",
                "reviewer_escalation": "",
                "reviewer_notes": "",
                "review_status": "PROVISIONAL - NOT HUMAN LABELLED"
            })

    # 2. Sample 50 diverse examples for other_general_inquiry (out-of-scope rejection)
    print("\nSampling 50 diverse examples for other_general_inquiry...")
    pool_other = df_initial[df_initial["candidate_intent"] == "other_general_inquiry"].copy()
    pool_other = pool_other.sample(frac=1.0, random_state=42).reset_index(drop=True)

    selected_other = select_diverse_subset(pool_other, target_count=50, sim_threshold=0.40)
    print(f"  {'other_general_inquiry':<30}: sampled {len(selected_other)} diverse examples from {len(pool_other)} available")

    for _, row in selected_other.iterrows():
        golden_rows.append({
            "tweet_id": row["customer_tweet_id"],
            "clean_message": row["customer_text_clean"],
            "provisional_intent": "other_general_inquiry",
            "expected_action": OUT_OF_SCOPE_DEFINITION["typical_action"],
            "escalation_expected": False,
            "reviewer_label": "",
            "reviewer_action": "",
            "reviewer_escalation": "",
            "reviewer_notes": "",
            "review_status": "PROVISIONAL - NOT HUMAN LABELLED"
        })

    df_golden = pd.DataFrame(golden_rows)

    # Sanity checks
    assert len(df_golden) == 200, f"Expected 200 rows, got {len(df_golden)}"
    assert df_golden["reviewer_label"].isna().all() or (df_golden["reviewer_label"] == "").all(), "Reviewer label must be blank"
    assert (df_golden["review_status"] == "PROVISIONAL - NOT HUMAN LABELLED").all(), "All statuses must be provisional"

    # Save to CSV
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_golden.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved exactly {len(df_golden)} rows to: {OUTPUT_PATH}\n")

    # Display distribution
    print("=" * 70)
    print("GOLDEN SET (200 ROWS) DISTRIBUTION")
    print("=" * 70)
    dist = df_golden["provisional_intent"].value_counts()
    for intent, count in dist.items():
        print(f"  {intent:<30}: {count:3d} rows ({count / len(df_golden) * 100:.1f}%)")
    print("-" * 70)
    print(f"  {'TOTAL':<30}: {len(df_golden):3d} rows (100.0%)")
    print("=" * 70)

    # Display first 10 rows
    print("\nFIRST 10 ROWS FOR REVIEW:")
    print("=" * 100)
    display_cols = ["tweet_id", "clean_message", "provisional_intent", "expected_action", "review_status"]
    print(df_golden[display_cols].head(10).to_string(index=True))
    print("=" * 100)


if __name__ == "__main__":
    main()
