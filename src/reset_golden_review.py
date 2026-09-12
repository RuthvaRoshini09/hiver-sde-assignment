"""
src/reset_golden_review.py
--------------------------
Resets human-review fields in data/processed/golden_set_200.csv to prepare for
a clean manual re-labeling process.

Preserves:
- tweet_id, clean_message, provisional_intent, expected_action, escalation_expected (100% unchanged)

Resets:
- reviewer_label -> ""
- reviewer_action -> ""
- reviewer_escalation -> ""
- reviewer_notes -> ""
- review_status -> "PROVISIONAL - NOT HUMAN LABELLED"

Backup:
- Saves data/processed/golden_set_200_before_relabel.csv before modifying.
"""

import os
import sys
import shutil
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

GOLDEN_SET_PATH = os.path.join("data", "processed", "golden_set_200.csv")
BACKUP_PATH = os.path.join("data", "processed", "golden_set_200_before_relabel.csv")


def reset_golden_review():
    print("=" * 80)
    print("RESETTING GOLDEN SET FOR HUMAN RE-LABELING")
    print("=" * 80)

    if not os.path.exists(GOLDEN_SET_PATH):
        print(f"Error: {GOLDEN_SET_PATH} not found!")
        sys.exit(1)

    # 1. Create backup
    print(f"Creating backup copy at: {BACKUP_PATH}...")
    shutil.copyfile(GOLDEN_SET_PATH, BACKUP_PATH)
    print("Backup copy created successfully.")

    # 2. Load dataset
    df = pd.read_csv(GOLDEN_SET_PATH)
    total_rows = len(df)
    assert total_rows == 200, f"Expected 200 rows, found {total_rows}!"

    # 3. Reset human-review fields
    df["reviewer_label"] = ""
    df["reviewer_action"] = ""
    df["reviewer_escalation"] = ""
    df["reviewer_notes"] = ""
    df["review_status"] = "PROVISIONAL - NOT HUMAN LABELLED"

    # 4. Save clean file
    df.to_csv(GOLDEN_SET_PATH, index=False)
    print(f"Clean review file saved to: {GOLDEN_SET_PATH}")

    print("\nVerification of reset fields:")
    print(f"  Total rows:               {len(df)}")
    print(f"  Empty reviewer_label:     {(df['reviewer_label'] == '').sum()} / {len(df)}")
    print(f"  Empty reviewer_action:    {(df['reviewer_action'] == '').sum()} / {len(df)}")
    print(f"  Empty reviewer_escalate:  {(df['reviewer_escalation'] == '').sum()} / {len(df)}")
    print(f"  Empty reviewer_notes:     {(df['reviewer_notes'] == '').sum()} / {len(df)}")
    print(f"  review_status:            {df['review_status'].iloc[0]}")
    print("=" * 80)
    print("RESET COMPLETE - READY FOR MANUAL LABELING")
    print("=" * 80)


if __name__ == "__main__":
    reset_golden_review()
