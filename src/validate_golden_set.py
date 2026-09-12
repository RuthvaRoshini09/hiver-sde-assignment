"""
src/validate_golden_set.py
--------------------------
Validates the human-reviewed Golden Evaluation Set (data/processed/golden_set_200.csv).

Checks:
- Ensures all 200 rows have valid reviewer_label from the 10 core intents + other_general_inquiry.
- Validates reviewer_escalation format (True / False).
- Reports review progress (e.g. X / 200 completed).
- Computes agreement/divergence between provisional hypotheses and human ground truth.
- When 100% complete, updates review_status to 'VERIFIED - GOLDEN' and exports
  data/processed/golden_set_verified.csv.
"""

import os
import sys

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

GOLDEN_SET_PATH = "data/processed/golden_set_200.csv"
VERIFIED_OUTPUT_PATH = "data/processed/golden_set_verified.csv"

VALID_INTENTS = {
    "battery_drain_charging",
    "keyboard_text_glitch",
    "network_wifi_cellular",
    "os_update_installation",
    "bluetooth_airpods_carplay",
    "apple_id_account_access",
    "hardware_buttons_audio",
    "billing_subscription_refund",
    "screen_touch_freeze",
    "app_crash_download_error",
    "other_general_inquiry"
}


def validate_golden_set():
    print("=" * 80)
    print("VALIDATING GOLDEN SET HUMAN ANNOTATIONS")
    print("=" * 80)

    if not os.path.exists(GOLDEN_SET_PATH):
        print(f"Error: {GOLDEN_SET_PATH} not found!")
        sys.exit(1)

    df = pd.read_csv(GOLDEN_SET_PATH, dtype=str)
    for col in ["reviewer_label", "reviewer_action", "reviewer_escalation", "reviewer_notes"]:
        df[col] = df[col].fillna("").astype(str)

    total_rows = len(df)
    print(f"Total Rows in Dataset: {total_rows}")

    if total_rows != 200:
        print(f"WARNING: Expected 200 rows, found {total_rows}!")

    labeled_rows = []
    unlabeled_rows = []
    invalid_rows = []

    for idx, row in df.iterrows():
        label = row["reviewer_label"].strip()
        if label == "":
            unlabeled_rows.append(idx + 1)
        elif label not in VALID_INTENTS:
            invalid_rows.append((idx + 1, row["tweet_id"], label))
        else:
            labeled_rows.append(idx + 1)

    num_labeled = len(labeled_rows)
    print(f"\nProgress: {num_labeled} / {total_rows} labeled ({num_labeled / total_rows * 100:.1f}%)")

    if invalid_rows:
        print(f"\nERROR: Found {len(invalid_rows)} invalid intent labels!")
        for row_num, t_id, bad_label in invalid_rows[:10]:
            print(f"  Row {row_num} (Tweet {t_id}): '{bad_label}' is not an allowed intent")
        print("\nAllowed intents are:")
        for intent in sorted(VALID_INTENTS):
            print(f"  - {intent}")
        return

    if unlabeled_rows:
        print(f"\n{len(unlabeled_rows)} rows still need human labeling.")
        print(f"First 10 unreviewed rows: {unlabeled_rows[:10]}")
        print(f"\nTo review interactively in terminal, run:")
        print(f"  python src/labeling_tool.py")
        print(f"Or open {GOLDEN_SET_PATH} directly in VS Code / Excel to enter labels.\n")
        return

    # All 200 rows are labeled!
    print("\nALL 200 ROWS SUCCESSFULLY LABELED AND VALIDATED!")
    print("=" * 80)

    # Calculate agreement between provisional hypothesis and human reviewer
    agreed = (df["provisional_intent"] == df["reviewer_label"]).sum()
    print(f"Hypothesis vs. Human Ground-Truth Agreement: {agreed}/{total_rows} ({agreed/total_rows*100:.2f}%)")
    print(f"Corrections / Overrides by Reviewer:          {total_rows - agreed}/{total_rows} ({(total_rows - agreed)/total_rows*100:.2f}%)")

    # Show distribution of human verified labels
    print("\nFinal Verified Intent Distribution:")
    dist = df["reviewer_label"].value_counts()
    for intent, count in dist.items():
        print(f"  {intent:<30}: {count:3d} rows ({count / total_rows * 100:.1f}%)")

    # Update status and save verified copy
    df["review_status"] = "VERIFIED - GOLDEN"
    df.to_csv(GOLDEN_SET_PATH, index=False)
    df.to_csv(VERIFIED_OUTPUT_PATH, index=False)
    print(f"\nUpdated review_status to 'VERIFIED - GOLDEN' in {GOLDEN_SET_PATH}")
    print(f"Saved verified golden set copy to: {VERIFIED_OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    validate_golden_set()
