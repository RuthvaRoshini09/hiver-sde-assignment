"""
src/audit_golden_set.py
-----------------------
Audits the current 200-row Golden Evaluation Set (data/processed/golden_set_200.csv).

Read-only analysis:
- Does NOT modify golden_set_200.csv.
- Does NOT replace human labels or make automatic corrections.
- Compares provisional_intent vs. reviewer_label for every row.
- Identifies all disagreement cases with exact row details.
- Groups disagreements by (provisional_intent -> reviewer_label).
- Compares intended sampling design (15 x 10 core + 50 out-of-scope) vs. actual reviewer distribution.
- Flags suspicious patterns, input misalignments, or systematic drift.
"""

import os
import sys
from collections import Counter

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

GOLDEN_SET_PATH = os.path.join("data", "processed", "golden_set_200.csv")

INTENDED_DISTRIBUTION = {
    "battery_drain_charging": 15,
    "keyboard_text_glitch": 15,
    "network_wifi_cellular": 15,
    "os_update_installation": 15,
    "bluetooth_airpods_carplay": 15,
    "apple_id_account_access": 15,
    "hardware_buttons_audio": 15,
    "billing_subscription_refund": 15,
    "screen_touch_freeze": 15,
    "app_crash_download_error": 15,
    "other_general_inquiry": 50
}


def audit_golden_set():
    print("=" * 90)
    print("AUDITING GOLDEN EVALUATION SET (data/processed/golden_set_200.csv)")
    print("=" * 90)

    if not os.path.exists(GOLDEN_SET_PATH):
        print(f"Error: {GOLDEN_SET_PATH} does not exist!")
        sys.exit(1)

    df = pd.read_csv(GOLDEN_SET_PATH, dtype=str)
    for col in ["reviewer_label", "reviewer_action", "reviewer_escalation", "reviewer_notes"]:
        df[col] = df[col].fillna("").astype(str)

    total_rows = len(df)
    print(f"Total Rows Analyzed: {total_rows}\n")

    # 1. Compare provisional_intent vs reviewer_label
    agreements = []
    disagreements = []

    for idx, row in df.iterrows():
        row_num = idx + 1
        p_intent = row["provisional_intent"].strip()
        r_label = row["reviewer_label"].strip()

        info = {
            "row_number": row_num,
            "tweet_id": row["tweet_id"],
            "clean_message": row["clean_message"],
            "provisional_intent": p_intent,
            "reviewer_label": r_label,
            "reviewer_action": row["reviewer_action"],
            "reviewer_escalation": row["reviewer_escalation"],
            "reviewer_notes": row["reviewer_notes"]
        }

        if p_intent == r_label:
            agreements.append(info)
        else:
            disagreements.append(info)

    print(f"Total Agreements:    {len(agreements):3d} / {total_rows} ({len(agreements) / total_rows * 100:.1f}%)")
    print(f"Total Disagreements: {len(disagreements):3d} / {total_rows} ({len(disagreements) / total_rows * 100:.1f}%)\n")

    # 2. Intended vs Actual Distribution Comparison
    print("=" * 90)
    print("INTENDED SAMPLING DISTRIBUTION VS. ACTUAL REVIEWER_LABEL DISTRIBUTION")
    print("=" * 90)
    actual_counts = df["reviewer_label"].value_counts().to_dict()

    print(f"{'Intent Label':<30} | {'Intended':<10} | {'Actual':<10} | {'Delta':<10} | {'Status'}")
    print("-" * 90)

    all_labels = sorted(list(set(list(INTENDED_DISTRIBUTION.keys()) + list(actual_counts.keys()))))
    for label in all_labels:
        intended = INTENDED_DISTRIBUTION.get(label, 0)
        actual = actual_counts.get(label, 0)
        delta = actual - intended
        delta_str = f"+{delta}" if delta > 0 else f"{delta}"
        status = "MATCH" if delta == 0 else ("SURPLUS" if delta > 0 else "DEFICIT")
        print(f"{label:<30} | {intended:<10} | {actual:<10} | {delta_str:<10} | {status}")

    # 3. Disagreement Grouping (Transitions)
    print("\n" + "=" * 90)
    print("DISAGREEMENT TRANSITIONS (provisional_intent -> reviewer_label)")
    print("=" * 90)
    transition_counter = Counter(
        (d["provisional_intent"], d["reviewer_label"]) for d in disagreements
    )

    sorted_transitions = sorted(transition_counter.items(), key=lambda x: x[1], reverse=True)
    for (p_from, r_to), count in sorted_transitions:
        print(f"  {p_from:<28} -> {r_to:<28}: {count:3d} cases")

    # 4. Detailed Breakdown of Disagreements
    print("\n" + "=" * 90)
    print(f"DETAILED LIST OF ALL {len(disagreements)} DISAGREEMENTS")
    print("=" * 90)

    for d in disagreements:
        print(f"\n[Row {d['row_number']:03d}] Tweet ID: {d['tweet_id']}")
        print(f"  Message:            \"{d['clean_message']}\"")
        print(f"  Provisional Intent: {d['provisional_intent']}")
        print(f"  Reviewer Label:     {d['reviewer_label']}")
        print(f"  Reviewer Action:    {d['reviewer_action']}")
        print(f"  Reviewer Escalate:  {d['reviewer_escalation']}")
        if d['reviewer_notes']:
            print(f"  Reviewer Notes:     {d['reviewer_notes']}")

    print("\n" + "=" * 90)
    print("AUDIT COMPLETE - NO FILES MODIFIED")
    print("=" * 90)

    return {
        "total_rows": total_rows,
        "agreements": len(agreements),
        "disagreements": len(disagreements),
        "actual_counts": actual_counts,
        "disagreement_list": disagreements,
        "transition_counter": transition_counter
    }


if __name__ == "__main__":
    audit_golden_set()
