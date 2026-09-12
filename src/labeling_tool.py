"""
src/labeling_tool.py
--------------------
Accelerated but SAFE interactive human-labeling tool for data/processed/golden_set_200.csv.

Key Design & Workflow:
1. Loads the 200 real AppleSupport customer messages.
2. For each row, displays:
   - Row number [X / 200]
   - tweet_id
   - clean_message
   - provisional_intent (SUGGESTION ONLY, never ground truth)
3. Clearly shows the 11 allowed labels with numbers:
   1  battery_drain_charging
   2  keyboard_text_glitch
   3  network_wifi_cellular
   4  os_update_installation
   5  bluetooth_airpods_carplay
   6  apple_id_account_access
   7  hardware_buttons_audio
   8  billing_subscription_refund
   9  screen_touch_freeze
   10 app_crash_download_error
   11 other_general_inquiry
4. Single-keystroke speed:
   - Press [Enter] to accept the suggested provisional_intent.
   - Enter [1-11] to override with a different intent.
5. Confirmation:
   - Displays "Selected: <label>"
   - y = save / proceed
   - n = choose label again
6. Escalation:
   - "Escalation expected? (y/n, default n): "
7. Optional Reviewer Notes:
   - "Reviewer notes (optional, press Enter to skip): "
8. Navigation:
   - 'b' = go back one row
   - 's' = skip row
   - 'q' = save and quit
9. Safety Guarantees:
   - Zero state carry-over across rows.
   - Preserves tweet_id, clean_message, provisional_intent, expected_action, escalation_expected unchanged.
   - Saves immediately after every row.
   - review_status remains 'PROVISIONAL' until all 200 rows are reviewed.
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

GOLDEN_SET_PATH = os.path.join("data", "processed", "golden_set_200.csv")
VERIFIED_OUTPUT_PATH = os.path.join("data", "processed", "golden_set_verified.csv")

INTENT_MAP = {
    "1": "battery_drain_charging",
    "2": "keyboard_text_glitch",
    "3": "network_wifi_cellular",
    "4": "os_update_installation",
    "5": "bluetooth_airpods_carplay",
    "6": "apple_id_account_access",
    "7": "hardware_buttons_audio",
    "8": "billing_subscription_refund",
    "9": "screen_touch_freeze",
    "10": "app_crash_download_error",
    "11": "other_general_inquiry",
}

DEFAULT_ACTIONS = {
    "battery_drain_charging": "Ask for device model & iOS version; suggest Settings > Battery check; recommend updating or battery diagnostic.",
    "keyboard_text_glitch": "Suggest updating to iOS 11.1.1 bug-fix release or setting temporary Text Replacement workaround.",
    "network_wifi_cellular": "Suggest toggling Airplane Mode, restarting phone, or resetting Network Settings.",
    "os_update_installation": "Guide user through Settings > General > Software Update; provide Apple support link for recovery mode / iTunes backup.",
    "bluetooth_airpods_carplay": "Guide user to Settings > Bluetooth to forget device and re-pair; suggest network reset or case cleaning.",
    "apple_id_account_access": "Direct user to iforgot.apple.com; explain account recovery wait period; escalate if identity proof needed.",
    "hardware_buttons_audio": "Run diagnostic steps; if physical failure persists, book Genius Bar appointment or send to service center.",
    "billing_subscription_refund": "Direct user to reportaproblem.apple.com to view purchase history and request refund; guide to subscription cancellation.",
    "screen_touch_freeze": "Provide forced restart instructions (Power + Vol Down); check for physical display damage.",
    "app_crash_download_error": "Suggest force-closing app, checking available storage, restarting device, or deleting and reinstalling app.",
    "other_general_inquiry": "Acknowledge message politely; provide general Apple Support portal or deflect to retail/carrier channel."
}


def print_menu():
    print("Allowed Labels:")
    print("  1  battery_drain_charging        7  hardware_buttons_audio")
    print("  2  keyboard_text_glitch          8  billing_subscription_refund")
    print("  3  network_wifi_cellular         9  screen_touch_freeze")
    print("  4  os_update_installation        10 app_crash_download_error")
    print("  5  bluetooth_airpods_carplay     11 other_general_inquiry")
    print("  6  apple_id_account_access")


def run_labeling_tool():
    if not os.path.exists(GOLDEN_SET_PATH):
        print(f"Error: {GOLDEN_SET_PATH} not found!")
        sys.exit(1)

    df = pd.read_csv(GOLDEN_SET_PATH, dtype=str)
    for col in ["reviewer_label", "reviewer_action", "reviewer_escalation", "reviewer_notes", "review_status"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    total_rows = len(df)
    labeled_count = (df["reviewer_label"].str.strip() != "").sum()

    print("=" * 80)
    print("ACCELERATED HUMAN REVIEW TOOL (200 GOLDEN SET EXAMPLES)")
    print("=" * 80)
    print(f"Total rows:      {total_rows}")
    print(f"Already labeled: {labeled_count}")
    print(f"Remaining:       {total_rows - labeled_count}")
    print("=" * 80)

    # Resume from first unlabeled row
    first_unlabeled = 0
    for i in range(total_rows):
        if df.at[i, "reviewer_label"].strip() == "":
            first_unlabeled = i
            break

    idx = first_unlabeled

    while idx < total_rows:
        row = df.iloc[idx]
        current_labeled = (df["reviewer_label"].str.strip() != "").sum()

        print("\n" + "=" * 80)
        print(f"Row [{idx + 1} / {total_rows}]  |  Tweet ID: {row['tweet_id']}  |  Labeled: {current_labeled}/{total_rows}")
        print("=" * 80)
        print("clean_message:")
        print(f"  \"{row['clean_message']}\"")
        print()
        provisional = str(row["provisional_intent"]).strip()
        print(f"provisional_intent (SUGGESTION ONLY): {provisional}")
        print("-" * 80)

        existing_label = row["reviewer_label"].strip()
        if existing_label:
            print(f"[Existing label for this row: '{existing_label}']")

        print_menu()
        print("-" * 80)
        print("Controls: [Enter] = Accept suggestion | [1-11] = Override | [b] Back | [s] Skip | [q] Save & Quit")

        # Step-by-step label selection with loop for 'n' (choose label again)
        selected_label = None
        while True:
            choice = input(f"\nSelect label for Row [{idx + 1}/{total_rows}] ([Enter] for suggestion): ").strip().lower()

            if choice == "q":
                print("\nSaving and quitting...")
                df.to_csv(GOLDEN_SET_PATH, index=False)
                print(f"Progress saved to {GOLDEN_SET_PATH}. ({current_labeled}/{total_rows} labeled)")
                sys.exit(0)

            elif choice == "b":
                if idx > 0:
                    idx -= 1
                    print(f"Going back to Row {idx + 1}...")
                else:
                    print("Already at Row 1.")
                selected_label = None
                break

            elif choice == "s":
                print(f"Row {idx + 1} skipped.")
                idx += 1
                selected_label = None
                break

            elif choice == "":
                # ENTER pressed: accept provisional_intent if valid
                if provisional in INTENT_MAP.values():
                    selected_label = provisional
                else:
                    print(f"Provisional intent '{provisional}' is not recognized. Please choose 1-11.")
                    continue
            elif choice in INTENT_MAP:
                selected_label = INTENT_MAP[choice]
            else:
                print(f"Invalid input '{choice}'. Enter 1-11, press [Enter] to accept suggestion, or 'b'/'s'/'q'.")
                continue

            # Confirm chosen label
            print(f"\nSelected: {selected_label}")
            confirm = input("Save this label? (y/n): ").strip().lower()
            if confirm in ["y", "yes"]:
                break
            else:
                print("Label selection cancelled. Please choose label again.")
                selected_label = None
                continue

        # If user navigated via 'b' or 's', continue the outer row loop
        if selected_label is None:
            continue

        # Step 8: Escalation prompt (y/n, default n)
        esc_choice = input("Escalation expected? (y/n, default n): ").strip().lower()
        if esc_choice in ["y", "yes"]:
            reviewer_escalation = "True"
        else:
            reviewer_escalation = "False"

        # Step 9: Optional Reviewer Notes
        reviewer_notes = input("Reviewer notes (optional, press Enter to skip): ").strip()

        # Determine reviewer_action
        if selected_label == provisional and row.get("expected_action", "").strip():
            reviewer_action = row["expected_action"].strip()
        else:
            reviewer_action = DEFAULT_ACTIONS.get(selected_label, "")

        # Step 14 & 15: Update fields (review_status stays PROVISIONAL until all 200 are reviewed)
        df.at[idx, "reviewer_label"] = selected_label
        df.at[idx, "reviewer_action"] = reviewer_action
        df.at[idx, "reviewer_escalation"] = reviewer_escalation
        df.at[idx, "reviewer_notes"] = reviewer_notes
        df.at[idx, "review_status"] = "PROVISIONAL"

        # Step 10: Save immediately after every row
        df.to_csv(GOLDEN_SET_PATH, index=False)
        print(f"\n[SAVED] Row {idx + 1} recorded: label='{selected_label}', escalation={reviewer_escalation}.")

        idx += 1

    # Check completion
    final_labeled = (df["reviewer_label"].str.strip() != "").sum()
    print("\n" + "=" * 80)
    print(f"REVIEW SESSION COMPLETE: {final_labeled}/{total_rows} rows labeled.")
    print("=" * 80)
    if final_labeled == total_rows:
        df["review_status"] = "VERIFIED - GOLDEN"
        df.to_csv(GOLDEN_SET_PATH, index=False)
        df.to_csv(VERIFIED_OUTPUT_PATH, index=False)
        print(f"All 200 rows verified! Updated review_status to 'VERIFIED - GOLDEN'.")
        print(f"Saved verified dataset to {VERIFIED_OUTPUT_PATH}.")


if __name__ == "__main__":
    run_labeling_tool()
