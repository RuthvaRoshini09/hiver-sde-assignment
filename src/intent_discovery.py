"""
src/intent_discovery.py
-----------------------
Performs empirical intent discovery on AppleSupport customer interactions.
Analyzes initial customer inquiries using frequency analysis, n-grams, and keyword matching.

Core 10-Class Taxonomy + Explicit Out-of-Scope (other_general_inquiry).
Generates:
1. data/processed/intent_analysis.csv
2. data/processed/golden_set_template.csv (Marked as PROVISIONAL / NOT HUMAN LABELLED)
"""

import os
import sys
import re
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
import numpy as np

PAIRS_PATH = "data/processed/applesupport_pairs.csv"
OUTPUT_DIR = "data/processed"
INTENT_ANALYSIS_PATH = os.path.join(OUTPUT_DIR, "intent_analysis.csv")
GOLDEN_TEMPLATE_PATH = os.path.join(OUTPUT_DIR, "golden_set_template.csv")

# 10 Core Grounded AppleSupport Intents
CORE_INTENT_DEFINITIONS = {
    "battery_drain_charging": {
        "description": "Rapid battery depletion, device overheating, charging cable/port failure, or abrupt shutdowns.",
        "patterns": [
            r"\bbattery\b", r"\bdrain(ing|s|ed)?\b", r"\bcharger\b", r"\bcharging\s+cable\b",
            r"\bwirelessly\s+charg\w+\b", r"\bnot\s+charging\b", r"\bwon'?t\s+charge\b",
            r"\bpercentage\b", r"\bdies\s+at\b", r"\bdying\b", r"\boverheat(ing)?\b", r"\bhot\b", r"\bbattery\s+life\b"
        ],
        "typical_action": "Ask for device model & iOS version; suggest Settings > Battery check; recommend updating to latest iOS or running battery diagnostic.",
        "escalation_expected": False,
        "auto_handling_suitability": "High",
        "evidence_needed": "Keywords indicating power, charging hardware, percentage drops, or battery longevity."
    },
    "keyboard_text_glitch": {
        "description": "iOS 11 text replacement autocorrect glitch ('I' changes to 'A [?]' or symbol), keyboard lagging/freezing.",
        "patterns": [
            r"\bkeyboard\b", r"\bautocorrect\b", r"\btyping\b", r"\bletter\s+i\b", r"\bcapital\s+i\b",
            r"\bsymbol\b", r"\bpredictive\b", r"\btext\s+replacement\b", r"\bquestion\s+mark\b",
            r"type\s+the\s+letter\s+i", r"type\s+i\s+and"
        ],
        "typical_action": "Suggest updating to iOS 11.1.1 bug-fix release or configuring temporary Text Replacement workaround.",
        "escalation_expected": False,
        "auto_handling_suitability": "High",
        "evidence_needed": "Mention of typing errors, autocorrect substitutions, or keyboard character anomalies."
    },
    "network_wifi_cellular": {
        "description": "Wi-Fi dropping/disconnecting, cellular 'No Service' or 'Searching', dropped calls, or slow mobile data.",
        "patterns": [
            r"\bwi-?fi\b", r"\bcellular\b", r"\bno\s+service\b", r"\bsearching\b",
            r"\bdropped\s+calls?\b", r"\bdrop\s+calls?\b", r"\blte\b", r"\bmobile\s+data\b",
            r"\bcarrier\b", r"\bsim\s+card\b", r"\bhotspot\b"
        ],
        "typical_action": "Suggest toggling Airplane Mode, restarting phone, or resetting Network Settings (Settings > General > Reset > Reset Network Settings).",
        "escalation_expected": False,
        "auto_handling_suitability": "High",
        "evidence_needed": "Inability to connect to Wi-Fi, cellular reception errors, or call drops."
    },
    "os_update_installation": {
        "description": "Failures downloading, verifying, or installing iOS updates; device stuck on Apple logo or recovery screen.",
        "patterns": [
            r"\b(can'?t|unable\s+to|won'?t)\s+(update|download\s+update|install\s+update)\b",
            r"\bupdate\s+failed\b", r"\bupdate\s+error\b", r"\bverifying\s+update\b",
            r"\bapple\s+logo\b", r"\brecovery\s+mode\b", r"\bitunes\s+restore\b",
            r"\bupdated\s+(my\s+phone|to\s+ios)\b", r"\boperating\s+system\b", r"\bsoftware\s+update\b"
        ],
        "typical_action": "Guide user through Settings > General > Software Update; provide Apple support link for recovery mode / iTunes backup.",
        "escalation_expected": False,
        "auto_handling_suitability": "High",
        "evidence_needed": "Mention of iOS update process, version installation, or boot loop."
    },
    "bluetooth_airpods_carplay": {
        "description": "AirPods dropping audio, Bluetooth pairing failure with accessories, or CarPlay connection drops.",
        "patterns": [
            r"\bairpods?\b", r"\bbluetooth\b", r"\bcarplay\b", r"\bpairing\b", r"\bpair\s+with\b",
            r"\bwireless\s+headphones?\b", r"\bbluetooth\s+disconnect\b"
        ],
        "typical_action": "Guide user to Settings > Bluetooth to forget device and re-pair; suggest resetting network settings or cleaning case.",
        "escalation_expected": False,
        "auto_handling_suitability": "High",
        "evidence_needed": "Mention of AirPods, Bluetooth accessories, car audio, or wireless audio drops."
    },
    "apple_id_account_access": {
        "description": "Forgotten passwords, Apple ID locked or disabled for security reasons, two-factor authentication codes not received.",
        "patterns": [
            r"\bapple\s+id\b", r"\bicloud\s+(password|account|backup|login)\b", r"\bforgot\s+password\b",
            r"\baccount\s+(locked|disabled)\b", r"\bverification\s+code\b", r"\btwo-factor\b", r"\b2fa\b",
            r"\breset\s+password\b", r"\bcan'?t\s+(sign|log)\s+in\b", r"\bregion\s+change\b"
        ],
        "typical_action": "Direct user to iforgot.apple.com; explain account recovery wait period; escalate to security team if identity proof needed.",
        "escalation_expected": True,
        "auto_handling_suitability": "Low",
        "evidence_needed": "Account lockout, password reset difficulty, or security verification blocks."
    },
    "hardware_buttons_audio": {
        "description": "Physical buttons (Home button, volume buttons, power button, mute switch) or built-in speakers/mic malfunctioning.",
        "patterns": [
            r"\bhome\s+button\b", r"\bpower\s+button\b", r"\bvolume\s+button\b",
            r"\b(built-in\s+)?speaker\b", r"\bmicrophone\b", r"\bmic\b", r"\bcamera\s+(black|flash|lens)\b",
            r"\bearpiece\b", r"\bmute\s+switch\b", r"\bringer\s+volume\b"
        ],
        "typical_action": "Run diagnostic steps; if physical failure persists, book Genius Bar appointment or send to authorized repair center.",
        "escalation_expected": True,
        "auto_handling_suitability": "Low",
        "evidence_needed": "Physical button failure, tactile switch issues, or internal speaker/microphone defects."
    },
    "billing_subscription_refund": {
        "description": "Financial charges from App Store / iTunes, unauthorized subscription renewals, payment methods, or refund requests.",
        "patterns": [
            r"\bcharg(ed|ing)\s+(my|the|a)?\s*(card|account|bank|credit|money)\b",
            r"\b(card|account|bank|credit)\s+was\s+charged\b",
            r"\bbilling\b", r"\bsubscription\b", r"\bitunes\s+(purchase|bill|charge|store)\b",
            r"\brefund\b", r"\breceipt\b", r"\bmoney\s+back\b", r"\bcancel\s+subscription\b",
            r"\bcharged\s+for\s+(free|an?)\s+apps?\b", r"\bdouble\s+charged\b", r"\bunauthorized\s+charge\b"
        ],
        "typical_action": "Direct user to reportaproblem.apple.com to review purchase history and submit refund; guide to subscription cancellation in Settings.",
        "escalation_expected": False,
        "auto_handling_suitability": "Medium",
        "evidence_needed": "Financial terms (charge, refund, billed, subscription, credit card) regarding Apple digital purchases."
    },
    "screen_touch_freeze": {
        "description": "Unresponsive touch screen, frozen user interface, display lag, or blank/black display.",
        "patterns": [
            r"\b(unresponsive|frozen|lagging|black|blank)\s+(screen|display)\b",
            r"\bscreen\s+(is\s+)?(frozen|black|blank|unresponsive|flickering)\b",
            r"\btouch\s+screen\b", r"\btouch\s+isn'?t\s+working\b", r"\bcan'?t\s+swipe\b",
            r"\bforce\s+reset\b", r"\bforce\s+restart\b"
        ],
        "typical_action": "Provide forced restart instructions (e.g. Power + Volume Down buttons); check for physical display damage.",
        "escalation_expected": False,
        "auto_handling_suitability": "Medium",
        "evidence_needed": "Touch input failure, frozen UI, or display unresponsive to finger gestures."
    },
    "app_crash_download_error": {
        "description": "Third-party apps (Spotify, YouTube, WhatsApp) crashing on launch, or App Store unable to download/update apps.",
        "patterns": [
            r"\bapp\s+(crash|crashes|crashing|closing)\b", r"\b(can'?t|won'?t)\s+download\s+(apps?|anything)\b",
            r"\bapp\s+store\s+(error|waiting|won'?t\s+open)\b", r"\bapps?\s+keep\s+crashing\b",
            r"\bforce\s+close\s+app\b", r"\bapp\s+won'?t\s+load\b"
        ],
        "typical_action": "Suggest force-closing app, checking available device storage, restarting device, or deleting and reinstalling app.",
        "escalation_expected": False,
        "auto_handling_suitability": "High",
        "evidence_needed": "Specific app crashes, freeze upon opening an app, or App Store download stalls."
    }
}

OUT_OF_SCOPE_DEFINITION = {
    "description": "Explicit out-of-scope / rejection class for customer messages that do not match the 10 supported core technical support intents (e.g., general feedback, casual comments, retail/shipping inquiries, rare cases).",
    "typical_action": "Acknowledge message politely; provide general Apple Support portal or deflect to appropriate retail/carrier channel.",
    "escalation_expected": False,
    "auto_handling_suitability": "Low",
    "evidence_needed": "Absence of core technical issue indicators; general remarks, feature requests, or retail preorder inquiries."
}


def compile_regex_matchers():
    matchers = {}
    for intent, info in CORE_INTENT_DEFINITIONS.items():
        combined = "|".join(info["patterns"])
        matchers[intent] = re.compile(combined, re.IGNORECASE)
    return matchers


def classify_intent_rule(text: str, matchers: dict) -> str:
    """Classifies customer inquiry based on pattern matching with priority hierarchy."""
    text_lower = text.lower()

    # Priority hierarchy across the 10 core classes
    priority_order = [
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

    for intent in priority_order:
        if matchers[intent].search(text_lower):
            return intent

    return "other_general_inquiry"


def main():
    print("=" * 80)
    print("UPDATING INTENT TAXONOMY: 10 CORE CLASSES + other_general_inquiry")
    print("=" * 80)

    if not os.path.exists(PAIRS_PATH):
        print(f"Error: {PAIRS_PATH} not found!")
        sys.exit(1)

    print(f"Loading {PAIRS_PATH}...")
    df = pd.read_csv(PAIRS_PATH)
    print(f"Total pairs loaded: {len(df):,d}")

    # Subsetting: initial turns that are usable
    df_initial = df[(df["is_initial_turn"] == True) & (df["is_usable"] == True)].copy()
    total_inquiries = len(df_initial)
    print(f"Usable initial customer inquiries: {total_inquiries:,d}\n")

    # Compile matchers
    matchers = compile_regex_matchers()

    # Assign candidate intents
    print("Classifying customer inquiries against 10 core intents + other_general_inquiry...")
    df_initial["candidate_intent"] = df_initial["customer_text_clean"].apply(
        lambda t: classify_intent_rule(t, matchers)
    )

    counts = df_initial["candidate_intent"].value_counts()
    print("\nFinal Intent Distribution:")
    for intent, count in counts.items():
        pct = (count / total_inquiries) * 100
        print(f"  {intent:30s}: {count:6,d} ({pct:5.2f}%)")

    # Build intent_analysis.csv
    intent_analysis_rows = []
    golden_template_rows = []

    print("\nExtracting representative examples and support responses...")

    # 1. Process 10 core intents
    for intent_name, info in CORE_INTENT_DEFINITIONS.items():
        sub = df_initial[df_initial["candidate_intent"] == intent_name]
        count = len(sub)
        pct = (count / total_inquiries) * 100

        # Sample 5 diverse customer messages and matching support replies
        sample_pairs = sub.head(5)
        sample_cust = [row["customer_text_clean"] for _, row in sample_pairs.iterrows()]
        sample_supp = [row["support_text_clean"] for _, row in sample_pairs.iterrows()]

        intent_analysis_rows.append({
            "intent_name": intent_name,
            "category_type": "core_intent",
            "description": info["description"],
            "approx_count": count,
            "dataset_pct": round(pct, 2),
            "typical_action": info["typical_action"],
            "escalation_expected": info["escalation_expected"],
            "auto_handling_suitability": info["auto_handling_suitability"],
            "sample_customer_1": sample_cust[0] if len(sample_cust) > 0 else "",
            "sample_customer_2": sample_cust[1] if len(sample_cust) > 1 else "",
            "sample_customer_3": sample_cust[2] if len(sample_cust) > 2 else "",
            "sample_customer_4": sample_cust[3] if len(sample_cust) > 3 else "",
            "sample_customer_5": sample_cust[4] if len(sample_cust) > 4 else "",
            "sample_support_response": sample_supp[0] if len(sample_supp) > 0 else ""
        })

        # Add 10 candidate examples for each intent into the golden-set template
        golden_candidates = sub.iloc[5:15] if len(sub) >= 15 else sub.head(10)
        for _, row in golden_candidates.iterrows():
            golden_template_rows.append({
                "example_id": f"gold_{len(golden_template_rows) + 1:03d}",
                "customer_tweet_id": row["customer_tweet_id"],
                "customer_message": row["customer_text_clean"],
                "intent": intent_name,
                "expected_action": info["typical_action"],
                "escalation_expected": info["escalation_expected"],
                "evidence_needed": info["evidence_needed"],
                "review_status": "PROVISIONAL - NOT HUMAN LABELLED",
                "notes": f"[PROVISIONAL] Candidate example for '{intent_name}'. Pair ID: {row['pair_id']}. Awaiting human audit."
            })

    # 2. Process other_general_inquiry (explicit out-of-scope rejection class)
    sub_other = df_initial[df_initial["candidate_intent"] == "other_general_inquiry"]
    count_other = len(sub_other)
    pct_other = (count_other / total_inquiries) * 100
    sample_other = sub_other.head(5)
    sample_cust_other = [row["customer_text_clean"] for _, row in sample_other.iterrows()]
    sample_supp_other = [row["support_text_clean"] for _, row in sample_other.iterrows()]

    intent_analysis_rows.append({
        "intent_name": "other_general_inquiry",
        "category_type": "out_of_scope_rejection",
        "description": OUT_OF_SCOPE_DEFINITION["description"],
        "approx_count": count_other,
        "dataset_pct": round(pct_other, 2),
        "typical_action": OUT_OF_SCOPE_DEFINITION["typical_action"],
        "escalation_expected": OUT_OF_SCOPE_DEFINITION["escalation_expected"],
        "auto_handling_suitability": OUT_OF_SCOPE_DEFINITION["auto_handling_suitability"],
        "sample_customer_1": sample_cust_other[0] if len(sample_cust_other) > 0 else "",
        "sample_customer_2": sample_cust_other[1] if len(sample_cust_other) > 1 else "",
        "sample_customer_3": sample_cust_other[2] if len(sample_cust_other) > 2 else "",
        "sample_customer_4": sample_cust_other[3] if len(sample_cust_other) > 3 else "",
        "sample_customer_5": sample_cust_other[4] if len(sample_cust_other) > 4 else "",
        "sample_support_response": sample_supp_other[0] if len(sample_supp_other) > 0 else ""
    })

    # Add 20 candidate examples of other_general_inquiry into golden-set template (total 120 rows)
    # Includes preorder/shipping and general comments
    golden_other_candidates = sub_other.iloc[10:30] if len(sub_other) >= 30 else sub_other.head(20)
    for _, row in golden_other_candidates.iterrows():
        golden_template_rows.append({
            "example_id": f"gold_{len(golden_template_rows) + 1:03d}",
            "customer_tweet_id": row["customer_tweet_id"],
            "customer_message": row["customer_text_clean"],
            "intent": "other_general_inquiry",
            "expected_action": OUT_OF_SCOPE_DEFINITION["typical_action"],
            "escalation_expected": False,
            "evidence_needed": "Does not match any of the 10 core technical intent definitions.",
            "review_status": "PROVISIONAL - NOT HUMAN LABELLED",
            "notes": f"[PROVISIONAL] Out-of-scope / rejection candidate. Pair ID: {row['pair_id']}. Awaiting human audit."
        })

    # Save intent_analysis.csv
    df_analysis = pd.DataFrame(intent_analysis_rows)
    df_analysis.to_csv(INTENT_ANALYSIS_PATH, index=False)
    print(f"\nSaved intent analysis to: {INTENT_ANALYSIS_PATH}")

    # Save golden_set_template.csv
    df_golden = pd.DataFrame(golden_template_rows)
    df_golden.to_csv(GOLDEN_TEMPLATE_PATH, index=False)
    print(f"Saved golden set template ({len(df_golden)} PROVISIONAL candidate rows) to: {GOLDEN_TEMPLATE_PATH}\n")

    # Print summary table
    print("=" * 95)
    print("FINAL APPLESUPPORT INTENT TAXONOMY (10 CORE + 1 OUT-OF-SCOPE)")
    print("=" * 95)
    print(f"{'Intent Name':<30} | {'Type':<22} | {'Count':<8} | {'Share (%)':<10} | {'Auto-Handle':<12} | {'Escalate'}")
    print("-" * 95)
    for _, row in df_analysis.iterrows():
        print(f"{row['intent_name']:<30} | {row['category_type']:<22} | {row['approx_count']:<8,d} | {row['dataset_pct']:<10.2f}% | {row['auto_handling_suitability']:<12} | {row['escalation_expected']}")

    print("=" * 95)
    print("ADJUSTMENTS COMPLETE")
    print("=" * 95)


if __name__ == "__main__":
    main()
