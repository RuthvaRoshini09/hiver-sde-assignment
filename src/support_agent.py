"""
src/support_agent.py
--------------------
AI Customer Support Agent for AppleSupport.

Pipeline:
  Customer Message
       |
  1. Intent Classification  (reuses Baseline 1 TF-IDF + Logistic Regression)
       |
  2. Retrieve Similar Historical Cases  (TF-IDF cosine similarity on training corpus)
       |
  3. Draft Grounded Reply  (template from retrieved AppleSupport responses)
       |
  4. Auto-Handle OR Escalate  (rule-based on intent, confidence, evidence quality)

Key Rules:
  - The 200-row golden set is NEVER used for retrieval, training, or tuning.
  - Replies are ONLY grounded in real historical AppleSupport responses.
  - Nothing is invented: no fake policies, no fake URLs, no fake refunds.
  - Escalation triggers for security-sensitive intents, low confidence,
    or insufficient retrieval evidence.
"""

import os
import sys
from typing import Dict, Any, List, Optional

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data_cleaning import clean_tweet_text

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join("data", "processed", "baseline1_tfidf_logreg.joblib")
PAIRS_PATH = os.path.join("data", "processed", "applesupport_pairs.csv")
GOLDEN_SET_PATH = os.path.join("data", "processed", "golden_set_verified.csv")
RETRIEVAL_INDEX_PATH = os.path.join("data", "processed", "retrieval_index.joblib")

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.35       # Below this -> escalate due to low confidence
MIN_RETRIEVAL_SIMILARITY = 0.15   # Minimum cosine similarity for a useful case
MIN_USEFUL_CASES = 1              # Need at least 1 case above threshold
TOP_K_RETRIEVE = 5                # Number of similar cases to retrieve

# ---------------------------------------------------------------------------
# Intent-specific escalation rules
# ---------------------------------------------------------------------------
ALWAYS_ESCALATE_INTENTS = {
    "apple_id_account_access",   # Account/security access — normally escalate
}

CONDITIONAL_ESCALATE_INTENTS = {
    "hardware_buttons_audio",    # Escalate when hardware failure/damage/repair implied
    "other_general_inquiry",     # Escalate when system can't confidently identify intent
}

HARDWARE_ESCALATION_KEYWORDS = [
    "broken", "cracked", "smashed", "shattered", "repair", "replace",
    "damage", "damaged", "physical", "genius bar", "appointment",
    "service center", "warranty", "water damage", "dropped",
    "bent", "dent", "scratched", "hardware", "defective",
]

# ---------------------------------------------------------------------------
# Default actions per intent (for grounded reply assembly)
# ---------------------------------------------------------------------------
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
    "other_general_inquiry": "Acknowledge message politely; provide general Apple Support portal or deflect to retail/carrier channel.",
}


# ===========================================================================
# Retrieval Index Builder
# ===========================================================================
class RetrievalIndex:
    """
    TF-IDF based retrieval index over historical AppleSupport customer-support
    pairs. Excludes all 200 golden set tweets to prevent data leakage.

    Stores:
      - TF-IDF vectorizer fitted on customer messages
      - TF-IDF matrix for the corpus
      - Corresponding customer messages, support replies, and metadata
    """

    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.corpus_customer_texts: List[str] = []
        self.corpus_support_texts: List[str] = []
        self.corpus_tweet_ids: List[str] = []
        self.corpus_intents: List[str] = []  # Rule-based labels for filtering

    def build(self):
        """
        Builds the retrieval index from applesupport_pairs.csv,
        strictly excluding golden set tweet IDs.
        """
        print("=" * 80)
        print("BUILDING RETRIEVAL INDEX (EXCLUDING GOLDEN SET)")
        print("=" * 80)

        # Load golden set tweet IDs for strict exclusion
        df_golden = pd.read_csv(GOLDEN_SET_PATH)
        golden_ids = set(df_golden["tweet_id"].astype(int))
        print(f"Golden set IDs to exclude: {len(golden_ids)}")

        # Load all pairs
        df_pairs = pd.read_csv(PAIRS_PATH)
        print(f"Total pairs loaded: {len(df_pairs):,d}")

        # Filter for usable pairs with meaningful support replies
        mask = (
            (df_pairs["is_usable"] == True) &
            (df_pairs["support_text_clean"].fillna("").str.strip() != "")
        )
        df_usable = df_pairs[mask].copy()
        print(f"Usable pairs with support replies: {len(df_usable):,d}")

        # Strictly exclude golden set tweets
        df_usable = df_usable[~df_usable["customer_tweet_id"].isin(golden_ids)].copy()
        print(f"After excluding golden set: {len(df_usable):,d}")

        # Extract texts
        self.corpus_customer_texts = df_usable["customer_text_clean"].fillna("").astype(str).tolist()
        self.corpus_support_texts = df_usable["support_text_clean"].fillna("").astype(str).tolist()
        self.corpus_tweet_ids = df_usable["customer_tweet_id"].astype(str).tolist()

        # Build TF-IDF index on customer messages
        print("\nFitting retrieval TF-IDF (unigrams + bigrams)...")
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000,
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus_customer_texts)
        print(f"Retrieval index built: {self.tfidf_matrix.shape[0]:,d} documents, "
              f"{self.tfidf_matrix.shape[1]:,d} features")
        print("=" * 80)

    def save(self, path: str = RETRIEVAL_INDEX_PATH):
        """Serialize the retrieval index to disk."""
        payload = {
            "vectorizer": self.vectorizer,
            "tfidf_matrix": self.tfidf_matrix,
            "corpus_customer_texts": self.corpus_customer_texts,
            "corpus_support_texts": self.corpus_support_texts,
            "corpus_tweet_ids": self.corpus_tweet_ids,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(payload, path)
        print(f"Retrieval index saved to: {path}")

    def load(self, path: str = RETRIEVAL_INDEX_PATH):
        """Load the retrieval index from disk."""
        data = joblib.load(path)
        self.vectorizer = data["vectorizer"]
        self.tfidf_matrix = data["tfidf_matrix"]
        self.corpus_customer_texts = data["corpus_customer_texts"]
        self.corpus_support_texts = data["corpus_support_texts"]
        self.corpus_tweet_ids = data["corpus_tweet_ids"]

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVE) -> List[Dict[str, Any]]:
        """
        Retrieves the top-k most similar historical cases to the query
        using cosine similarity on TF-IDF vectors.

        Returns a list of dicts with:
          - similarity: cosine similarity score
          - customer_message: the original customer text
          - support_reply: the AppleSupport response
          - tweet_id: the tweet ID
        """
        if self.vectorizer is None or self.tfidf_matrix is None:
            raise RuntimeError("Retrieval index not loaded. Call build() or load() first.")

        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for i in top_indices:
            results.append({
                "similarity": round(float(sims[i]), 4),
                "customer_message": self.corpus_customer_texts[i],
                "support_reply": self.corpus_support_texts[i],
                "tweet_id": self.corpus_tweet_ids[i],
            })
        return results


# ===========================================================================
# Support Agent
# ===========================================================================

# Module-level singletons (loaded once, reused across calls)
_classifier_pipeline: Optional[Dict[str, Any]] = None
_retrieval_index: Optional[RetrievalIndex] = None


def _load_classifier():
    """Load the Baseline 1 TF-IDF + Logistic Regression model."""
    global _classifier_pipeline
    if _classifier_pipeline is not None:
        return _classifier_pipeline

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Classifier model not found at {MODEL_PATH}. "
            "Run: python src/intent_classifier.py  to train it first."
        )
    _classifier_pipeline = joblib.load(MODEL_PATH)
    return _classifier_pipeline


def _load_retrieval_index():
    """Load or build the retrieval index."""
    global _retrieval_index
    if _retrieval_index is not None:
        return _retrieval_index

    _retrieval_index = RetrievalIndex()
    if os.path.exists(RETRIEVAL_INDEX_PATH):
        print("Loading existing retrieval index...")
        _retrieval_index.load()
        print(f"Retrieval index loaded: {_retrieval_index.tfidf_matrix.shape[0]:,d} documents")
    else:
        print("Retrieval index not found. Building from scratch...")
        _retrieval_index.build()
        _retrieval_index.save()
    return _retrieval_index


def classify_message(clean_text: str) -> Dict[str, Any]:
    """
    Classify a cleaned customer message using the Baseline 1 model.

    Returns:
        dict with predicted_intent, confidence, and top_candidates
    """
    pipeline = _load_classifier()
    vectorizer = pipeline["vectorizer"]
    classifier = pipeline["classifier"]
    classes = pipeline["classes"]

    if not clean_text:
        return {
            "predicted_intent": "other_general_inquiry",
            "confidence": 0.0,
            "top_candidates": [],
        }

    X = vectorizer.transform([clean_text])
    probs = classifier.predict_proba(X)[0]
    ranked = np.argsort(probs)[::-1]

    top_intent = classes[ranked[0]]
    confidence = float(probs[ranked[0]])

    top_candidates = [
        {"intent": classes[i], "probability": round(float(probs[i]), 4)}
        for i in ranked[:3]
    ]

    return {
        "predicted_intent": top_intent,
        "confidence": round(confidence, 4),
        "top_candidates": top_candidates,
    }


def draft_reply(
    predicted_intent: str,
    retrieved_cases: List[Dict[str, Any]],
    clean_text: str,
) -> str:
    """
    Draft a grounded reply based on retrieved historical AppleSupport responses.

    The reply is assembled from the best-matching historical support reply,
    NOT invented. If no useful cases are found, returns a generic hand-off.
    """
    # Filter to cases with reasonable similarity
    useful_cases = [c for c in retrieved_cases if c["similarity"] >= MIN_RETRIEVAL_SIMILARITY]

    if not useful_cases:
        # No grounded evidence — provide safe fallback
        return (
            "Thank you for reaching out to Apple Support. "
            "We'd like to help you further — please DM us your device details "
            "so we can look into this for you."
        )

    # Use the top retrieved support reply as the basis for the drafted response
    best_case = useful_cases[0]
    base_reply = best_case["support_reply"]

    # If the best reply is a DM deflection, keep it (that IS what Apple does)
    # If it's substantive troubleshooting, use it directly — it's grounded
    return base_reply


def decide_escalation(
    predicted_intent: str,
    confidence: float,
    retrieved_cases: List[Dict[str, Any]],
    clean_text: str,
) -> Dict[str, Any]:
    """
    Determine whether to auto-handle or escalate.

    Escalation triggers:
      1. apple_id_account_access → always escalate (security)
      2. hardware_buttons_audio → escalate if message implies physical damage/repair
      3. other_general_inquiry → escalate (unsupported intent)
      4. Low classifier confidence (below threshold)
      5. No useful retrieved cases (insufficient evidence)

    Returns:
        dict with decision ("AUTO_HANDLE" or "ESCALATE") and escalation_reason
    """
    useful_cases = [c for c in retrieved_cases if c["similarity"] >= MIN_RETRIEVAL_SIMILARITY]

    # Rule 1: Always escalate apple_id_account_access
    if predicted_intent in ALWAYS_ESCALATE_INTENTS:
        return {
            "decision": "ESCALATE",
            "escalation_reason": (
                f"Intent '{predicted_intent}' involves account/security access. "
                "Requires human agent for identity verification."
            ),
        }

    # Rule 2: hardware_buttons_audio — check for hardware failure keywords
    if predicted_intent == "hardware_buttons_audio":
        text_lower = clean_text.lower()
        if any(kw in text_lower for kw in HARDWARE_ESCALATION_KEYWORDS):
            return {
                "decision": "ESCALATE",
                "escalation_reason": (
                    "Hardware failure, physical damage, or repair service indicated. "
                    "Requires Genius Bar booking or service center referral."
                ),
            }

    # Rule 3: other_general_inquiry — system cannot confidently identify intent
    if predicted_intent == "other_general_inquiry":
        return {
            "decision": "ESCALATE",
            "escalation_reason": (
                "Message does not match any supported intent category. "
                "Routing to human agent for proper handling."
            ),
        }

    # Rule 4: Low classifier confidence
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "decision": "ESCALATE",
            "escalation_reason": (
                f"Classifier confidence ({confidence:.2%}) is below the "
                f"minimum threshold ({CONFIDENCE_THRESHOLD:.2%}). "
                "Insufficient certainty for automated handling."
            ),
        }

    # Rule 5: Insufficient retrieval evidence
    if len(useful_cases) < MIN_USEFUL_CASES:
        return {
            "decision": "ESCALATE",
            "escalation_reason": (
                "No sufficiently similar historical cases found in the retrieval corpus. "
                "Cannot ground a response — routing to human agent."
            ),
        }

    # All checks passed — auto-handle
    return {
        "decision": "AUTO_HANDLE",
        "escalation_reason": "N/A — confidence and retrieval evidence are sufficient.",
    }


def run_agent(message: str) -> Dict[str, Any]:
    """
    Main agent entry point. Processes a customer message through the
    full pipeline:

      1. Clean the input text
      2. Classify intent
      3. Retrieve similar historical cases
      4. Draft a grounded reply
      5. Decide auto-handle or escalate

    Args:
        message: Raw customer message string.

    Returns:
        Dict with:
          - predicted_intent
          - confidence
          - retrieved_cases (list of similar historical interactions)
          - drafted_reply
          - decision ("AUTO_HANDLE" or "ESCALATE")
          - escalation_reason
    """
    # Step 1: Clean the input
    clean_text = clean_tweet_text(message)

    # Step 2: Classify intent using Baseline 1 model
    classification = classify_message(clean_text)

    # Step 3: Retrieve similar historical cases
    index = _load_retrieval_index()
    retrieved_cases = index.retrieve(clean_text, top_k=TOP_K_RETRIEVE)

    # Step 4: Draft a grounded reply from retrieved evidence
    reply = draft_reply(
        predicted_intent=classification["predicted_intent"],
        retrieved_cases=retrieved_cases,
        clean_text=clean_text,
    )

    # Step 5: Decide auto-handle vs. escalate
    escalation = decide_escalation(
        predicted_intent=classification["predicted_intent"],
        confidence=classification["confidence"],
        retrieved_cases=retrieved_cases,
        clean_text=clean_text,
    )

    return {
        "predicted_intent": classification["predicted_intent"],
        "confidence": classification["confidence"],
        "top_candidates": classification["top_candidates"],
        "retrieved_cases": retrieved_cases,
        "drafted_reply": reply,
        "decision": escalation["decision"],
        "escalation_reason": escalation["escalation_reason"],
    }


# ===========================================================================
# Pretty-Print Helper
# ===========================================================================
def print_agent_result(result: Dict[str, Any], message: str):
    """Pretty-print the agent's full result for a single message."""
    print("\n" + "=" * 80)
    print("AGENT RESULT")
    print("=" * 80)
    print(f"Customer Message: \"{message}\"")
    print("-" * 80)
    print(f"Predicted Intent:   {result['predicted_intent']}")
    print(f"Confidence:         {result['confidence']:.2%}")
    print(f"Decision:           {result['decision']}")
    print(f"Escalation Reason:  {result['escalation_reason']}")
    print("-" * 80)
    print(f"Drafted Reply:")
    print(f"  \"{result['drafted_reply']}\"")
    print("-" * 80)
    print(f"Top Candidates:")
    for c in result.get("top_candidates", []):
        print(f"  {c['intent']:30s}  {c['probability']:.2%}")
    print("-" * 80)
    print(f"Retrieved Cases ({len(result['retrieved_cases'])} total):")
    for i, case in enumerate(result["retrieved_cases"], 1):
        sim_pct = f"{case['similarity']:.2%}"
        print(f"  [{i}] Similarity: {sim_pct}")
        print(f"      Customer: \"{case['customer_message'][:120]}{'...' if len(case['customer_message']) > 120 else ''}\"")
        print(f"      Support:  \"{case['support_reply'][:120]}{'...' if len(case['support_reply']) > 120 else ''}\"")
    print("=" * 80)


# ===========================================================================
# Self-Test: 10 Example Messages from Historical Data (NOT golden set)
# ===========================================================================
def run_self_test():
    """
    Tests the agent with 10 example customer messages drawn from
    historical AppleSupport data — NOT from the golden set.
    """
    print("\n" + "#" * 80)
    print("# SELF-TEST: 10 HISTORICAL CUSTOMER MESSAGES (NOT FROM GOLDEN SET)")
    print("#" * 80)

    test_messages = [
        # 1. battery_drain_charging
        "My iPhone 7 battery drains from 100% to 20% in less than 3 hours since the iOS 11 update. This is ridiculous!",
        # 2. keyboard_text_glitch
        "Every time I type the letter I it changes to A [?] on my iPhone. This autocorrect bug is so annoying!",
        # 3. network_wifi_cellular
        "My iPhone keeps dropping WiFi connection every few minutes. I've tried resetting network settings but it keeps happening.",
        # 4. os_update_installation
        "I've been trying to update to iOS 11.2 for 3 days but it keeps failing. Says unable to install update.",
        # 5. bluetooth_airpods_carplay
        "My AirPods keep disconnecting from my iPhone every 5 minutes. I've tried re-pairing them multiple times.",
        # 6. apple_id_account_access (should ESCALATE)
        "I forgot my Apple ID password and now my account is locked. I can't access anything on my phone!",
        # 7. hardware_buttons_audio (should ESCALATE — physical damage)
        "The home button on my iPhone 7 is completely broken and doesn't respond to touch anymore.",
        # 8. billing_subscription_refund
        "I was charged $14.99 for an app subscription I never signed up for. How do I get a refund?",
        # 9. screen_touch_freeze
        "My iPhone screen is completely frozen and won't respond to touch. Force restart doesn't work either.",
        # 10. other_general_inquiry (should ESCALATE — out of scope)
        "Hey Apple, just wanted to say your customer service team is amazing. Keep up the great work!",
    ]

    results_summary = []
    for i, msg in enumerate(test_messages, 1):
        print(f"\n--- Test Case {i}/10 ---")
        result = run_agent(msg)
        print_agent_result(result, msg)
        results_summary.append({
            "test_case": i,
            "predicted_intent": result["predicted_intent"],
            "confidence": f"{result['confidence']:.2%}",
            "decision": result["decision"],
        })

    print("\n" + "=" * 80)
    print("SELF-TEST SUMMARY")
    print("=" * 80)
    print(f"{'#':<4} {'Predicted Intent':<32} {'Confidence':<12} {'Decision':<14}")
    print("-" * 62)
    for r in results_summary:
        print(f"{r['test_case']:<4} {r['predicted_intent']:<32} {r['confidence']:<12} {r['decision']:<14}")
    print("=" * 80)


# ===========================================================================
# Interactive CLI
# ===========================================================================
def interactive_cli():
    """
    Simple command-line interface for the support agent.
    Enter a customer message and see the full agent pipeline result.
    Type 'quit' or 'exit' to stop.
    """
    print("\n" + "=" * 80)
    print("APPLE SUPPORT AI AGENT - INTERACTIVE MODE")
    print("=" * 80)
    print("Enter a customer message to see the agent's response.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            message = input("Customer> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not message:
            continue
        if message.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        result = run_agent(message)
        print_agent_result(result, message)


# ===========================================================================
# Main Entry Point
# ===========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AppleSupport AI Agent")
    parser.add_argument("--test", action="store_true", help="Run self-test with 10 example messages")
    parser.add_argument("--build-index", action="store_true", help="Force rebuild retrieval index")
    parser.add_argument("--message", type=str, default=None, help="Process a single message")
    args = parser.parse_args()

    if args.build_index:
        idx = RetrievalIndex()
        idx.build()
        idx.save()
        print("Retrieval index built and saved.")
    elif args.test:
        run_self_test()
    elif args.message:
        result = run_agent(args.message)
        print_agent_result(result, args.message)
    else:
        # Default: run self-test first, then interactive mode
        run_self_test()
        interactive_cli()
