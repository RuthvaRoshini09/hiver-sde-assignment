"""
src/intent_classifier.py
------------------------
Baseline 1 for AppleSupport Intent Classification using TF-IDF + Logistic Regression.

Key Capabilities:
1. Loads historical AppleSupport interactions from data/processed/applesupport_pairs.csv.
2. Strictly EXCLUDES all 200 Golden Set tweets (golden_set_verified.csv is evaluation-only).
3. Labels the historical training pool using the approved 10 core intents + other_general_inquiry.
4. Performs a reproducible 80/20 train/test split.
5. Trains a TF-IDF (1-2 ngrams) + balanced Logistic Regression model.
6. Evaluates and reports:
   - Accuracy, Macro F1, Weighted F1
   - Per-class Precision, Recall, F1, Support
   - Confusion Matrix
   - Evaluation on held-out test split AND on the 200-row human-verified golden set
7. Serializes the trained pipeline under data/processed/baseline1_tfidf_logreg.joblib.
8. Provides a standalone `predict_intent(text)` function for real-time inference.
"""

import os
import sys
from typing import Dict, Any, Optional

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from src.data_cleaning import clean_tweet_text
from src.intent_discovery import (
    CORE_INTENT_DEFINITIONS,
    compile_regex_matchers,
    classify_intent_rule
)

PAIRS_PATH = os.path.join("data", "processed", "applesupport_pairs.csv")
GOLDEN_SET_PATH = os.path.join("data", "processed", "golden_set_verified.csv")
MODEL_OUTPUT_PATH = os.path.join("data", "processed", "baseline1_tfidf_logreg.joblib")

LABEL_SPACE = sorted(list(CORE_INTENT_DEFINITIONS.keys()) + ["other_general_inquiry"])

# Cache for loaded model artifacts during inference
_LOADED_PIPELINE: Optional[Dict[str, Any]] = None


def load_training_data():
    """
    Loads initial usable customer inquiries from historical AppleSupport data,
    STRICTLY EXCLUDING all 200 Golden Set tweets to prevent data leakage.
    """
    print("=" * 80)
    print("LOADING AND PREPARING HISTORICAL TRAINING DATA")
    print("=" * 80)

    if not os.path.exists(PAIRS_PATH):
        raise FileNotFoundError(f"Pairs file not found at {PAIRS_PATH}")
    if not os.path.exists(GOLDEN_SET_PATH):
        raise FileNotFoundError(f"Golden set file not found at {GOLDEN_SET_PATH}")

    # Load Golden Set tweet IDs to ensure strict exclusion
    df_golden = pd.read_csv(GOLDEN_SET_PATH)
    golden_tweet_ids = set(df_golden["tweet_id"].astype(int))
    print(f"Golden Set loaded: {len(df_golden)} examples (held out strictly for evaluation)")

    # Load historical AppleSupport pairs
    df_pairs = pd.read_csv(PAIRS_PATH)
    print(f"Historical pairs loaded: {len(df_pairs):,d}")

    # Filter for initial usable customer inquiries
    mask_initial = (df_pairs["is_initial_turn"] == True) & (df_pairs["is_usable"] == True)
    df_initial = df_pairs[mask_initial].copy()
    print(f"Initial usable customer inquiries: {len(df_initial):,d}")

    # STRICT EXCLUSION: drop any tweet that appears in the golden set
    df_clean_pool = df_initial[~df_initial["customer_tweet_id"].isin(golden_tweet_ids)].copy()
    excluded_count = len(df_initial) - len(df_clean_pool)
    print(f"Strictly excluded {excluded_count} Golden Set tweets from the training pool.")
    print(f"Final training pool size: {len(df_clean_pool):,d} customer inquiries.")

    # Assign target labels using approved domain taxonomy rules
    print("\nAssigning labels across the approved 11-class taxonomy...")
    matchers = compile_regex_matchers()
    df_clean_pool["intent"] = df_clean_pool["customer_text_clean"].apply(
        lambda t: classify_intent_rule(t, matchers)
    )

    print("\nTraining Pool Label Distribution:")
    counts = df_clean_pool["intent"].value_counts()
    for intent, count in counts.items():
        pct = (count / len(df_clean_pool)) * 100
        print(f"  {intent:30s}: {count:6,d} ({pct:5.2f}%)")

    return df_clean_pool, df_golden


def train_baseline_model(df_train_pool: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
    """
    Trains TF-IDF + Logistic Regression on the training pool with an 80/20 train/test split.
    """
    print("\n" + "=" * 80)
    print("TRAINING BASELINE 1: TF-IDF + LOGISTIC REGRESSION")
    print("=" * 80)

    X = [str(t) for t in df_train_pool["customer_text_clean"]]
    y = [str(l) for l in df_train_pool["intent"]]

    # Reproducible stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"Train samples: {len(X_train):,d} ({(1 - test_size) * 100:.0f}%)")
    print(f"Test samples:  {len(X_test):,d} ({test_size * 100:.0f}%)")

    # TF-IDF Feature Extraction
    print("\nExtracting TF-IDF features (unigrams + bigrams, max_features=25000)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=25000,
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_):,d} features")

    # Logistic Regression Classifier with balanced class weights
    print("\nTraining Logistic Regression (class_weight='balanced', max_iter=1000)...")
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs"
    )

    classifier.fit(X_train_tfidf, y_train)
    print("Model training complete.")

    return {
        "vectorizer": vectorizer,
        "classifier": classifier,
        "classes": classifier.classes_,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_test_tfidf": X_test_tfidf
    }


def evaluate_dataset(y_true, y_pred, labels, title: str):
    """
    Evaluates predictions and prints accuracy, macro F1, weighted F1,
    classification report, and confusion matrix.
    """
    print("\n" + "=" * 80)
    print(f"EVALUATION: {title.upper()}")
    print("=" * 80)

    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    print(f"Total Samples: {len(y_true):,d}")
    print(f"Accuracy:      {acc * 100:6.2f}%")
    print(f"Macro F1:      {f1_macro * 100:6.2f}% (Precision: {p_macro * 100:.2f}%, Recall: {r_macro * 100:.2f}%)")
    print(f"Weighted F1:   {f1_weighted * 100:6.2f}% (Precision: {p_weighted * 100:.2f}%, Recall: {r_weighted * 100:.2f}%)")

    print("\nPer-Class Detailed Report:")
    print("-" * 80)
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0, digits=4))

    print("Confusion Matrix:")
    print("-" * 80)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    print(df_cm.to_string())

    return {
        "accuracy": acc,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "confusion_matrix": df_cm
    }


def save_model(vectorizer, classifier, output_path: str = MODEL_OUTPUT_PATH):
    """
    Serializes the vectorizer and classifier into a joblib artifact.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = {
        "vectorizer": vectorizer,
        "classifier": classifier,
        "classes": classifier.classes_,
        "model_type": "TF-IDF + Logistic Regression",
        "label_space": list(classifier.classes_)
    }
    joblib.dump(payload, output_path)
    print(f"\nSaved trained model artifact to: {output_path}")


def predict_intent(text: str, model_path: str = MODEL_OUTPUT_PATH, return_top_k: int = 3) -> Dict[str, Any]:
    """
    Inference function for classifying a new customer message.

    Args:
        text: Raw customer message.
        model_path: Path to serialized model artifact.
        return_top_k: Number of top candidate probabilities to return.

    Returns:
        Dict containing clean_text, predicted_intent, confidence, and top_k probabilities.
    """
    global _LOADED_PIPELINE

    if _LOADED_PIPELINE is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
        _LOADED_PIPELINE = joblib.load(model_path)

    vectorizer = _LOADED_PIPELINE["vectorizer"]
    classifier = _LOADED_PIPELINE["classifier"]
    classes = _LOADED_PIPELINE["classes"]

    clean_msg = clean_tweet_text(text)
    if not clean_msg:
        return {
            "raw_text": text,
            "clean_text": "",
            "predicted_intent": "other_general_inquiry",
            "confidence": 0.0,
            "top_candidates": []
        }

    X_tfidf = vectorizer.transform([clean_msg])
    probs = classifier.predict_proba(X_tfidf)[0]

    # Sort probabilities
    ranked_indices = np.argsort(probs)[::-1]
    top_intent = classes[ranked_indices[0]]
    confidence = float(probs[ranked_indices[0]])

    top_candidates = [
        {"intent": classes[i], "probability": round(float(probs[i]), 4)}
        for i in ranked_indices[:return_top_k]
    ]

    return {
        "raw_text": text,
        "clean_text": clean_msg,
        "predicted_intent": top_intent,
        "confidence": round(confidence, 4),
        "top_candidates": top_candidates
    }


def main():
    # 1. Load training data and strictly exclude golden set
    df_train_pool, df_golden = load_training_data()

    # 2. Train baseline model with 80/20 train/test split
    pipeline = train_baseline_model(df_train_pool, test_size=0.20, random_state=42)

    # 3. Evaluate on held-out test split (15,959 samples)
    y_test_pred = pipeline["classifier"].predict(pipeline["X_test_tfidf"])
    test_metrics = evaluate_dataset(
        y_true=pipeline["y_test"],
        y_pred=y_test_pred,
        labels=pipeline["classes"],
        title="Held-Out Test Split (20% Historical Data, 15,959 Samples)"
    )

    # 4. Evaluate on Human-Verified Golden Evaluation Set (200 samples)
    print("\n" + "=" * 80)
    print("EVALUATING ON HUMAN-VERIFIED GOLDEN EVALUATION SET (200 SAMPLES)")
    print("=" * 80)

    X_golden = [str(m) for m in df_golden["clean_message"]]
    y_golden_true = [str(l) for l in df_golden["reviewer_label"]]

    X_golden_tfidf = pipeline["vectorizer"].transform(X_golden)
    y_golden_pred = pipeline["classifier"].predict(X_golden_tfidf)

    golden_metrics = evaluate_dataset(
        y_true=y_golden_true,
        y_pred=y_golden_pred,
        labels=pipeline["classes"],
        title="Human-Verified Golden Evaluation Set (Evaluation-Only, 200 Samples)"
    )

    # 5. Save model artifact
    save_model(pipeline["vectorizer"], pipeline["classifier"], MODEL_OUTPUT_PATH)

    # 6. Verify inference function on sample inputs
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS USING predict_intent()")
    print("=" * 80)

    sample_queries = [
        "My iPhone 7 battery is draining from 100% to 15% in less than an hour since the update! Please fix this.",
        "Why does typing capital letter I turn into a weird A and question mark symbol on my keyboard???",
        "I was charged $9.99 on my credit card for an iTunes subscription I never signed up for. Need a refund ASAP.",
        "Ever since iOS 11 my screen is completely frozen and unresponsive to touch. Force restart didn't help.",
        "My AirPods keep cutting out and disconnecting from Bluetooth while playing music.",
        "I forgot my Apple ID password and my account is locked. How can I reset it?",
        "When will my iPhone X preorder ship? The order status says December.",
        "Thanks so much for the quick help earlier today, you guys are awesome!"
    ]

    for q in sample_queries:
        res = predict_intent(q)
        print(f"\nInput Query:      \"{res['raw_text']}\"")
        print(f"Predicted Intent: {res['predicted_intent']} (Confidence: {res['confidence'] * 100:.2f}%)")
        print(f"Top Candidates:   {', '.join([f'{c['intent']} ({c['probability'] * 100:.1f}%)' for c in res['top_candidates']])}")

    print("\n" + "=" * 80)
    print("BASELINE 1 IMPLEMENTATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
