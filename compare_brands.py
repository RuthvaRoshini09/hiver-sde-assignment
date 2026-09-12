"""
compare_brands.py
-----------------
Memory-safe, chunk-based comparison analysis of candidate brands in twcs.csv.

Candidate Brands:
- AmazonHelp
- AppleSupport
- Uber_Support
- SpotifyCares
- Delta

Evaluates:
1. Number of brand/support tweets
2. Number of customer tweets directly connected to brand tweets
3. Number of usable customer messages
4. Number of customer -> brand response pairs
5. Conversation completeness
6. Number of conversations with meaningful text on both sides
7. Examples of real customer messages
8. Examples of corresponding brand responses
9. Repeated customer issues that could become intents
10. Useful resolution information vs. canned DM deflection in historical responses
"""

import sys
import re
import gc
from collections import Counter
import pandas as pd
import numpy as np

DATA_PATH = "data/twcs.csv"
CHUNK_SIZE = 100000

TARGET_BRANDS = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "Delta"
]

TARGET_SET = set(TARGET_BRANDS)

# Regex helpers for text cleaning
RE_MENTION = re.compile(r"@[A-Za-z0-9_]+")
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_WHITESPACE = re.compile(r"\s+")

DM_PATTERNS = [
    r"\bdm\b",
    r"\bdms\b",
    r"\bdirect message\b",
    r"\bprivate message\b",
    r"\bpm\b",
    r"send us a message",
    r"reach out via dm",
    r"shoot us a dm",
    r"send a dm",
    r"send me a dm",
    r"send us a private",
    r"send me a private",
    r"click the link",
    r"click here to chat",
    r"follow and dm",
    r"follow us and dm"
]
RE_DM = re.compile("|".join(DM_PATTERNS), re.IGNORECASE)

RESOLUTION_PATTERNS = [
    r"\btry\b",
    r"\brestart\b",
    r"\breinstall\b",
    r"\bsettings\b",
    r"\bsteps\b",
    r"\bupdate\b",
    r"\breset\b",
    r"\bclear cache\b",
    r"\bhelp center\b",
    r"\barticle\b",
    r"\bguide\b",
    r"\bcheck if\b",
    r"\bmake sure\b",
    r"\bturn off\b",
    r"\bturn on\b",
    r"\bforce close\b",
    r"\bgo to\b",
    r"\btap on\b",
    r"\bclick on\b",
    r"\bversion\b"
]
RE_RESOLUTION = re.compile("|".join(RESOLUTION_PATTERNS), re.IGNORECASE)


def clean_text(text: str) -> str:
    """Strip mentions, URLs, and excessive whitespace to evaluate true content."""
    if not isinstance(text, str):
        return ""
    t = RE_MENTION.sub("", text)
    t = RE_URL.sub("", t)
    t = RE_WHITESPACE.sub(" ", t).strip()
    return t


def is_usable_customer_message(clean_msg: str) -> bool:
    """Determines if a customer message has sufficient substance."""
    if len(clean_msg) < 15:
        return False
    words = clean_msg.split()
    return len(words) >= 3


def is_meaningful_brand_response(clean_msg: str) -> bool:
    """Determines if a brand response has meaningful substance."""
    if len(clean_msg) < 20:
        return False
    words = clean_msg.split()
    return len(words) >= 4


def extract_keywords(texts, top_n=20):
    """Extract informative keywords/n-grams to identify candidate intents."""
    stopwords = {
        "the", "to", "and", "a", "i", "my", "is", "in", "it", "for", "of", "on", "you",
        "this", "with", "have", "that", "me", "so", "be", "but", "are", "at", "not",
        "was", "can", "your", "get", "we", "just", "from", "do", "like", "an", "if",
        "up", "out", "when", "all", "about", "no", "how", "what", "has", "as", "or",
        "will", "by", "there", "they", "been", "one", "would", "am", "more", "now",
        "us", "had", "he", "she", "his", "her", "its", "their", "our", "were", "which",
        "who", "whom", "into", "than", "then", "them", "some", "could", "also", "any",
        "why", "did", "please", "thanks", "thank", "hi", "hello", "hey", "help",
        "amazon", "apple", "uber", "spotify", "delta", "support", "amazonhelp",
        "applesupport", "uber_support", "spotifycares"
    }
    words = []
    for t in texts:
        t_clean = re.findall(r"\b[a-z]{3,15}\b", t.lower())
        words.extend([w for w in t_clean if w not in stopwords])
    return Counter(words).most_common(top_n)


def main():
    print("=" * 70)
    print("STARTING BRAND COMPARISON ANALYSIS")
    print("=" * 70)
    print(f"Target Brands: {', '.join(TARGET_BRANDS)}")
    print(f"Dataset: {DATA_PATH}")
    print(f"Chunk Size: {CHUNK_SIZE:,}\n")

    # -------------------------------------------------------------
    # PASS 1: Extract all brand tweets for the 5 target brands
    # -------------------------------------------------------------
    print("[PASS 1/2] Scanning dataset for brand support tweets...")

    # Data structures to store brand tweets per brand
    # brand -> list of dicts: {'tweet_id', 'created_at', 'text', 'in_response_to_tweet_id'}
    brand_tweets = {b: [] for b in TARGET_BRANDS}
    # brand -> set of parent customer tweet IDs
    brand_parent_ids = {b: set() for b in TARGET_BRANDS}
    # brand -> set of brand tweet IDs
    brand_tweet_ids = {b: set() for b in TARGET_BRANDS}

    total_rows = 0

    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            DATA_PATH,
            usecols=[
                "tweet_id",
                "author_id",
                "inbound",
                "created_at",
                "text",
                "in_response_to_tweet_id",
                "response_tweet_id"
            ],
            dtype={
                "tweet_id": "int64",
                "author_id": "str",
                "inbound": "bool",
                "created_at": "str",
                "text": "str",
                "in_response_to_tweet_id": "float64",
                "response_tweet_id": "str"
            },
            chunksize=CHUNK_SIZE
        )
    ):
        total_rows += len(chunk)

        # Filter for our 5 target brands outbound tweets
        brand_mask = (chunk["inbound"] == False) & (chunk["author_id"].isin(TARGET_SET))
        target_chunk = chunk[brand_mask]

        if not target_chunk.empty:
            for row in target_chunk.itertuples(index=False):
                b = row.author_id
                t_id = row.tweet_id
                brand_tweet_ids[b].add(t_id)

                parent_id = row.in_response_to_tweet_id
                has_parent = not pd.isna(parent_id)
                parent_int = int(parent_id) if has_parent else None

                if parent_int is not None:
                    brand_parent_ids[b].add(parent_int)

                brand_tweets[b].append({
                    "tweet_id": t_id,
                    "created_at": row.created_at,
                    "text": row.text if isinstance(row.text, str) else "",
                    "in_response_to_tweet_id": parent_int
                })

        if (chunk_idx + 1) % 5 == 0:
            print(f"  Processed {total_rows:,} rows...")

    print(f"Finished Pass 1. Total rows read: {total_rows:,}\n")

    # Union of all parent tweet IDs to fetch in Pass 2
    all_parent_ids = set()
    for b in TARGET_BRANDS:
        all_parent_ids.update(brand_parent_ids[b])
        print(f"Brand: {b:15s} | Support Tweets: {len(brand_tweets[b]):,d} | Citing Parent Tweet: {len(brand_parent_ids[b]):,d}")

    # Also combine all brand tweet IDs to capture customer tweets replying TO brand tweets
    all_brand_tweet_ids = set()
    for b in TARGET_BRANDS:
        all_brand_tweet_ids.update(brand_tweet_ids[b])

    print(f"\nTotal unique customer tweet IDs to locate: {len(all_parent_ids):,d}")

    # -------------------------------------------------------------
    # PASS 2: Extract customer tweets corresponding to these brands
    # -------------------------------------------------------------
    print("\n[PASS 2/2] Scanning dataset for connected customer tweets...")

    # Maps customer tweet_id -> {'author_id', 'text', 'clean_text', 'created_at', 'in_response_to_tweet_id'}
    customer_tweets = {}
    # Count customer follow-ups replying to brand tweets
    customer_replies_to_brand = {b: 0 for b in TARGET_BRANDS}

    total_rows = 0

    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            DATA_PATH,
            usecols=[
                "tweet_id",
                "author_id",
                "inbound",
                "created_at",
                "text",
                "in_response_to_tweet_id"
            ],
            dtype={
                "tweet_id": "int64",
                "author_id": "str",
                "inbound": "bool",
                "created_at": "str",
                "text": "str",
                "in_response_to_tweet_id": "float64"
            },
            chunksize=CHUNK_SIZE
        )
    ):
        total_rows += len(chunk)

        # Only inbound customer tweets
        inbound_chunk = chunk[chunk["inbound"] == True]
        if inbound_chunk.empty:
            continue

        # Check if customer tweet is one of the referenced parents
        parent_matches = inbound_chunk[inbound_chunk["tweet_id"].isin(all_parent_ids)]
        if not parent_matches.empty:
            for row in parent_matches.itertuples(index=False):
                raw_txt = row.text if isinstance(row.text, str) else ""
                customer_tweets[row.tweet_id] = {
                    "author_id": row.author_id,
                    "text": raw_txt,
                    "clean_text": clean_text(raw_txt),
                    "created_at": row.created_at,
                    "in_response_to_tweet_id": int(row.in_response_to_tweet_id) if not pd.isna(row.in_response_to_tweet_id) else None
                }

        # Check if customer tweet was replying to a brand tweet
        reply_matches = inbound_chunk[inbound_chunk["in_response_to_tweet_id"].isin(all_brand_tweet_ids)]
        if not reply_matches.empty:
            for row in reply_matches.itertuples(index=False):
                p_id = int(row.in_response_to_tweet_id)
                for b in TARGET_BRANDS:
                    if p_id in brand_tweet_ids[b]:
                        customer_replies_to_brand[b] += 1
                        break

        if (chunk_idx + 1) % 5 == 0:
            print(f"  Processed {total_rows:,} rows... Found {len(customer_tweets):,d} customer parent tweets so far.")

    print(f"Finished Pass 2. Captured {len(customer_tweets):,d} customer parent tweets.\n")

    # -------------------------------------------------------------
    # METRICS EVALUATION PER BRAND
    # -------------------------------------------------------------
    print("=" * 70)
    print("ANALYZING METRICS & DATA QUALITY PER BRAND")
    print("=" * 70)

    summary_records = []
    brand_example_pairs = []

    for brand in TARGET_BRANDS:
        b_tweets = brand_tweets[brand]
        total_support_tweets = len(b_tweets)

        # 1. Total brand replies that reference a customer tweet
        replies_with_parent = [bt for bt in b_tweets if bt["in_response_to_tweet_id"] is not None]
        total_with_parent = len(replies_with_parent)

        # 2. Matched pairs where customer tweet exists in customer_tweets
        matched_pairs = []
        for bt in replies_with_parent:
            p_id = bt["in_response_to_tweet_id"]
            if p_id in customer_tweets:
                matched_pairs.append({
                    "brand": brand,
                    "customer_tweet_id": p_id,
                    "brand_tweet_id": bt["tweet_id"],
                    "customer_text": customer_tweets[p_id]["text"],
                    "customer_clean": customer_tweets[p_id]["clean_text"],
                    "brand_text": bt["text"],
                    "brand_clean": clean_text(bt["text"]),
                })

        num_matched_pairs = len(matched_pairs)

        # Customer tweets directly connected:
        # parent customer tweets found + customer follow-up replies to brand tweets
        # Unique customer tweet IDs connected
        unique_connected_cust_ids = set([p["customer_tweet_id"] for p in matched_pairs])
        total_connected_customer_tweets = len(unique_connected_cust_ids) + customer_replies_to_brand[brand]

        # 3. Usable customer messages
        usable_pairs = [p for p in matched_pairs if is_usable_customer_message(p["customer_clean"])]
        num_usable_customer_messages = len(usable_pairs)
        usable_pct = (num_usable_customer_messages / num_matched_pairs * 100) if num_matched_pairs > 0 else 0

        # 4. Customer -> Brand response pairs (matched_pairs count)
        num_pairs = num_matched_pairs

        # 5. Conversation completeness
        completeness_rate = (num_matched_pairs / total_with_parent * 100) if total_with_parent > 0 else 0

        # 6. Meaningful text on both sides
        meaningful_pairs = [
            p for p in usable_pairs
            if is_meaningful_brand_response(p["brand_clean"])
        ]
        num_meaningful_pairs = len(meaningful_pairs)
        meaningful_pct = (num_meaningful_pairs / num_matched_pairs * 100) if num_matched_pairs > 0 else 0

        # 7 & 8: DM Deflection vs. Resolution Information
        dm_deflection_count = 0
        resolution_info_count = 0

        for p in matched_pairs:
            b_text = p["brand_text"]
            is_dm = bool(RE_DM.search(b_text))
            is_res = bool(RE_RESOLUTION.search(b_text))

            if is_dm:
                dm_deflection_count += 1
            if is_res and not is_dm:
                resolution_info_count += 1

        dm_deflection_rate = (dm_deflection_count / num_matched_pairs * 100) if num_matched_pairs > 0 else 0
        direct_resolution_rate = (resolution_info_count / num_matched_pairs * 100) if num_matched_pairs > 0 else 0

        # 9. Topic/Intent Keywords
        all_cust_clean = [p["customer_clean"] for p in usable_pairs]
        top_keywords = extract_keywords(all_cust_clean, top_n=12)
        top_keywords_str = ", ".join([f"{k} ({c})" for k, c in top_keywords[:8]])

        # Store Summary Record
        summary_records.append({
            "brand": brand,
            "support_tweets": total_support_tweets,
            "connected_customer_tweets": total_connected_customer_tweets,
            "response_pairs": num_pairs,
            "completeness_pct": round(completeness_rate, 2),
            "usable_customer_msgs": num_usable_customer_messages,
            "usable_customer_pct": round(usable_pct, 2),
            "meaningful_pairs": num_meaningful_pairs,
            "meaningful_pairs_pct": round(meaningful_pct, 2),
            "dm_deflection_pct": round(dm_deflection_rate, 2),
            "direct_resolution_pct": round(direct_resolution_rate, 2),
            "top_customer_keywords": top_keywords_str
        })

        # Select 5 representative diverse pairs for brand_example_pairs.csv
        # Prioritize meaningful pairs with different keyword topics
        sample_candidates = meaningful_pairs if len(meaningful_pairs) >= 5 else matched_pairs
        step = max(1, len(sample_candidates) // 8)
        selected_samples = sample_candidates[::step][:5]

        for s in selected_samples:
            # Determine issue snippet
            brand_example_pairs.append({
                "brand": brand,
                "customer_tweet_id": s["customer_tweet_id"],
                "customer_text": s["customer_text"],
                "customer_text_clean": s["customer_clean"],
                "brand_tweet_id": s["brand_tweet_id"],
                "brand_text": s["brand_text"],
                "brand_text_clean": s["brand_clean"],
                "is_dm_deflection": bool(RE_DM.search(s["brand_text"])),
                "has_resolution_steps": bool(RE_RESOLUTION.search(s["brand_text"]))
            })

    # Convert to DataFrames
    df_summary = pd.DataFrame(summary_records)
    df_examples = pd.DataFrame(brand_example_pairs)

    # Save outputs
    df_summary.to_csv("candidate_brands.csv", index=False)
    df_examples.to_csv("brand_example_pairs.csv", index=False)

    print("\nSaved summary to: candidate_brands.csv")
    print("Saved example pairs to: brand_example_pairs.csv\n")

    print("=" * 100)
    print("CANDIDATE BRAND COMPARISON SUMMARY")
    print("=" * 100)
    display_cols = [
        "brand",
        "support_tweets",
        "response_pairs",
        "completeness_pct",
        "usable_customer_msgs",
        "meaningful_pairs",
        "dm_deflection_pct",
        "direct_resolution_pct"
    ]
    print(df_summary[display_cols].to_string(index=False))

    print("\n" + "=" * 100)
    print("TOP CUSTOMER ISSUE KEYWORDS BY BRAND")
    print("=" * 100)
    for row in summary_records:
        print(f"[{row['brand']}]")
        print(f"  Keywords: {row['top_customer_keywords']}\n")

    print("=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
