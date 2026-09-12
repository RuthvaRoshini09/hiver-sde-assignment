"""
src/conversation_builder.py
---------------------------
Reconstructs AppleSupport conversations and customer -> support pairs
from data/twcs.csv using a chunk-based, memory-safe approach.

Outputs:
1. data/processed/applesupport_conversations.csv: Thread-level multi-turn dataset.
2. data/processed/applesupport_pairs.csv: Interaction-level pair dataset.
"""

import os
import sys
from datetime import datetime

# Add project root to sys.path to allow direct execution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np

# Import cleaning utilities
from src.data_cleaning import (
    clean_tweet_text,
    is_usable_customer_message,
    is_usable_support_reply,
    detect_dm_deflection,
    detect_troubleshooting
)

DATA_PATH = "data/twcs.csv"
OUTPUT_DIR = "data/processed"
CONVERSATIONS_OUTPUT = os.path.join(OUTPUT_DIR, "applesupport_conversations.csv")
PAIRS_OUTPUT = os.path.join(OUTPUT_DIR, "applesupport_pairs.csv")

TARGET_BRAND = "AppleSupport"
CHUNK_SIZE = 100000


def parse_twitter_date(date_str: str):
    """Safely parse Twitter timestamp format."""
    try:
        # e.g., "Tue Oct 31 22:10:47 +0000 2017"
        return datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")
    except Exception:
        return datetime.min


def build_applesupport_dataset():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("=" * 75)
    print("BUILDING APPLESUPPORT DATASET & RECONSTRUCTING CONVERSATIONS")
    print("=" * 75)
    print(f"Source: {DATA_PATH}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Chunk Size: {CHUNK_SIZE:,}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -----------------------------------------------------------------
    # PASS 1: Extract all AppleSupport tweets and identify parent IDs
    # -----------------------------------------------------------------
    print("[PASS 1/2] Streaming dataset for AppleSupport replies...")

    # Dictionary: tweet_id -> dict
    apple_tweets = {}
    # Set of customer tweet IDs that received an AppleSupport reply
    parent_customer_ids = set()
    apple_tweet_ids = set()

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

        # Filter for AppleSupport tweets
        mask = (chunk["author_id"] == TARGET_BRAND) & (chunk["inbound"] == False)
        brand_chunk = chunk[mask]

        if not brand_chunk.empty:
            for row in brand_chunk.itertuples(index=False):
                t_id = row.tweet_id
                apple_tweet_ids.add(t_id)

                p_id = row.in_response_to_tweet_id
                has_parent = not pd.isna(p_id)
                parent_int = int(p_id) if has_parent else None

                if parent_int is not None:
                    parent_customer_ids.add(parent_int)

                apple_tweets[t_id] = {
                    "tweet_id": t_id,
                    "author_id": row.author_id,
                    "role": "support",
                    "inbound": False,
                    "created_at": row.created_at,
                    "text": row.text if isinstance(row.text, str) else "",
                    "in_response_to_tweet_id": parent_int,
                    "response_tweet_id": row.response_tweet_id if isinstance(row.response_tweet_id, str) else None
                }

        if (chunk_idx + 1) % 5 == 0:
            print(f"  Processed {total_rows:,} rows... Found {len(apple_tweets):,d} AppleSupport tweets.")

    print(f"Finished Pass 1. Total AppleSupport tweets: {len(apple_tweets):,d}")
    print(f"Total parent customer tweets to locate: {len(parent_customer_ids):,d}\n")

    # -----------------------------------------------------------------
    # PASS 2: Extract connected customer inbound tweets
    # -----------------------------------------------------------------
    print("[PASS 2/2] Streaming dataset for connected customer tweets...")

    # Dictionary: tweet_id -> dict
    customer_tweets = {}
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

        inbound_chunk = chunk[chunk["inbound"] == True]
        if inbound_chunk.empty:
            continue

        # Inbound tweets that either:
        # 1. are parent tweets replied to by AppleSupport
        # 2. are follow-up replies to an AppleSupport tweet
        inbound_pids = inbound_chunk["in_response_to_tweet_id"]
        mask = (
            inbound_chunk["tweet_id"].isin(parent_customer_ids) |
            inbound_pids.isin(apple_tweet_ids)
        )

        matched_chunk = inbound_chunk[mask]
        if not matched_chunk.empty:
            for row in matched_chunk.itertuples(index=False):
                t_id = row.tweet_id
                p_id = row.in_response_to_tweet_id
                has_parent = not pd.isna(p_id)
                parent_int = int(p_id) if has_parent else None

                customer_tweets[t_id] = {
                    "tweet_id": t_id,
                    "author_id": str(row.author_id),
                    "role": "customer",
                    "inbound": True,
                    "created_at": row.created_at,
                    "text": row.text if isinstance(row.text, str) else "",
                    "in_response_to_tweet_id": parent_int,
                    "response_tweet_id": row.response_tweet_id if isinstance(row.response_tweet_id, str) else None
                }

        if (chunk_idx + 1) % 5 == 0:
            print(f"  Processed {total_rows:,} rows... Captured {len(customer_tweets):,d} connected customer tweets.")

    print(f"Finished Pass 2. Total connected customer tweets: {len(customer_tweets):,d}\n")

    # -----------------------------------------------------------------
    # RECONSTRUCT CONVERSATION THREADS & FIND ROOTS
    # -----------------------------------------------------------------
    print("Reconstructing conversation threads...")

    all_tweets = {}
    all_tweets.update(apple_tweets)
    all_tweets.update(customer_tweets)

    # Memoized root finder
    memo_roots = {}

    def find_root(t_id: int, visited: set) -> int:
        if t_id in memo_roots:
            return memo_roots[t_id]
        if t_id in visited:
            return t_id  # cycle protection
        visited.add(t_id)

        t_info = all_tweets.get(t_id)
        if not t_info:
            return t_id

        parent_id = t_info.get("in_response_to_tweet_id")
        if parent_id is None or parent_id not in all_tweets:
            memo_roots[t_id] = t_id
            return t_id

        root = find_root(parent_id, visited)
        memo_roots[t_id] = root
        return root

    # Group all tweets by root conversation_id
    conversation_groups = {}
    for t_id in all_tweets:
        root_id = find_root(t_id, set())
        if root_id not in conversation_groups:
            conversation_groups[root_id] = []
        conversation_groups[root_id].append(all_tweets[t_id])

    print(f"Identified {len(conversation_groups):,d} total conversation clusters.\n")

    # -----------------------------------------------------------------
    # BUILD DATASET 1: applesupport_conversations.csv
    # -----------------------------------------------------------------
    print("Generating applesupport_conversations.csv...")

    conversation_rows = []
    multi_turn_count = 0

    for root_id, tweets in conversation_groups.items():
        # Sort chronologically
        # To avoid slow string parsing for all 200k, sort by tweet_id (which is monotonic)
        # or parse dates. In Twitter dataset, tweet_id order matches chronological order.
        sorted_tweets = sorted(tweets, key=lambda x: x["tweet_id"])

        cust_tweets = [t for t in sorted_tweets if t["role"] == "customer"]
        supp_tweets = [t for t in sorted_tweets if t["role"] == "support"]

        # Only retain valid conversations that have at least one customer and one support tweet
        if not cust_tweets or not supp_tweets:
            continue

        num_turns = len(sorted_tweets)
        is_multi = (num_turns > 2)
        if is_multi:
            multi_turn_count += 1

        first_cust = cust_tweets[0]
        first_supp = supp_tweets[0]

        # Build clean dialogue transcript
        transcript_lines = []
        for t in sorted_tweets:
            prefix = "[Customer]" if t["role"] == "customer" else "[AppleSupport]"
            clean_msg = clean_tweet_text(t["text"])
            transcript_lines.append(f"{prefix}: {clean_msg}")
        transcript = "\n".join(transcript_lines)

        tweet_chain = ",".join(str(t["tweet_id"]) for t in sorted_tweets)

        conversation_rows.append({
            "conversation_id": root_id,
            "customer_id": first_cust["author_id"],
            "created_at_start": sorted_tweets[0]["created_at"],
            "created_at_end": sorted_tweets[-1]["created_at"],
            "num_turns": num_turns,
            "num_customer_messages": len(cust_tweets),
            "num_support_replies": len(supp_tweets),
            "is_multi_turn": is_multi,
            "initial_customer_tweet_id": first_cust["tweet_id"],
            "initial_customer_text": clean_tweet_text(first_cust["text"]),
            "first_support_tweet_id": first_supp["tweet_id"],
            "first_support_text": clean_tweet_text(first_supp["text"]),
            "conversation_transcript": transcript,
            "tweet_ids_chain": tweet_chain
        })

    df_convs = pd.DataFrame(conversation_rows)
    df_convs.to_csv(CONVERSATIONS_OUTPUT, index=False)
    print(f"Saved {len(df_convs):,d} conversations to {CONVERSATIONS_OUTPUT}")

    # -----------------------------------------------------------------
    # BUILD DATASET 2: applesupport_pairs.csv
    # -----------------------------------------------------------------
    print("\nGenerating applesupport_pairs.csv...")

    pair_rows = []
    usable_pairs_count = 0
    pair_idx = 1

    # For each AppleSupport tweet, find its direct parent customer tweet
    for supp_id, s_info in apple_tweets.items():
        parent_id = s_info["in_response_to_tweet_id"]
        if parent_id is None or parent_id not in customer_tweets:
            continue

        c_info = customer_tweets[parent_id]

        c_raw = c_info["text"]
        c_clean = clean_tweet_text(c_raw)
        s_raw = s_info["text"]
        s_clean = clean_tweet_text(s_raw)

        conv_id = memo_roots.get(parent_id, parent_id)
        is_initial = (parent_id == conv_id)

        conv_len = len(conversation_groups.get(conv_id, []))
        is_multi = (conv_len > 2)

        is_dm = detect_dm_deflection(s_raw)
        has_res = detect_troubleshooting(s_raw)

        usable = is_usable_customer_message(c_raw) and is_usable_support_reply(s_raw)
        if usable:
            usable_pairs_count += 1

        pair_rows.append({
            "pair_id": f"asp_{pair_idx:06d}",
            "conversation_id": conv_id,
            "customer_tweet_id": c_info["tweet_id"],
            "customer_author_id": c_info["author_id"],
            "customer_created_at": c_info["created_at"],
            "customer_text_raw": c_raw,
            "customer_text_clean": c_clean,
            "support_tweet_id": s_info["tweet_id"],
            "support_author_id": TARGET_BRAND,
            "support_created_at": s_info["created_at"],
            "support_text_raw": s_raw,
            "support_text_clean": s_clean,
            "is_initial_turn": is_initial,
            "is_multi_turn": is_multi,
            "is_usable": usable,
            "is_dm_deflection": is_dm,
            "has_troubleshooting": has_res
        })
        pair_idx += 1

    df_pairs = pd.DataFrame(pair_rows)
    df_pairs.to_csv(PAIRS_OUTPUT, index=False)
    print(f"Saved {len(df_pairs):,d} customer -> AppleSupport pairs to {PAIRS_OUTPUT}\n")

    # -----------------------------------------------------------------
    # METRIC SUMMARY & 10 EXAMPLES
    # -----------------------------------------------------------------
    num_conversations = len(df_convs)
    num_customer_messages = len(customer_tweets)
    num_support_replies = len(apple_tweets)
    total_pairs = len(df_pairs)

    print("=" * 75)
    print("DATASET CONSTRUCTION SUMMARY")
    print("=" * 75)
    print(f"1. Number of AppleSupport conversations:      {num_conversations:,d}")
    print(f"2. Number of connected customer messages:    {num_customer_messages:,d}")
    print(f"3. Number of AppleSupport replies:           {num_support_replies:,d}")
    print(f"4. Total matched response pairs:             {total_pairs:,d}")
    print(f"   Usable customer -> brand pairs:           {usable_pairs_count:,d} ({usable_pairs_count/total_pairs*100:.2f}%)")
    print(f"5. Number of multi-turn conversations:       {multi_turn_count:,d} ({multi_turn_count/num_conversations*100:.2f}% of conversations)")
    print("=" * 75)

    # Sample 10 diverse, real examples: mix of troubleshooting, diagnostic, multi-turn, and initial
    print("\n" + "=" * 75)
    print("10 REAL CUSTOMER -> APPLESUPPORT EXAMPLES")
    print("=" * 75)

    # Pick 10 representative pairs: 5 with troubleshooting, 5 standard/multi-turn
    sample_res = df_pairs[df_pairs["has_troubleshooting"] & ~df_pairs["is_dm_deflection"] & df_pairs["is_usable"]].head(5)
    res_ids = set(sample_res["pair_id"])
    sample_multi = df_pairs[
        df_pairs["is_multi_turn"] & 
        df_pairs["is_usable"] & 
        ~df_pairs["pair_id"].isin(res_ids)
    ].iloc[10:15]

    sample_combined = pd.concat([sample_res, sample_multi]).reset_index(drop=True)

    for idx, row in sample_combined.iterrows():
        print(f"\n--- [Example {idx + 1}] (Pair ID: {row['pair_id']} | Conv ID: {row['conversation_id']}) ---")
        print(f"Customer Tweet ID: {row['customer_tweet_id']} | Created: {row['customer_created_at']}")
        print(f"Customer Text:     {row['customer_text_clean']}")
        print(f"AppleSupport ID:   {row['support_tweet_id']} | Created: {row['support_created_at']}")
        print(f"AppleSupport Text: {row['support_text_clean']}")
        print(f"Attributes:        Multi-turn={row['is_multi_turn']}, Initial={row['is_initial_turn']}, Troubleshooting={row['has_troubleshooting']}, DM-Deflection={row['is_dm_deflection']}")


if __name__ == "__main__":
    build_applesupport_dataset()
