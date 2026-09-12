import pandas as pd

FILE_PATH = "data/twcs.csv"

print("Analyzing the full dataset...")
print("This may take a few minutes.\n")

chunk_size = 100000

brand_stats = {}

for chunk in pd.read_csv(
    FILE_PATH,
    usecols=[
        "author_id",
        "inbound",
        "response_tweet_id",
        "in_response_to_tweet_id"
    ],
    chunksize=chunk_size
):

    # Count total tweets
    total_counts = chunk["author_id"].value_counts()

    # Count customer messages
    inbound_counts = chunk[chunk["inbound"] == True]["author_id"].value_counts()

    # Count support replies
    outbound_counts = chunk[chunk["inbound"] == False]["author_id"].value_counts()

    # Count replies sent by each author
    response_counts = (
        chunk[chunk["response_tweet_id"].notna()]
        ["author_id"]
        .value_counts()
    )

    for author in total_counts.index:

        if author not in brand_stats:
            brand_stats[author] = {
                "total_tweets": 0,
                "customer_messages": 0,
                "support_replies": 0,
                "responses": 0
            }

        brand_stats[author]["total_tweets"] += int(total_counts.get(author, 0))
        brand_stats[author]["customer_messages"] += int(inbound_counts.get(author, 0))
        brand_stats[author]["support_replies"] += int(outbound_counts.get(author, 0))
        brand_stats[author]["responses"] += int(response_counts.get(author, 0))


# Convert results to DataFrame
stats = pd.DataFrame.from_dict(
    brand_stats,
    orient="index"
)

stats.index.name = "author_id"

stats = stats.reset_index()

# We are interested mainly in accounts that send support replies
brands = stats[
    stats["support_replies"] > 100
].copy()

# Sort by number of support replies
brands = brands.sort_values(
    "support_replies",
    ascending=False
)

print("\n==========================================")
print("TOP SUPPORT ACCOUNTS")
print("==========================================")

print(
    brands.head(30).to_string(index=False)
)

# Save results
brands.to_csv(
    "brand_statistics.csv",
    index=False
)

print("\n==========================================")
print("DONE")
print("==========================================")

print("\nSaved results to:")
print("brand_statistics.csv")