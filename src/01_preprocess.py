# File: src/01_preprocess.py
import polars as pl
import numpy as np
import config

def process_data():
    print("Loading Data with Polars...")
    
    # 1. Load Ratings & Movies
    # We only need userId, movieId, timestamp
    ratings = pl.read_csv(config.RATINGS_PATH, columns=["userId", "movieId", "timestamp"])
    
    # 2. Filter Noisy Data (Tier-1 Practice)
    # Remove users who have watched < 5 movies. They don't provide enough signal for SASRec.
    user_counts = ratings["userId"].value_counts()
    active_users = user_counts.filter(pl.col("count") >= config.MIN_INTERACTIONS)["userId"]
    ratings = ratings.filter(pl.col("userId").is_in(active_users))
    
    print(f"Filtered Ratings: {len(ratings)}")

    # 3. Create Contiguous Movie ID Map
    # Neural Networks need inputs like [0, 1, 2], not [1, 590, 2093]
    unique_movie_ids = ratings["movieId"].unique().sort()
    movie_map = pl.DataFrame({
        "original_movieId": unique_movie_ids,
        "mapped_movieId": np.arange(1, len(unique_movie_ids) + 1) # 0 is reserved for padding
    })
    
    # Save the map so we can reverse it later (to show movie titles)
    movie_map.write_parquet(config.MOVIE_MAP_PATH)
    print(f"Saved Movie Map. Total Unique Movies: {len(movie_map)}")

    # 4. Join Map to Ratings
    ratings = ratings.join(movie_map, left_on="movieId", right_on="original_movieId", how="inner")

    # 5. Create Sequences (The "Netflix" Logic)
    # Sort by User and Time
    ratings = ratings.sort(["userId", "timestamp"])
    
    # Group by User and collect list of movies
    # This creates a "session" for every user
    user_sequences = ratings.group_by("userId").agg(
        pl.col("mapped_movieId").alias("sequence")
    )

    # 6. Temporal Split (Leave-One-Out)
    # We use the last item for validation, and the rest for training
    # This prevents "looking into the future"
    
    # We can't easily do LOO in pure Polars without custom logic, so we'll save the sequences
    # and handle the splitting in the Dataset pipeline or here.
    # Let's simple split: 90% users for train, 10% for val (simpler for now)
    
    # Shuffle users
    user_sequences = user_sequences.sample(fraction=1.0, shuffle=True, seed=42)
    
    test_size = int(len(user_sequences) * 0.1)
    train_df = user_sequences.slice(0, len(user_sequences) - test_size)
    val_df = user_sequences.slice(len(user_sequences) - test_size, test_size)

    print(f"Train Sequences: {len(train_df)}")
    print(f"Val Sequences: {len(val_df)}")

    # Save to Parquet (Fast binary format)
    train_df.write_parquet(config.TRAIN_DATA_PATH)
    val_df.write_parquet(config.VAL_DATA_PATH)
    print("Data Preprocessing Complete!")

if __name__ == "__main__":
    process_data()