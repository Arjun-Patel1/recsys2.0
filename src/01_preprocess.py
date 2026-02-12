
import polars as pl
import numpy as np
import config

def process_data():
    print("Loading Data with Polars...")
    
    ratings = pl.read_csv(config.RATINGS_PATH, columns=["userId", "movieId", "timestamp"])
    
    user_counts = ratings["userId"].value_counts()
    active_users = user_counts.filter(pl.col("count") >= config.MIN_INTERACTIONS)["userId"]
    ratings = ratings.filter(pl.col("userId").is_in(active_users))
    
    print(f"Filtered Ratings: {len(ratings)}")

    unique_movie_ids = ratings["movieId"].unique().sort()
    movie_map = pl.DataFrame({
        "original_movieId": unique_movie_ids,
        "mapped_movieId": np.arange(1, len(unique_movie_ids) + 1)
    })
    
    movie_map.write_parquet(config.MOVIE_MAP_PATH)
    print(f"Saved Movie Map. Total Unique Movies: {len(movie_map)}")

    ratings = ratings.join(movie_map, left_on="movieId", right_on="original_movieId", how="inner")

    ratings = ratings.sort(["userId", "timestamp"])
    
    user_sequences = ratings.group_by("userId").agg(
        pl.col("mapped_movieId").alias("sequence")
    )

    user_sequences = user_sequences.sample(fraction=1.0, shuffle=True, seed=42)
    
    test_size = int(len(user_sequences) * 0.1)
    train_df = user_sequences.slice(0, len(user_sequences) - test_size)
    val_df = user_sequences.slice(len(user_sequences) - test_size, test_size)

    print(f"Train Sequences: {len(train_df)}")
    print(f"Val Sequences: {len(val_df)}")

    train_df.write_parquet(config.TRAIN_DATA_PATH)
    val_df.write_parquet(config.VAL_DATA_PATH)
    print("Data Preprocessing Complete!")

if __name__ == "__main__":
    process_data()
