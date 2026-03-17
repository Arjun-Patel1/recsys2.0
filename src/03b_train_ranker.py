# File: src/03b_train_ranker.py  Deep & Cross Network
import os

# --- THIS IS THE FIX ---
# Must be set before importing TensorFlow or TFRS
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# -----------------------

import polars as pl
import tensorflow as tf
import tensorflow_recommenders as tfrs
import config


class DCNRanker(tf.keras.Model):
    def __init__(self, num_users, num_movies, embed_dim=64):
        super().__init__()

        # 1. Embeddings
        self.user_embedding = tf.keras.layers.Embedding(num_users + 1, embed_dim)
        self.movie_embedding = tf.keras.layers.Embedding(num_movies + 1, embed_dim)

        # 2. Deep & Cross Network
        self.cross_layer = tfrs.layers.dcn.Cross()

        # 3. Deep Layers
        self.deep_layers = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
            ]
        )

        # 4. Final Output
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        user_emb = self.user_embedding(inputs["userId"])
        movie_emb = self.movie_embedding(inputs["movieId"])
        x = tf.concat([user_emb, movie_emb], axis=1)
        cross_out = self.cross_layer(x)
        deep_out = self.deep_layers(x)
        combined = tf.concat([cross_out, deep_out], axis=1)
        return self.output_layer(combined)


def train_ranker():
    print("Loading Ratings Data...")
    ratings = pl.read_csv(config.RATINGS_PATH, columns=["userId", "movieId", "rating"])

    movie_map = pl.read_parquet(config.MOVIE_MAP_PATH)
    ratings = ratings.join(
        movie_map, left_on="movieId", right_on="original_movieId", how="inner"
    )
    ratings = ratings.select(["userId", "mapped_movieId", "rating"])

    num_users = ratings["userId"].max()
    num_movies = movie_map["mapped_movieId"].max()

    print(f"Users: {num_users}, Movies: {num_movies}")

    # 5 Million row sample
    sample = ratings.sample(n=5_000_000, seed=42)

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                "userId": sample["userId"].to_numpy(),
                "movieId": sample["mapped_movieId"].to_numpy(),
            },
            sample["rating"].to_numpy(),
        )
    )

    dataset = dataset.shuffle(100_000).batch(2048).prefetch(tf.data.AUTOTUNE)

    model = DCNRanker(num_users=num_users, num_movies=num_movies)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )

    print("Training DCN Ranker...")
    model.fit(dataset, epochs=3)

    model.save_weights(os.path.join(config.PROCESSED_DIR, "dcn_ranker.weights.h5"))
    print("Ranking Model Saved!")


if __name__ == "__main__":
    train_ranker()
