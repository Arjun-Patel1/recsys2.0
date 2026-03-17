# File: src/04_inference.py
import sys
import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Force Python to look in the exact folder where this file lives
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

import config

import numpy as np
import polars as pl
import tensorflow as tf
import tensorflow_recommenders as tfrs
import faiss
import uvicorn
from fastapi import FastAPI, HTTPException

# --- Add these two missing imports ---
from pydantic import BaseModel
from typing import List

# -------------------------------------

from tensorflow.keras.preprocessing.sequence import pad_sequences

from models import SASRec

# ... (the rest of your code remains unchanged)


# --- 1. Define DCN Ranker Class (Included here to avoid numeric import errors) ---
class DCNRanker(tf.keras.Model):
    def __init__(self, num_users, num_movies, embed_dim=64):
        super().__init__()
        self.user_embedding = tf.keras.layers.Embedding(num_users + 1, embed_dim)
        self.movie_embedding = tf.keras.layers.Embedding(num_movies + 1, embed_dim)
        self.cross_layer = tfrs.layers.dcn.Cross()
        self.deep_layers = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
            ]
        )
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        user_emb = self.user_embedding(inputs["userId"])
        movie_emb = self.movie_embedding(inputs["movieId"])
        x = tf.concat([user_emb, movie_emb], axis=1)
        cross_out = self.cross_layer(x)
        deep_out = self.deep_layers(x)
        combined = tf.concat([cross_out, deep_out], axis=1)
        return self.output_layer(combined)


# --- 2. Initialize API ---
app = FastAPI(title="Tier-1 Movie Recommender", version="2.0")


# --- 3. Global State ---
class Artifacts:
    model = None
    movie_map = None
    reverse_movie_map = None
    movie_titles = None
    index = None  # FAISS Index
    genome_embeddings = None
    ranker = None  # DCN Ranker


artifacts = Artifacts()


def load_artifacts():
    print("Loading Artifacts...")

    # A. Load Maps
    df_map = pl.read_parquet(config.MOVIE_MAP_PATH)
    artifacts.movie_map = dict(
        zip(df_map["original_movieId"], df_map["mapped_movieId"])
    )
    artifacts.reverse_movie_map = dict(
        zip(df_map["mapped_movieId"], df_map["original_movieId"])
    )

    # Load Titles
    df_movies = pl.read_csv(config.MOVIES_PATH)
    artifacts.movie_titles = dict(zip(df_movies["movieId"], df_movies["title"]))

    # B. Load Genome
    if os.path.exists(config.GENOME_EMBEDDINGS_PATH):
        artifacts.genome_embeddings = np.load(config.GENOME_EMBEDDINGS_PATH)

    # C. Initialize & Load SASRec Model
    num_items = len(df_map)
    artifacts.model = SASRec(
        num_items=num_items, config=config, genome_features=artifacts.genome_embeddings
    )
    dummy_input = tf.zeros((1, config.MAX_LEN), dtype=tf.int32)
    artifacts.model(dummy_input)
    artifacts.model.load_weights(
        os.path.join(config.PROCESSED_DIR, "best_model.weights.h5")
    )
    print("SASRec Retrieval Model Loaded.")

    # D. Build FAISS Index
    item_embeddings = artifacts.model.item_embedding.get_weights()[0]
    faiss.normalize_L2(item_embeddings)
    d = item_embeddings.shape[1]
    artifacts.index = faiss.IndexFlatIP(d)
    artifacts.index.add(item_embeddings)
    print(f"FAISS Index Built with {artifacts.index.ntotal} items.")

    # E. Initialize & Load DCN Ranker
    # Note: 162541 is the total unique users in MovieLens 25M
    artifacts.ranker = DCNRanker(num_users=162541, num_movies=num_items)

    # Dummy pass to build layers
    artifacts.ranker({"userId": tf.constant([1]), "movieId": tf.constant([1])})

    ranker_weights = os.path.join(config.PROCESSED_DIR, "dcn_ranker.weights.h5")
    if os.path.exists(ranker_weights):
        artifacts.ranker.load_weights(ranker_weights)
        print("DCN Ranking Model Loaded.")
    else:
        print("WARNING: DCN Ranker weights not found! Ranking will be bypassed.")


# Load artifacts on startup
load_artifacts()


# --- Helper Functions ---
def get_user_embedding(sequence):
    seq_padded = pad_sequences(
        [sequence], maxlen=config.MAX_LEN, padding="pre", truncating="pre"
    )
    inputs = tf.convert_to_tensor(seq_padded)

    x = artifacts.model.item_embedding(inputs)

    if artifacts.model.use_genome:
        genome_emb = tf.nn.embedding_lookup(artifacts.model.genome_matrix, inputs)
        genome_emb = artifacts.model.genome_projection(genome_emb)
        x = x + genome_emb

    seq_len = tf.shape(inputs)[1]
    positions = tf.range(start=0, limit=seq_len, delta=1)
    x = x + artifacts.model.pos_embedding(positions)

    mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    for block in artifacts.model.blocks:
        x = artifacts.model.layer_norm1(x + block(x, x, x, attention_mask=mask))

    user_vector = x[:, -1, :]
    return user_vector.numpy()


# --- API Endpoints ---
class RecommendationRequest(BaseModel):
    user_history_ids: List[int]
    k: int = 10


@app.post("/recommend")
def recommend(req: RecommendationRequest):
    if not req.user_history_ids:
        raise HTTPException(status_code=400, detail="History cannot be empty")

    internal_ids = [
        artifacts.movie_map[mid]
        for mid in req.user_history_ids
        if mid in artifacts.movie_map
    ]

    if not internal_ids:
        raise HTTPException(
            status_code=404,
            detail="None of the movies in history were found in database.",
        )

    # -------------------------------------------------------------------
    # STAGE 1: RETRIEVAL (Fetch Top 100 via FAISS)
    # -------------------------------------------------------------------
    user_vector = get_user_embedding(internal_ids)
    faiss.normalize_L2(user_vector)

    # We grab 100 candidates to rank, plus extra to account for history filtering
    distances, indices = artifacts.index.search(user_vector, k=100 + len(internal_ids))

    seen_ids = set(internal_ids)
    valid_candidates = []

    for idx in indices[0]:
        if idx in artifacts.reverse_movie_map and idx not in seen_ids:
            valid_candidates.append(idx)
            if len(valid_candidates) == 100:  # Cap at 100
                break

    # -------------------------------------------------------------------
    # STAGE 2: RANKING (Score the Top 100 via DCN)
    # -------------------------------------------------------------------
    if artifacts.ranker and valid_candidates:
        # Default user ID to 1 for anonymous web queries
        demo_user_id = tf.constant([1] * len(valid_candidates), dtype=tf.int32)
        candidate_tensor = tf.constant(valid_candidates, dtype=tf.int32)

        # Predict the Star Rating
        predicted_ratings = (
            artifacts.ranker({"userId": demo_user_id, "movieId": candidate_tensor})
            .numpy()
            .flatten()
        )

        # Zip IDs with predicted ratings and Sort High->Low
        scored_items = list(zip(valid_candidates, predicted_ratings))
        scored_items.sort(key=lambda x: x[1], reverse=True)
    else:
        # Fallback if ranker fails or isn't trained yet
        scored_items = [
            (valid_candidates[i], float(distances[0][i]))
            for i in range(len(valid_candidates))
        ]

    # -------------------------------------------------------------------
    # STAGE 3: SERVE (Return Top K to Frontend)
    # -------------------------------------------------------------------
    results = []
    for idx, score in scored_items[: req.k]:
        original_id = artifacts.reverse_movie_map[idx]
        results.append(
            {
                "movie_id": original_id,
                "title": artifacts.movie_titles.get(original_id, "Unknown Title"),
                "score": float(score),
            }
        )

    return {"recommendations": results}


@app.get("/")
def health_check():
    return {
        "status": "Model is Live",
        "movies_indexed": artifacts.index.ntotal,
        "ranker_active": artifacts.ranker is not None,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
