# File: src/04_inference.py
import os
import numpy as np
import polars as pl
import tensorflow as tf
import faiss
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from tensorflow.keras.preprocessing.sequence import pad_sequences

import config
from models import SASRec

# 1. Initialize API
app = FastAPI(title="Tier-1 Movie Recommender", version="1.0")

# 2. Global State (Load once, serve many)
class Artifacts:
    model = None
    movie_map = None
    reverse_movie_map = None
    movie_titles = None
    index = None # FAISS Index
    genome_embeddings = None

artifacts = Artifacts()

def load_artifacts():
    print("Loading Artifacts...")
    
    # A. Load Maps
    # Maps internal ID (0,1,2) back to original ID (1, 592, ...)
    df_map = pl.read_parquet(config.MOVIE_MAP_PATH)
    artifacts.movie_map = dict(zip(df_map["original_movieId"], df_map["mapped_movieId"]))
    artifacts.reverse_movie_map = dict(zip(df_map["mapped_movieId"], df_map["original_movieId"]))
    
    # Load Titles for display
    df_movies = pl.read_csv(config.MOVIES_PATH)
    artifacts.movie_titles = dict(zip(df_movies["movieId"], df_movies["title"]))
    
    # B. Load Genome (if exists)
    if os.path.exists(config.GENOME_EMBEDDINGS_PATH):
        artifacts.genome_embeddings = np.load(config.GENOME_EMBEDDINGS_PATH)
    
    # C. Initialize & Load Model
    num_items = len(df_map)
    artifacts.model = SASRec(
        num_items=num_items, 
        config=config, 
        genome_features=artifacts.genome_embeddings
    )
    
    # Build the model by running a dummy input
    dummy_input = tf.zeros((1, config.MAX_LEN), dtype=tf.int32)
    artifacts.model(dummy_input) 
    
    # Load Trained Weights
    weights_path = os.path.join(config.PROCESSED_DIR, "best_model.weights.h5")
    artifacts.model.load_weights(weights_path)
    print("Model Loaded.")

    # D. Build FAISS Index (The "Two-Tower" Retrieval Trick)
    # We extract the Item Embeddings from the model to search against them.
    item_embeddings = artifacts.model.item_embedding.get_weights()[0] # Shape: (Num_Items, 128)
    
    # Normalize for Cosine Similarity (Dot Product on normalized vectors = Cosine)
    faiss.normalize_L2(item_embeddings)
    
    # Create Index
    d = item_embeddings.shape[1] # 128
    artifacts.index = faiss.IndexFlatIP(d) # Inner Product
    artifacts.index.add(item_embeddings)
    print(f"FAISS Index Built with {artifacts.index.ntotal} items.")

# Load artifacts on startup
load_artifacts()

# --- Helper Functions ---

def get_user_embedding(sequence):
    """
    Runs the Transformer on the user history to get the 'Query Vector'.
    This mimics the forward pass of SASRec but stops before the final classification layer.
    """
    # 1. Pad sequence
    seq_padded = pad_sequences([sequence], maxlen=config.MAX_LEN, padding='pre', truncating='pre')
    inputs = tf.convert_to_tensor(seq_padded)
    
    # 2. Forward Pass (Manual extraction to get the state)
    # Note: We duplicate logic from models.py slightly to get intermediate output
    
    # Embeddings
    x = artifacts.model.item_embedding(inputs)
    
    # Genome Injection
    if artifacts.model.use_genome:
        genome_emb = tf.nn.embedding_lookup(artifacts.model.genome_matrix, inputs)
        genome_emb = artifacts.model.genome_projection(genome_emb)
        x = x + genome_emb
        
    # Positional
    seq_len = tf.shape(inputs)[1]
    positions = tf.range(start=0, limit=seq_len, delta=1)
    x = x + artifacts.model.pos_embedding(positions)
    
    # Blocks
    mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    for block in artifacts.model.blocks:
        x = artifacts.model.layer_norm1(x + block(x, x, x, attention_mask=mask))
        
    # Get the embedding of the LAST item in the sequence (The User's Current State)
    user_vector = x[:, -1, :] # Shape: (1, 128)
    return user_vector.numpy()

# --- API Endpoints ---

class RecommendationRequest(BaseModel):
    user_history_ids: List[int] # List of original Movie IDs (e.g., [1, 296, 318])
    k: int = 10

@app.post("/recommend")
def recommend(req: RecommendationRequest):
    if not req.user_history_ids:
        raise HTTPException(status_code=400, detail="History cannot be empty")

    # 1. Convert Original IDs to Internal IDs
    internal_ids = []
    for mid in req.user_history_ids:
        if mid in artifacts.movie_map:
            internal_ids.append(artifacts.movie_map[mid])
    
    if not internal_ids:
         raise HTTPException(status_code=404, detail="None of the movies in history were found in database.")

    # 2. Get User Embedding (Query Vector)
    user_vector = get_user_embedding(internal_ids)
    
    # 3. Retrieve with FAISS
    faiss.normalize_L2(user_vector)
    distances, indices = artifacts.index.search(user_vector, k=req.k + len(internal_ids))
    
    # 4. Filter & Format Results
    results = []
    seen_ids = set(internal_ids)
    
    for i, idx in enumerate(indices[0]):
        # idx is the Internal ID
        if idx in artifacts.reverse_movie_map:
            original_id = artifacts.reverse_movie_map[idx]
            
            # Don't recommend what they just watched
            if idx not in seen_ids:
                results.append({
                    "movie_id": original_id,
                    "title": artifacts.movie_titles.get(original_id, "Unknown Title"),
                    "score": float(distances[0][i])
                })
        
        if len(results) >= req.k:
            break
            
    return {"recommendations": results}

@app.get("/")
def health_check():
    return {"status": "Model is Live", "movies_indexed": artifacts.index.ntotal}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)