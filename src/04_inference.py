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

app = FastAPI(title="Tier-1 Movie Recommender", version="1.0")

class Artifacts:
    model = None
    movie_map = None
    reverse_movie_map = None
    movie_titles = None
    index = None
    genome_embeddings = None

artifacts = Artifacts()

def load_artifacts():
    print("Loading Artifacts...")
    
    df_map = pl.read_parquet(config.MOVIE_MAP_PATH)
    artifacts.movie_map = dict(zip(df_map["original_movieId"], df_map["mapped_movieId"]))
    artifacts.reverse_movie_map = dict(zip(df_map["mapped_movieId"], df_map["original_movieId"]))
    
    df_movies = pl.read_csv(config.MOVIES_PATH)
    artifacts.movie_titles = dict(zip(df_movies["movieId"], df_movies["title"]))
    
    if os.path.exists(config.GENOME_EMBEDDINGS_PATH):
        artifacts.genome_embeddings = np.load(config.GENOME_EMBEDDINGS_PATH)
    
    num_items = len(df_map)
    artifacts.model = SASRec(
        num_items=num_items, 
        config=config, 
        genome_features=artifacts.genome_embeddings
    )
    
    dummy_input = tf.zeros((1, config.MAX_LEN), dtype=tf.int32)
    artifacts.model(dummy_input) 
    
    weights_path = os.path.join(config.PROCESSED_DIR, "best_model.weights.h5")
    artifacts.model.load_weights(weights_path)
    print("Model Loaded.")

    item_embeddings = artifacts.model.item_embedding.get_weights()[0]
    faiss.normalize_L2(item_embeddings)
    
    d = item_embeddings.shape[1]
    artifacts.index = faiss.IndexFlatIP(d)
    artifacts.index.add(item_embeddings)
    print(f"FAISS Index Built with {artifacts.index.ntotal} items.")

load_artifacts()

def get_user_embedding(sequence):
    seq_padded = pad_sequences([sequence], maxlen=config.MAX_LEN, padding='pre', truncating='pre')
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

class RecommendationRequest(BaseModel):
    user_history_ids: List[int]
    k: int = 10

@app.post("/recommend")
def recommend(req: RecommendationRequest):
    if not req.user_history_ids:
        raise HTTPException(status_code=400, detail="History cannot be empty")

    internal_ids = []
    for mid in req.user_history_ids:
        if mid in artifacts.movie_map:
            internal_ids.append(artifacts.movie_map[mid])
    
    if not internal_ids:
         raise HTTPException(status_code=404, detail="None of the movies in history were found in database.")

    user_vector = get_user_embedding(internal_ids)
    
    faiss.normalize_L2(user_vector)
    distances, indices = artifacts.index.search(user_vector, k=req.k + len(internal_ids))
    
    results = []
    seen_ids = set(internal_ids)
    
    for i, idx in enumerate(indices[0]):
        if idx in artifacts.reverse_movie_map:
            original_id = artifacts.reverse_movie_map[idx]
            if idx not in seen_ids:
                results.append({
                    "movie_id": original_id,
                    "title": artifacts.movie_titles.get(original_id, "Unknown Title"),
                    "score": float(distances[0][i])
                })
        
        if len(results) >= req.k:
            break
            
    return {"recommendations": results}
