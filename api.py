# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from recommender import MovieRecommender
from llm import explain_recommendation, llm_rerank

app = FastAPI(title="Movie Recommendation API")

# ---------------- INIT ----------------
recommender = MovieRecommender(
    movies_path="data/raw/ml-25m/movies.csv",
    embeddings_path="item_embeddings.npy",
)

# ---------------- SCHEMAS ----------------
class RecommendRequest(BaseModel):
    title: str
    k: int = 10


class Recommendation(BaseModel):
    title: str
    year: int | None
    score: float
    reason: str


class RecommendResponse(BaseModel):
    cold_start: bool
    results: List[Recommendation]


# ---------------- ENDPOINT ----------------
@app.post("/recommend", response_model=RecommendResponse)
def recommend_movies(req: RecommendRequest):
    cold, recs = recommender.recommend(req.title, req.k)

    # ---- LLM RE-RANK (SAFE) ----
    try:
        recs = llm_rerank(req.title, recs)
    except Exception:
        pass  # never break API

    results = []
    for r in recs:
        reason = explain_recommendation(
            seed_title=req.title,
            candidate_title=r["title"],
            score=r.get("score"),
            year=r.get("year"),
            popularity=r.get("num_ratings"),
        )

        results.append(
            {
                "title": r["title"],
                "year": r["year"],
                "score": round(float(r["score"]), 4),
                "reason": reason,
            }
        )

    return {
        "cold_start": cold,
        "results": results,
    }
