# recommender.py
import numpy as np
import pandas as pd
import faiss
import math
from datetime import datetime

CURRENT_YEAR = datetime.now().year
EMBEDDING_DIM = 64


class MovieRecommender:
    def __init__(self, movies_path, embeddings_path):
        self.movies = pd.read_csv(movies_path)
        self.embeddings = np.load(embeddings_path).astype("float32")

        # extract year
        self.movies["year"] = (
            self.movies["title"]
            .str.extract(r"\((\d{4})\)", expand=False)
            .astype(float)
        )

        # safe defaults
        self.movies["avg_rating"] = self.movies.get(
            "avg_rating", pd.Series(3.2, index=self.movies.index)
        )
        self.movies["num_ratings"] = self.movies.get(
            "num_ratings", pd.Series(20, index=self.movies.index)
        )

        faiss.normalize_L2(self.embeddings)
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.index.add(self.embeddings)

        print(f"FAISS ready | Items: {self.index.ntotal}")

    def time_decay(self, year):
        if np.isnan(year):
            return 0.85
        age = CURRENT_YEAR - year
        if age <= 10:
            return 1.0
        elif age <= 25:
            return 1.0 - (age - 10) * 0.015
        return max(0.55 - (age - 25) * 0.01, 0.25)

    def final_score(self, sim, year, avg_rating, num_ratings):
        s = sim ** 1.25
        r = np.clip(1.0 + (avg_rating - 3.5) * 0.25, 0.7, 1.4)
        p = np.clip(math.log10(num_ratings + 50) / math.log10(2000), 0.4, 1.0)
        t = self.time_decay(year)
        return s * r * p * t

    def get_candidates(self, idx, k=300):
        q = self.embeddings[idx].reshape(1, -1)
        D, I = self.index.search(q, k)
        return list(zip(I[0], D[0]))

    def recommend(self, title, top_k=10):
        match = self.movies[self.movies["title"].str.contains(title, case=False)]

        if match.empty:
            cold = (
                self.movies.assign(
                    cold_score=self.movies["avg_rating"]
                    * np.log1p(self.movies["num_ratings"])
                    * self.movies["year"].apply(self.time_decay)
                )
                .sort_values("cold_score", ascending=False)
                .head(top_k)
            )

            return True, [
                {
                    "title": row.title,
                    "year": int(row.year) if not np.isnan(row.year) else None,
                    "score": float(row.cold_score),
                }
                for _, row in cold.iterrows()
            ]

        seed_idx = match.index[0]
        candidates = self.get_candidates(seed_idx)

        scored = []
        old_count = 0

        for idx, sim in candidates:
            if idx == seed_idx:
                continue

            m = self.movies.iloc[idx]
            score = self.final_score(sim, m.year, m.avg_rating, m.num_ratings)

            if m.year and m.year < 1995:
                if old_count >= 2:
                    continue
                old_count += 1

            scored.append((idx, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scored[:top_k]:
            m = self.movies.iloc[idx]
            results.append(
                {
                    "title": m.title,
                    "year": int(m.year) if not np.isnan(m.year) else None,
                    "score": float(score),
                }
            )

        return False, results
