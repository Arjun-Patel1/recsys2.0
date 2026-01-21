🎬 Hybrid Movie Recommendation System (Production-Scale Evolution)

A production-inspired hybrid movie recommendation system built on the MovieLens 25M dataset, evolving from a traditional ALS-based recommender into a modern, scalable system combining graph learning, vector search, time-aware ranking, and LLM-based reasoning.

This project demonstrates how a recommender system can be iteratively improved to reflect real-world architectures used by platforms like Netflix, Amazon Prime, and Spotify.

🚀 Why This Project Exists

Most academic recommender systems stop at:

Matrix factorization

Offline evaluation

Static recommendations

This project goes beyond that, focusing on:

Scalability

Modular architecture

Real-time serving

Explainability

Cold-start robustness

Production readiness

📊 Dataset: MovieLens 25M

This system is trained and evaluated using the MovieLens 25M dataset provided by GroupLens Research.

Dataset characteristics:

25+ million ratings

Tens of thousands of movies

Millions of users

Genome tag relevance vectors for semantic understanding

Sparse, implicit feedback–style interactions

The dataset enables experimentation at realistic scale, closer to industry workloads than toy datasets.

⚠️ Due to dataset size and licensing, raw data and large artifacts are not fully hosted in this repository.

🧠 System Evolution: From Classic to Production-Grade
🔹 Initial Version (Baseline)

Collaborative Filtering using ALS

Content-based similarity using genome tags

Static ranking

Limited cold-start handling

While effective, this approach had limitations:

Poor temporal awareness

No user–item graph structure

Expensive similarity computation at scale

Limited explainability

🔹 Improved Architecture (Current System)

The latest version introduces four major upgrades:

1️⃣ Graph-Based Collaborative Filtering (LightGCN)

Why the change?

ALS assumes linear latent interactions

Real user–item behavior is graph-structured

Upgrade:

LightGCN models users and movies as nodes in a bipartite graph

Learns embeddings via neighborhood aggregation

Better captures collaborative signals

More robust to sparsity

2️⃣ Vector Search with FAISS

Problem with classic similarity search:

O(N²) similarity computation

Not viable at scale

Solution:

Precomputed item embeddings

FAISS for fast Approximate Nearest Neighbor (ANN) search

Sub-second retrieval even with tens of thousands of items

This enables real-time recommendations.

3️⃣ Time-Aware Ranking

Traditional recommenders treat all interactions equally.

Improvement:

Recent interactions are weighted higher

Older preferences decay over time

Produces fresher, more relevant recommendations

This mimics real user behavior on streaming platforms.

4️⃣ LLM-Assisted Reasoning & Re-Ranking

To improve user trust and explainability, the system integrates an LLM (Mistral via Ollama):

Generates faithful, non-hallucinated explanations

Uses only collaborative signals (no invented plots)

Optional re-ranking layer to refine top-K results

Falls back to deterministic logic if LLM fails

This bridges the gap between ML systems and user experience.

🔁 Hybrid Recommendation Strategy

The final recommendation score combines multiple signals:

Graph-based collaborative relevance

Content similarity (semantic signals)

Time-aware weighting

Optional LLM refinement

This hybrid design ensures:

Strong personalization

Cold-start resilience

Diverse yet relevant recommendations

🖥️ Application Layer

The system is exposed via:

FastAPI for model serving

Streamlit for interactive UI

Features:

Title-based recommendations

Adjustable top-K results

Confidence scores

Natural-language explanations

Clean, production-inspired interface

🛠️ Tech Stack Overview
Machine Learning & Data

Python

NumPy, Pandas

SciPy (sparse matrices)

PyTorch (LightGCN)

Sentence Transformers (semantic embeddings)

FAISS (vector search)

Serving & Applications

FastAPI (inference API)

Streamlit (frontend UI)

Ollama + Mistral (LLM reasoning)

MLOps & Tooling

Hugging Face (artifact hosting)

Git & GitHub

Docker (containerization)

Modular, decoupled architecture

📈 Scalability & Production Thinking

This project mirrors real recommender pipelines:

Offline training

Artifact persistence

Fast online inference

Separation of concerns (model / API / UI)

Graceful degradation (LLM optional)

Future-ready extensions:

Distributed training (Spark)

Online learning

A/B testing

Caching layers (Redis)

Monitoring & metrics

🧪 Evaluation Philosophy

Offline metrics (Recall@K, Precision@K)

Qualitative evaluation via UI

Focus on diversity, freshness, and explainability

Hybrid approach shows improved relevance over pure CF

🎯 Why This Project Matters

✔ Demonstrates real-world recommender evolution
✔ Goes beyond academic baselines
✔ Handles scale, sparsity, and cold-start
✔ Combines ML + systems + UX
✔ Strong portfolio & interview-ready project

⚠️ Disclaimer

MovieLens data © GroupLens Research

LLM outputs are constrained to avoid hallucination

This project is for educational and portfolio purposes

👤 Author

Arjun Patel
AI / Machine Learning Engineer

GitHub: https://github.com/Arjun-Patel1

LinkedIn: https://www.linkedin.com/in/arjunpatel97259
