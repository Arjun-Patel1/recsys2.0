
google drive link rating.csv:- https://drive.google.com/file/d/1WsFSjJtkFeHTF-nHWGjFlGd7xtA7tFWp/view?usp=sharing
google drive link als.csv:- https://drive.google.com/file/d/1pq0nw2ATnzMTiudG4NHObNP1ydPijYIB/view?usp=sharing

# 🎬 Mini Hybrid Movie Recommendation System

A **production-style hybrid recommender system** built using **Collaborative Filtering (ALS)** and **Content-Based Filtering**, demonstrated on a **lightweight 100-movie subset** of the MovieLens dataset for fast execution and GitHub-friendly sharing.

This project mirrors **real-world recommender system design** while keeping the repository small and easy to run locally.

---

## 🚀 Key Features

- Hybrid Recommendation Engine (ALS + Content-Based)
- Cold-start user handling
- Sparse matrix optimization (CSR)
- Movie similarity using genome tag relevance
- Streamlit web app with:
  - Movie posters
  - Clickable links
  - Modern Netflix-style UI
- Fully reproducible pipeline

---

## 📦 Dataset (Mini Version)

This repository uses a **reduced dataset of 100 movies** derived from MovieLens 25M.

| Component | Description |
|---------|------------|
| Movies | 100 selected movies |
| Ratings | Filtered ratings for selected movies |
| Content | Genome tag relevance vectors |
| Purpose | Fast demo & GitHub hosting |

> ⚠️ **Note:**  
> This is a **demonstration-scale dataset**.  
> The same pipeline scales to **10K+ movies and millions of users** in production.

---

## 🧠 Recommendation Techniques

### 1️⃣ Collaborative Filtering (ALS)
- Matrix factorization using the `implicit` library
- Learns latent user–item interactions
- Optimized using sparse CSR matrices

### 2️⃣ Content-Based Filtering
- Movie similarity via genome tag relevance
- Cosine similarity on normalized feature vectors
- Handles cold-start users

### 3️⃣ Hybrid Strategy
Final recommendation score:

Hybrid Score = α × ALS Score + β × Content Score

Combines personalization with semantic similarity.

---

## 🖥️ Web Application (Streamlit)

Features:
- User ID based recommendations
- Hybrid / ALS / Content views
- Movie posters fetched via TMDB API
- Clickable movie pages
- Responsive, modern UI

Run locally:
```bash
streamlit run app.py
🛠️ Tech Stack

Python

Pandas, NumPy

SciPy (CSR Sparse Matrix)

Scikit-learn

Implicit (ALS)

Streamlit

TMDB API

Git & GitHub
📁 Project Structure
mini_rec_sys/
│
├── app.py
│
├── artifacts/
│   ├── als_model.pkl
│   ├── user_item_matrix.npz
│   ├── content_features.npy
│   ├── index_to_movie.pkl
│   └── movieid_to_content_index.pkl
│
├── data/
│   ├── movies_10k.csv
│   ├── ratings_10k.csv
│   ├── genome_scores_10k.csv
│   └── links_10k.csv
│
├── notebooks/
│   ├── 01_create_mini_dataset.ipynb
│   └── 02_train_hybrid_model.ipynb
│
└── README.md

📈 Scalability & Production Readiness

This mini version demonstrates:

End-to-end recommender system pipeline

Offline training & online inference

Hybrid recommendation logic

In production:

Datasets → millions of users & items

Training → distributed (Spark / GPUs)

Serving → APIs + caching layers

Metadata → batch ingestion pipelines

🎯 Why This Project Stands Out

Real-world recommender architecture

Cold-start & personalization handled

Industry-standard tools

Clean, deployable, and explainable

Optimized for interviews & portfolios

📌 Disclaimer

This project uses a reduced dataset for demonstration purposes only.
Original data source: MovieLens Dataset

👤 Author

Arjun Patel
AI / Machine Learning Engineer

🔗 GitHub: https://github.com/Arjun-Patel1
