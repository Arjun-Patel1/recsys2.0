
google drive link rating.csv:- https://drive.google.com/file/d/1WsFSjJtkFeHTF-nHWGjFlGd7xtA7tFWp/view?usp=sharing
google drive link als.csv:- https://drive.google.com/file/d/1pq0nw2ATnzMTiudG4NHObNP1ydPijYIB/view?usp=sharing

🎬 Mini Hybrid Movie Recommendation System (100 Movies)

This project is a mini, end-to-end hybrid recommender system built using a subset of 100 movies from the MovieLens dataset to demonstrate real-world recommendation system design while keeping the repository lightweight and GitHub-friendly.

🔍 Project Overview

The system combines:

Collaborative Filtering (ALS) using implicit feedback

Content-Based Filtering using MovieLens genome tag relevance

Hybrid Recommendation Strategy (weighted combination)

Streamlit Web App with movie posters and clickable links (TMDB API)

This mini version is intentionally reduced to 100 movies for:

Easy cloning & execution

Fast training and inference

Clean GitHub presentation for recruiters

📦 Dataset Details (Mini Version)

Movies: 100

Ratings: Filtered to include only these movies

Content Features: Genome tag relevance (subset)

Source: MovieLens 25M (processed & reduced)

Files used:

data/
├── movies_10k.csv        → reduced to 100 movies
├── ratings_10k.csv       → ratings for selected movies
├── genome_scores_10k.csv → content features


⚠️ This is a demonstration-scale dataset.
The same pipeline scales to 10K / 25M+ movies in production environments.
🧠 Recommendation Techniques Used
1️⃣ Collaborative Filtering (ALS)

Library: implicit

Matrix factorization on user–item sparse matrix

Captures user behavior & preferences

2️⃣ Content-Based Filtering

Uses genome tag relevance vectors

Cosine similarity between movies

Handles cold-start users

3️⃣ Hybrid Recommendation

Final score:

Hybrid Score = α × ALS Score + β × Content Score


Balances personalization + similarity.

🖥️ Web Application (Streamlit)

Features:

User ID based recommendations

Hybrid / ALS / Content views

Movie posters & clickable links (TMDB API)

Modern Netflix-style UI

Lightweight & fast execution

Run locally:

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
├── artifacts/
│   ├── als_model.pkl
│   ├── user_item_matrix.npz
│   ├── content_features.npy
│
├── data/
│   ├── movies_10k.csv
│   ├── ratings_10k.csv
│   └── genome_scores_10k.csv
│
├── notebooks/
│   ├── 01_create_mini_dataset.ipynb
│   └── 02_train_hybrid_model.ipynb
│
└── README.md

🚀 Scalability Note (Important for Recruiters)

This project is a scaled-down version for GitHub.

In production:

Dataset → millions of users & items

Models → trained offline (Spark / GPU)

Serving → APIs + caching layers

Posters → batch metadata pipelines

The architecture and logic remain identical.

🎯 Why This Project Matters

✔ Demonstrates real recommender system design
✔ Covers cold-start + personalization
✔ Uses industry-standard tools
✔ Clean, deployable, and explainable
