# 🎬 Hybrid Movie Recommendation System (Production-Scale)

A **full-scale hybrid recommendation system** inspired by real-world platforms like **Netflix & Amazon Prime**, combining **Collaborative Filtering (ALS)** and **Content-Based Filtering** to deliver highly personalized movie recommendations.

This project is designed to demonstrate **industry-level recommender system architecture**, scalability, and deployment readiness.

---

## 📸 Application Preview

> 📌 *Screenshots of the Streamlit web application will be added here*

- Home screen
- User-based recommendations
- Hybrid recommendations
- Movie posters with external links

---

## 🚀 Project Highlights

- Hybrid Recommendation Engine (ALS + Content-Based)
- Handles **cold-start users**
- Large-scale sparse matrix optimization
- Movie similarity via semantic genome tags
- API-driven poster & movie metadata fetching
- Modern, Netflix-inspired UI
- End-to-end ML pipeline (data → model → UI)

---

## 📊 Dataset Overview

This project uses the **MovieLens (25M+) dataset**, containing:

| Component | Description |
|--------|------------|
| Users | Millions of users |
| Movies | Tens of thousands of movies |
| Ratings | 25M+ interactions |
| Content | Genome tag relevance vectors |

> ⚠️ Due to size constraints, **trained artifacts are not fully hosted on GitHub**.  
> A reduced demo version is provided separately.

---

## 🧠 Recommendation Architecture

### 1️⃣ Collaborative Filtering (ALS)
- Matrix factorization using **Implicit ALS**
- Learns latent user–item interactions
- Efficient sparse matrix representation (CSR)
- Personalized ranking for each user

### 2️⃣ Content-Based Filtering
- Movie similarity using **genome tag relevance**
- Cosine similarity on normalized feature vectors
- Works for new users with no history

### 3️⃣ Hybrid Recommendation Strategy

Final score:


Hybrid Score = α × Collaborative Score + β × Content Score

This ensures:
- Personalization (ALS)
- Semantic relevance (Content)
- Robust cold-start handling

---

## 🖥️ Web Application

Built using **Streamlit**, featuring:

- User ID based recommendations
- Hybrid / ALS / Content-based toggles
- Movie posters fetched via **TMDB API**
- Clickable movie detail links
- Responsive, modern UI

Launch locally:
```bash
streamlit run app.py
```
🛠️ Tech Stack
Machine Learning

Python

NumPy, Pandas

SciPy (Sparse Matrices)

Scikit-learn

Implicit (ALS)

Backend & Serving

Streamlit

TMDB API

Pickle / NumPy persistence

DevOps & Tools

Git & GitHub

Jupyter Notebook

Environment-based API handling

📁 Project Structure
recsys2.0/
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
│   ├── movies.csv
│   ├── ratings.csv
│   ├── genome_scores.csv
│   └── links.csv
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── content_feature_engineering.ipynb
│   └── model_training.ipynb
│
└── README.md

📈 Scalability & Production Design

This system mirrors real-world recommender pipelines:

Offline training on large datasets

Sparse matrix factorization

Feature persistence for fast inference

Decoupled training & serving layers

Production Extensions:

Spark-based training

Online inference APIs (FastAPI)

Redis caching

A/B testing & metrics tracking

🧪 Evaluation Strategy

Offline validation via:

Precision@K

Recall@K

Coverage

Qualitative evaluation through UI testing

Hybrid model shows improved diversity and relevance
relevance

🎯 Why This Project Matters

✔ Real-world recommendation system design
✔ Hybrid modeling approach
✔ Cold-start problem handling
✔ Scalable architecture
✔ Deployable UI
✔ Strong portfolio & interview project

⚠️ Disclaimer

Movie data belongs to GroupLens Research

Posters & metadata fetched using TMDB API

This project is for educational and portfolio purposes

👤 Author

Arjun Patel
AI / Machine Learning Engineer

🔗 GitHub: https://github.com/Arjun-Patel1

📌 LinkedIn: www.linkedin.com/in/arjunpatel97259

