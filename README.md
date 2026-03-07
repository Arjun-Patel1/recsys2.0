# 🎬 ScaleRec: Tier-1 Multi-Stage Recommendation Engine & Agentic AI

[![Backend Docker](https://img.shields.io/badge/Docker-Backend-blue?logo=docker)](https://hub.docker.com/repository/docker/arjunpatel89806/movie-recsys-api)  
[![Frontend Docker](https://img.shields.io/badge/Docker-Frontend-blue?logo=docker)](https://hub.docker.com/repository/docker/arjunpatel89806/movie-recsys-app)

ScaleRec is a production-grade, multi-stage recommendation pipeline designed to mimic the core machine learning architectures of modern streaming platforms like Netflix and YouTube. 

Moving far beyond basic Matrix Factorization, ScaleRec implements a **Two-Stage Funnel** (High-Recall Retrieval + High-Precision Ranking) to process 25M+ interactions in under 50ms. Furthermore, it integrates a constrained **Agentic AI (Gemini 2.5 Flash)** to provide zero-shot conversational discovery for real-time trending pop culture.

---

## 🏗️ System Architecture: The 3 Pillars

1. **Stage 1: Candidate Generation (The "Librarian")**
   * Uses **SASRec (Self-Attentive Sequential Recommender)** to understand the temporal sequence of a user's watch history.
   * Semantic "Genome Tags" are injected via PCA to solve the cold-start problem.
   * Embeddings are stored in a **FAISS Vector Database** to filter 60,000+ items down to the Top 100 in `<10ms`.

2. **Stage 2: Heavy Ranking (The "Critic")**
   * The Top 100 candidates are passed into a **Deep & Cross Network (DCN-V2)** built with TensorFlow Recommenders.
   * The DCN explicitly learns complex, non-linear feature interactions between users and metadata to predict an exact Star Rating (0.5 to 5.0).
   * Sorts the subset to serve the absolute best items to the frontend.

3. **Stage 3: Conversational AI (The "Insider")**
   * Introduces **PopGuru**, an LLM-Augmented Agent powered by **Gemini 2.5 Flash**.
   * Bypasses the temporal limits of the training dataset by leveraging live web knowledge to recommend trending movies, Spotify songs, and YouTube trailers.
   * Uses strict system guardrails to decline out-of-domain queries and dynamically generates actionable Markdown links (JustWatch, BookMyShow).

---

## 🧰 Tech Stack

* **Frontend:** Streamlit (Custom Netflix-Tier CSS, Tabbed UI, Stateful Auth)
* **Backend:** FastAPI (Async REST APIs)
* **Machine Learning:** TensorFlow 2.x, TensorFlow Recommenders (TFRS), Keras
* **Vector Search:** FAISS (HNSW, Approximate Nearest Neighbor Search)
* **Generative AI:** Google GenAI SDK (Gemini 2.5 Flash)
* **Data Engine:** Polars (Rust-based DataFrame library processing 25M rows)

---

## 🚀 Key Differentiators (Why This Is Tier-1)

| Feature | Standard Student Project | ScaleRec Architecture |
| :--- | :--- | :--- |
| **Model Type** | Matrix Factorization (Static) | **Two-Stage:** Sequential Transformer + Deep Cross Network |
| **Latency Strategy** | Scores all items (Slow) | FAISS Retrieval (Top 100) → DCN Ranking (Top 10) |
| **Cold Start** | Fails on new items | Solved via dense Genome Semantic embeddings |
| **Temporal Data** | Model is stuck in the past | Solved via **Gemini 2.5 LLM Agent** for real-time trends |
| **UI/UX** | Jupyter Notebook | Containerized Web App with Auth & Dynamic Placeholders |

---

## ⚡ Quick Start (Docker)

Run the entire full-stack application with one command.

**Prerequisites:** Docker & Docker Compose

1. **Set your API Key:** Ensure your `docker-compose.yml` or local `.env` has your `GEMINI_API_KEY`.
2. **Run the App:**
   ```bash
   docker-compose up --build
   ```

3. **Access the Application:**
* Frontend Dashboard: `http://localhost:8501`
* Backend API Docs: `http://localhost:8000/docs`
* *Default Login - Username: `admin` | Password: `recsys123` (Or register a new account)*

---

## 🛠️ Local Development (Manual Setup)

**1. Environment Setup**

```bash
conda create -n recsys python=3.9 -y
conda activate recsys
pip install -r requirements.txt
export GEMINI_API_KEY="your_api_key_here"
```

**2. Data Pipeline (Polars)**
Process the MovieLens 25M dataset.

```bash
python src/01_preprocess.py
python src/02_process_genome.py
```

**3. Model Training (Two-Stage Pipeline)**

```bash
# Train Stage 1: SASRec Transformer (Retrieval)
python src/03_train.py

# Train Stage 2: Deep & Cross Network (Ranking)
python src/03b_train_ranker.py
```

**4. Run Services**
Open two terminals to launch the decoupled microservices:

```bash
# Terminal 1 (Backend)
python src/04_inference.py

# Terminal 2 (Frontend)
streamlit run src/frontend.py
```

---

## 📸 Features Showcase

* **Netflix-Tier Dashboard:** Interactive tabs, hover-scaling cards, and dynamic color-hash placeholders for missing posters.
* **Agentic Chat UI:** A dedicated interface for the PopGuru AI, featuring conversational memory and actionable Markdown routing.
* **Deep Linked Metadata:** Movies link directly to IMDb databases dynamically.
* **Secure Auth:** Session-state user registration and login.

---

## 📊 Performance Metrics

* **Dataset:** MovieLens 25M (25 million interactions)
* **Stage 1 (SASRec) Accuracy:** ~97.7% (Next-item prediction)
* **Stage 2 (DCN-V2) Optimization:** Root Mean Squared Error (RMSE) reduction on predicted star ratings.
* **End-to-End Latency:** `< 50ms` (FAISS Search + Neural Ranking)

---

### 👨‍💻 Author

**Arjun** *AI Engineer | ML Systems Architect*
* GitHub: [Arjun-Patel1](https://github.com/Arjun-Patel1)
* LinkedIn: [arjunpatel97259](https://www.linkedin.com/in/arjunpatel97259)