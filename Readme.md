# 🎬 ScaleRec: Full-Stack Real-Time Recommendation Engine
https://hub.docker.com/repository/docker/arjunpatel89806/movie-recsys-api
https://hub.docker.com/repository/docker/arjunpatel89806/movie-recsys-app
ScaleRec is a production-grade recommendation system designed to mimic the Candidate Generation phase of modern streaming platforms like Netflix and YouTube.

Unlike traditional static recommenders (Matrix Factorization), ScaleRec is session-aware. It uses a Transformer-based architecture (SASRec) to understand the sequence of user actions, enriched with semantic Genome Tags to solve the cold-start problem.

The system is deployed as a full-stack microservices application with secure authentication, an interactive dashboard, and deep-linked IMDb metadata.

---

## 🏗️ System Architecture

ScaleRec follows a Two-Tower Architecture optimized for low-latency retrieval (<10ms).

User / Client  
↓ Login & History  
Streamlit Frontend  
↓ JSON Payload  
FastAPI Backend  
→ Vector Search → FAISS Index  
→ Model Inference → SASRec Transformer  
→ Semantic Enrichment → Genome Tags (PCA)  
→ Deep Links → IMDb  

---

## 🧰 Tech Stack

Frontend  
- Streamlit (Custom CSS, State Management, Authentication)

Backend  
- FastAPI (Async REST APIs)

Model  
- TensorFlow 2.x  
- SASRec (Self-Attentive Sequential Recommender)

Vector Search  
- FAISS (HNSW, Approximate Nearest Neighbor Search)

Data Engine  
- Polars (Rust-based DataFrame library)  
- Processes 25M+ interactions

---

## 🚀 Key Differentiators (Why This Is Tier-1)

| Feature | Standard Student Project | ScaleRec |
|------|----------------|----------------|
| Model Type | Matrix Factorization (Static) | Sequential Transformer (Dynamic) |
| Context | Ignores order | Learns behavior sequence (A → B → C) |
| New Items | Fails (Cold Start) | Solved via Genome Semantics |
| UI | Notebook / Terminal | Full Web App with Login |
| Metadata | Text only | Clickable IMDb-linked cards |
| Deployment | Local scripts | Dockerized microservices |

---

## ⚡ Quick Start (Docker)

Run the entire full-stack application with one command.

Prerequisites  
- Docker & Docker Compose  
- (Optional) 4GB+ RAM allocated to Docker

### Run the App

---
docker-compose up --build

### Access the Application
- Frontend Dashboard: http://localhost:8501  
- Backend API Docs: http://localhost:8000/docs  

Default Login  

Username: admin
Password: recsys123
(Or register a new account)

---

## 🛠️ Local Development (Manual Setup)

### Environment Setup

conda create -n recsys python=3.9 -y
conda activate recsys
pip install -r requirements.txt

---

### Data Pipeline (Polars)

Process the MovieLens 25M dataset.


python src/01_preprocess.py
python src/02_process_genome.py

---

### Model Training

Train the SASRec Transformer.


python src/03_train.py

Achieved ~97.7% Top-K Recall on next-item prediction.

---

### Run Services

Open two terminals.

Terminal 1 (Backend)

python src/04_inference.py

Terminal 2 (Frontend)

streamlit run src/frontend.py

## 📸 Features Showcase

Secure Authentication  
- User registration and login  
- User data persisted in a JSON store  

Interactive Dashboard  
- Dark-mode UI with Netflix-style movie cards  
- Hover effects and Match Score indicators  

Smart Metadata  
- Movies linked directly to IMDb  
- Uses links.csv to map MovieLens IDs to external databases  

---

## 📊 Performance Metrics

- Dataset: MovieLens 25M (25 million interactions)
- Validation Accuracy: ~97.7% (Next-item prediction)
- Inference Latency: < 15ms per request
- Retrieval Engine: FAISS HNSW Index

---


## Author

**Arjun**  
AI Engineer | ML Engineer

GitHub: https://github.com/Arjun-Patel1

LinkedIn: https://www.linkedin.com/in/arjunpatel97259

