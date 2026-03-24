
# 🎬 ScaleRec: Tier-1 AI Recommendation Engine

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg?logo=tensorflow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi)
![Django](https://img.shields.io/badge/Django-5.0+-092E20.svg?logo=django)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg?logo=docker)

ScaleRec is a decoupled, production-ready machine learning application that demonstrates a modern recommendation pipeline. It separates a GPU-intensive neural network inference layer from a high-performance, user-facing web dashboard.

---

## 💻 Tech Stack

**Machine Learning & Data Processing**
* **TensorFlow & TensorFlow Recommenders (TFRS):** Core framework for sequence modeling and ranking.
* **SASRec:** Self-Attentive Sequential Recommendation model used to capture temporal user watch history.
* **DCN-V2:** Deep & Cross Network V2 used to predict accurate click-through/match rates.
* **FAISS (Facebook AI Similarity Search):** High-speed vector database for sub-millisecond candidate retrieval.
* **Polars:** Lightning-fast DataFrame library utilized over Pandas for optimized dataset loading.

**Backend Architecture (Microservice)**
* **FastAPI:** Asynchronous, high-performance REST API handling the ML inference layer.
* **Uvicorn:** ASGI web server for running the FastAPI application.
* **Pydantic:** Strict data validation ensuring the frontend only passes clean, tensor-ready arrays.

**Frontend Web Layer**
* **Django (Python):** Enterprise-grade framework handling routing, session security, and CSRF protection.
* **LocMemCache:** Django's built-in RAM caching API used to hold 60,000+ row datasets in memory, achieving zero-latency rendering and bypassing disk I/O bottlenecks.
* **HTML5 / CSS3 / JavaScript:** Custom-built "Three-Pane Workspace" UI utilizing CSS Flexbox, Grid, and Glassmorphism (No heavy UI frameworks like React/Vue needed).

**Generative AI & LLMs**
* **Google Gemini 2.5 Flash:** Integrated as an "always-on" Copilot agent to bypass static dataset limitations and provide real-time reviews, trending insights, and dynamic streaming links.

**DevOps & Deployment**
* **Docker & Docker Compose:** Containerized the decoupled frontend and backend services into isolated environments with virtual internal networking.

---

## 🏗️ System Architecture

The system is built on a microservice architecture to ensure independent scalability of the web and inference layers:

1. **The Recommender Core:** The user's sequence is passed through the SASRec model to generate a latent vector. This vector hits the FAISS index to retrieve the top 100 semantic candidates. 
2. **The Ranker:** The DCN-V2 ranker evaluates the top 100 candidates against the user's profile and returns the top *k* results with exact confidence scores.
3. **The Web Layer:** The Django application serves the Three-Pane Engineering UI. It fetches the results via an internal API call and dynamically generates actionable IMDb links and UI posters based on the model's output.

---

## 🚀 Quick Start (Docker)

The absolute easiest way to run ScaleRec is using Docker Compose.

### Prerequisites
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.
* A Google Gemini API Key.

### 1. Clone & Configure
Clone the repository and enter the directory:
```bash
git clone [https://github.com/yourusername/ScaleRec.git](https://github.com/yourusername/ScaleRec.git)
cd ScaleRec
````

Create a `.env` file in the root directory and add your Gemini API Key:

```env
GEMINI_API_KEY=your_actual_api_key_here
```

### 2\. Build and Launch

Run the following command to build the images and spin up the microservices:

```bash
docker compose up --build
```

### 3\. Access the Application

Once the containers are running and the cache is initialized, navigate to:

  * **Engineering Dashboard:** [http://localhost:8000](https://www.google.com/search?q=http://localhost:8000)
  * **Backend API Docs (Swagger UI):** [http://localhost:8001/docs](https://www.google.com/search?q=http://localhost:8001/docs)

-----

## 💻 Local Development (Without Docker)

If you prefer to run the services manually in a local Python environment:

**1. Set the PYTHONPATH**
Ensure Python knows where the `config.py` file is located:

```bash
# Windows
set PYTHONPATH=.
# Mac/Linux
export PYTHONPATH=.
```

**2. Start the FastAPI Backend (Port 8001)**

```bash
uvicorn src.04_inference:app --host 0.0.0.0 --port 8001
```

**3. Start the Django Frontend (Port 8000)**
Open a new terminal, activate your environment, set `PYTHONPATH=.` again, and run:

```bash
python manage.py runserver
```

-----

## 📁 Repository Structure

```text
MOVIE_rec_sys/
├── docker-compose.yml       # Multi-container orchestration
├── Dockerfile.backend       # FastAPI inference container setup
├── Dockerfile.frontend      # Django web container setup
├── requirements.txt         # Global dependencies
├── config.py                # Global path configurations
├── src/                     
│   ├── models.py            # TF Neural Network architectures
│   └── 04_inference.py      # FastAPI routing & FAISS logic
├── webapp/                  
│   ├── settings.py          # Django config (LocMemCache setup)
│   └── urls.py              
├── frontend/                
│   ├── views.py             # View logic & Backend API requests
│   └── templates/           # Three-pane UI, Profile, and About pages
└── processed_data/          # FAISS indices, datasets, and .h5 weights
```

## 🤝 License

This project is open-source and available under the [MIT License](https://www.google.com/search?q=LICENSE).

```

This acts as a perfect summary of all the hard work you've put in. The architecture description combined with the precise tech stack breakdown makes it immediately obvious to any technical recruiter that you are building at an enterprise level!
```
