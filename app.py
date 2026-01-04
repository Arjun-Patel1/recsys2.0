import os
import pickle
import requests
import numpy as np
import pandas as pd
import streamlit as st

from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="🎬 Hybrid Movie Recommender",
    layout="wide",
    initial_sidebar_state="expanded"
)

TMDB_API_KEY = ""  # INSERT YOUR TMDB API KEY HERE🔑 
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w300"
TMDB_MOVIE_BASE = "https://www.themoviedb.org/movie"

# ---------------- Paths (10K mini dataset) ----------------
DATA_DIR = "data"

RATINGS_PATH = f"C:/Users/arjun/Downloads/mini_rec_sys/data/ratings_100.csv"
MOVIES_PATH = f"C:/Users/arjun/Downloads/mini_rec_sys/data/movies_100.csv"
LINKS_PATH = f"C:/Users/arjun/Downloads/mini_rec_sys/data/links_100.csv"
GENOME_PATH = f"C:/Users/arjun/Downloads/mini_rec_sys/data/genome_scores_100.csv"

ALS_MODEL_PKL = "C:/Users/arjun/Downloads/mini_rec_sys/artifacts/als_model.pkl"
USER_ITEM_NPZ = "C:/Users/arjun/Downloads/mini_rec_sys/artifacts/user_item_matrix.npz"

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data(show_spinner=False)
def load_data():
    ratings = pd.read_csv(RATINGS_PATH)
    movies = pd.read_csv(MOVIES_PATH)
    links = pd.read_csv(LINKS_PATH)
    genome = pd.read_csv(GENOME_PATH)

    movie_titles = dict(zip(movies.movieId, movies.title))
    movieid_to_tmdb = dict(zip(links.movieId, links.tmdbId))

    # ----- Content features -----
    tag_matrix = genome.pivot(
        index="movieId",
        columns="tagId",
        values="relevance"
    ).fillna(0)

    content_movie_ids = tag_matrix.index.to_numpy()
    content_matrix = normalize(tag_matrix.values)

    movieid_to_content_idx = {
        mid: i for i, mid in enumerate(content_movie_ids)
    }

    return (
        ratings,
        movie_titles,
        movieid_to_tmdb,
        content_matrix,
        content_movie_ids,
        movieid_to_content_idx
    )

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource(show_spinner=False)
def load_model():
    with open(ALS_MODEL_PKL, "rb") as f:
        als = pickle.load(f)

    mat = np.load(USER_ITEM_NPZ)
    user_item = csr_matrix(
        (mat["data"], mat["indices"], mat["indptr"]),
        shape=mat["shape"]
    )

    return als, user_item

ratings, movie_titles, movieid_to_tmdb, content_matrix, content_movie_ids, movieid_to_content_idx = load_data()
als_model, user_item_matrix = load_model()

st.success("✅ Models & data loaded successfully")

# =====================================================
# TMDB POSTER (ID-BASED — RELIABLE)
# =====================================================
@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):
    tmdb_id = movieid_to_tmdb.get(movie_id)

    if pd.isna(tmdb_id):
        return None, None

    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}",
            params={"api_key": TMDB_API_KEY},
            timeout=5
        ).json()

        if r.get("poster_path"):
            poster = f"{TMDB_IMAGE_BASE}{r['poster_path']}"
            link = f"{TMDB_MOVIE_BASE}/{int(tmdb_id)}"
            return poster, link
    except:
        pass

    return None, None

# =====================================================
# RECOMMENDATION LOGIC
# =====================================================
def content_recommend(movie_id, top_n=10):
    idx = movieid_to_content_idx.get(movie_id)
    if idx is None:
        return []

    sims = cosine_similarity(
        content_matrix[idx:idx + 1],
        content_matrix
    )[0]

    top_idx = np.argsort(sims)[::-1][1:top_n + 1]
    return [(content_movie_ids[i], float(sims[i])) for i in top_idx]

def hybrid_recommend(user_idx, top_n=10):
    als_idx, als_scores = als_model.recommend(
        user_idx,
        user_item_matrix[user_idx],
        N=50
    )

    als_dict = {
        content_movie_ids[i] if i < len(content_movie_ids) else i: float(s)
        for i, s in zip(als_idx, als_scores)
    }

    liked = ratings[
        (ratings.userId == user_idx + 1) & (ratings.rating >= 4)
    ]

    if liked.empty:
        return sorted(als_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]

    seed_movie = liked.iloc[0].movieId
    content_dict = dict(content_recommend(seed_movie, 50))

    hybrid = {
        m: 0.7 * als_dict.get(m, 0) + 0.3 * content_dict.get(m, 0)
        for m in set(als_dict) | set(content_dict)
    }

    return sorted(hybrid.items(), key=lambda x: x[1], reverse=True)[:top_n]

# =====================================================
# UI
# =====================================================
st.title("🎬 Hybrid Movie Recommender")
st.caption("ALS + Content-Based | MovieLens + TMDB")

user_id = st.sidebar.number_input(
    "Enter User ID",
    min_value=1,
    max_value=int(ratings.userId.max()),
    value=10
)

num_recs = st.sidebar.slider("Recommendations", 5, 20, 10)

if st.sidebar.button("✨ Get Recommendations"):
    user_idx = int(user_id) - 1
    recs = hybrid_recommend(user_idx, num_recs)

    st.subheader("🔥 Hybrid Recommendations")
    cols = st.columns(5)

    for i, (mid, score) in enumerate(recs):
        title = movie_titles.get(mid, "Unknown")
        poster, link = fetch_poster(mid)

        with cols[i % 5]:
            if poster:
                st.image(poster, width=180)
            st.markdown(f"**[{title}]({link if link else '#'})**")
            st.caption(f"⭐ Score: {score:.3f}")
