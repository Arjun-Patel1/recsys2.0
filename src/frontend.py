import streamlit as st
import requests
import polars as pl
import os
import json
import config

# --- CONFIGURATION ---
USER_DB_FILE = "users.json"
# Use localhost for local dev, or the docker service name if in container
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/recommend")

# --- 1. AUTHENTICATION SYSTEM ---
def load_users():
    if os.path.exists(USER_DB_FILE):
        try:
            with open(USER_DB_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_user(username, password):
    users = load_users()
    users[username] = password
    with open(USER_DB_FILE, "w") as f:
        json.dump(users, f)

def auth_page():
    st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] { gap: 10px; }
        .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #0e1117; border-radius: 5px; }
        .auth-container { padding: 20px; border: 1px solid #333; border-radius: 10px; background-color: #161b22; }
        </style>
    """, unsafe_allow_html=True)

    st.title("🎬 ScaleRec Elite")
    st.caption("Tier-1 Recommendation Engine | Powered by SASRec & FAISS")

    tab1, tab2 = st.tabs(["🔒 Login", "📝 Register"])

    with tab1:
        with st.form("login_form"):
            user = st.text_input("Username")
            pw = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log In", type="primary")
            
            if submitted:
                users = load_users()
                if user in users and users[user] == pw:
                    st.session_state.logged_in = True
                    st.session_state.username = user # Set username here
                    st.rerun()
                else:
                    st.error("Invalid Username or Password")

    with tab2:
        with st.form("register_form"):
            new_user = st.text_input("Choose Username")
            new_pw = st.text_input("Choose Password", type="password")
            confirm_pw = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Create Account")

            if submitted:
                if new_pw != confirm_pw:
                    st.error("Passwords do not match!")
                elif new_user in load_users():
                    st.error("User already exists!")
                elif len(new_pw) < 4:
                    st.error("Password must be at least 4 characters")
                else:
                    save_user(new_user, new_pw)
                    st.success("Account Created! You can now log in.")

# --- SESSION STATE INITIALIZATION (THE FIX) ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = "Guest"

if not st.session_state.logged_in:
    auth_page()
    st.stop()

# --- 2. DATA LOADING (Movies + IMDb Links) ---
@st.cache_data
def load_data():
    # Load Movies
    df_movies = pl.read_csv(config.MOVIES_PATH)
    
    # Load Links (for IMDb IDs)
    # We force IDs to be strings to preserve leading zeros
    df_links = pl.read_csv(config.LINKS_PATH, dtypes={"imdbId": pl.Utf8})
    
    # Join them
    df_full = df_movies.join(df_links, on="movieId", how="left")
    
    # Create lookups
    title_map = dict(zip(df_full["title"], df_full["movieId"]))
    
    meta_map = {}
    # Use iter_rows to safely create dictionary
    for row in df_full.iter_rows(named=True):
        meta_map[row["movieId"]] = {
            "title": row["title"],
            "genres": row["genres"],
            "imdbId": row["imdbId"]
        }
    return title_map, meta_map

movie_to_id, movie_meta = load_data()

# --- 3. UI: MAIN DASHBOARD ---
st.set_page_config(page_title="ScaleRec Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Interactive Cards
st.markdown("""
<style>
    /* Card Container */
    .movie-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 15px;
        height: 280px; /* Fixed Height */
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.2s, box-shadow 0.2s;
        text-decoration: none !important;
        color: white !important;
        margin-bottom: 20px;
    }
    
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.5);
        border-color: #6366f1;
    }

    /* Typography */
    .movie-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 5px;
        background: -webkit-linear-gradient(#fff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .movie-genre {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-bottom: 10px;
    }
    .match-score {
        background-color: #22c55e;
        color: black;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: bold;
        width: fit-content;
    }
    .imdb-btn {
        background-color: #f5c518;
        color: black;
        text-align: center;
        padding: 8px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.9rem;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("👤 User Profile")
    st.markdown(f"**Welcome, {st.session_state.username}**")
    
    if st.button("Logout", type="primary"):
        st.session_state.logged_in = False
        st.session_state.username = "Guest" # Reset username
        st.rerun()
        
    st.divider()
    st.subheader("🕑 Watch History")
    
    # Pre-select some interesting movies
    default_opts = ["Toy Story (1995)", "Jumanji (1995)"]
    valid_defaults = [m for m in default_opts if m in movie_to_id]
    
    selected_movies = st.multiselect(
        "What have you watched?",
        options=list(movie_to_id.keys()),
        default=valid_defaults
    )
    
    k_items = st.slider("Recommendations", 4, 12, 8)

# Main Area
st.title("🍿 Your Personal Recommendations")
st.markdown("Our **Transformer Model** analyzes your sequence to predict what you'll watch next.")

if st.button("🚀 Generate Feed", type="primary"):
    if not selected_movies:
        st.warning("Please add movies to your history first.")
    else:
        # Prepare Request
        history_ids = [movie_to_id[m] for m in selected_movies]
        
        with st.spinner("Processing embeddings..."):
            try:
                resp = requests.post(
                    API_URL, 
                    json={"user_history_ids": history_ids, "k": k_items},
                    timeout=5
                )
                
                if resp.status_code == 200:
                    data = resp.json()["recommendations"]
                    
                    # RENDER THE GRID
                    cols = st.columns(4)
                    for i, item in enumerate(data):
                        with cols[i % 4]:
                            # Get Metadata
                            mid = item["movie_id"]
                            meta = movie_meta.get(mid, {})
                            imdb_id = meta.get("imdbId")
                            genre = meta.get("genres", "Unknown Genre").replace("|", " • ")
                            
                            # Build Link
                            if imdb_id:
                                # Ensure we don't have double tt
                                clean_id = str(imdb_id).replace("tt", "")
                                link = f"https://www.imdb.com/title/tt{clean_id}/"
                            else:
                                link = "#"

                            score_pct = int(item["score"] * 100)
                            
                            # HTML CARD
                            st.markdown(f"""
                            <a href="{link}" target="_blank" style="text-decoration: none;">
                                <div class="movie-card">
                                    <div>
                                        <div class="match-score">{score_pct}% Match</div>
                                        <div class="movie-title">{item['title']}</div>
                                        <div class="movie-genre">{genre}</div>
                                    </div>
                                    <div class="imdb-btn">IMDb ↗</div>
                                </div>
                            </a>
                            """, unsafe_allow_html=True)
                            
                else:
                    st.error(f"Backend Error: {resp.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Backend is offline. Run `python src/04_inference.py`")