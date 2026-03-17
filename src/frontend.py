# File: src/frontend.py
"""run this 
pip uninstall google-generativeai -y
pip install google-genai
"""
import streamlit as st
import requests
import polars as pl
import os
import json
import urllib.parse
import re
import google.generativeai as genai
import config
from dotenv import load_dotenv  # <--- ADD THIS

# --- CONFIGURATION ---
# Tell Python to load variables from the .env file if it exists
load_dotenv()  # <--- ADD THIS

USER_DB_FILE = "users.json"
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/recommend")

# --- GEMINI SETUP ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    # This prevents the app from crashing and tells you exactly what went wrong
    st.error(
        "🚨 CRITICAL: GEMINI_API_KEY environment variable is not set! Check your .env file."
    )
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel("gemini-2.5-flash")


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
    st.markdown(
        """
        <style>
        .stApp { background-color: #000000; color: white; }
        .stTabs [data-baseweb="tab-list"] { gap: 10px; }
        .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #141414; border-radius: 4px; color: #b3b3b3; }
        .stTabs [aria-selected="true"] { color: white !important; border-bottom: 3px solid #E50914 !important; }
        .stButton>button { background-color: #E50914 !important; color: white !important; border: none; font-weight: bold; border-radius: 4px; }
        .stButton>button:hover { background-color: #f40612 !important; }
        </style>
    """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            "<h1 style='color: #E50914; text-align: center; font-weight: 900;'>ScaleRec</h1>",
            unsafe_allow_html=True,
        )
        st.caption(
            "<div style='text-align: center; color: #b3b3b3; margin-bottom: 20px;'>Sign In to your AI Cinema</div>",
            unsafe_allow_html=True,
        )

        tab1, tab2 = st.tabs(["🔒 Sign In", "📝 Sign Up"])

        with tab1:
            with st.form("login_form"):
                user = st.text_input("Username")
                pw = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign In", use_container_width=True)

                if submitted:
                    users = load_users()
                    if user in users and users[user] == pw:
                        st.session_state.logged_in = True
                        st.session_state.username = user
                        st.rerun()
                    else:
                        st.error("Invalid Username or Password")

        with tab2:
            with st.form("register_form"):
                new_user = st.text_input("Choose Username")
                new_pw = st.text_input("Choose Password", type="password")
                confirm_pw = st.text_input("Confirm Password", type="password")
                submitted = st.form_submit_button("Sign Up", use_container_width=True)

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


# --- SESSION STATE INITIALIZATION ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = "Guest"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if not st.session_state.logged_in:
    auth_page()
    st.stop()


# --- 2. DATA LOADING ---
@st.cache_data
def load_data():
    df_movies = pl.read_csv(config.MOVIES_PATH)
    df_links = pl.read_csv(config.LINKS_PATH, dtypes={"imdbId": pl.Utf8})
    df_full = df_movies.join(df_links, on="movieId", how="left")

    title_map = dict(zip(df_full["title"], df_full["movieId"]))
    meta_map = {}
    for row in df_full.iter_rows(named=True):
        meta_map[row["movieId"]] = {
            "title": row["title"],
            "genres": row["genres"],
            "imdbId": row["imdbId"],
        }
    return title_map, meta_map


movie_to_id, movie_meta = load_data()

# --- 3. UI: MAIN DASHBOARD ---
st.set_page_config(
    page_title="ScaleRec | Netflix Mode",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #000; }
    ::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #E50914; }

    .stApp { background-color: #141414; color: #ffffff; }
    
    .stButton>button { background-color: white; color: black; font-weight: bold; border-radius: 4px; border: none; transition: 0.2s; }
    .stButton>button:hover { background-color: #e6e6e6; color: black; }
    div[data-testid="stSidebar"] { background-color: #000000; border-right: 1px solid #333; }
    div[data-testid="stSidebar"] .stButton>button { background-color: transparent; color: #b3b3b3; border: 1px solid #b3b3b3; }
    div[data-testid="stSidebar"] .stButton>button:hover { color: white; border-color: white; }

    .stTabs [data-baseweb="tab-list"] { gap: 24px; padding-bottom: 10px; }
    .stTabs [data-baseweb="tab"] { font-size: 1.2rem; font-weight: 600; color: #808080; background-color: transparent; border: none; padding: 10px 0; }
    .stTabs [aria-selected="true"] { color: #fff !important; border-bottom: 3px solid #E50914 !important; background-color: transparent !important; }

    .netflix-card {
        border-radius: 6px;
        height: 350px;
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
        cursor: pointer;
        transition: transform 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94), box-shadow 0.3s;
        box-shadow: 0 4px 8px rgba(0,0,0,0.5);
    }
    .netflix-card:hover {
        transform: scale(1.06);
        z-index: 10;
        box-shadow: 0 10px 25px rgba(229, 9, 20, 0.4);
        border: 1px solid #E50914;
    }
    
    .netflix-card-overlay {
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 65%;
        background: linear-gradient(to top, rgba(20,20,20,1) 0%, rgba(20,20,20,0.85) 45%, rgba(20,20,20,0) 100%);
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        padding: 15px;
    }

    .match-score { color: #46d369; font-weight: 800; font-size: 0.9rem; margin-bottom: 4px; text-shadow: 1px 1px 2px #000; }
    .movie-title { font-size: 1.15rem; font-weight: 800; text-shadow: 1px 1px 3px black; line-height: 1.2; margin-bottom: 4px; color: white;}
    .movie-genre { font-size: 0.75rem; color: #cccccc; }
    .imdb-badge { background-color: #f5c518; color: black; padding: 2px 6px; border-radius: 3px; font-weight: 900; font-size: 0.7rem; float: right; box-shadow: 0 2px 4px rgba(0,0,0,0.5); }
    
</style>
""",
    unsafe_allow_html=True,
)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown(
        f"<h2 style='color: #E50914; font-weight: 900; margin-bottom: 0;'>ScaleRec</h2>",
        unsafe_allow_html=True,
    )
    st.caption(f"Logged in as: **{st.session_state.username}**")

    if st.button("Sign Out", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = "Guest"
        st.rerun()

    st.divider()
    st.subheader("🕑 My Watch History")

    default_opts = ["Toy Story (1995)", "Jumanji (1995)"]
    valid_defaults = [m for m in default_opts if m in movie_to_id]

    selected_movies = st.multiselect(
        "Titles you've watched:",
        options=list(movie_to_id.keys()),
        default=valid_defaults,
    )

    k_items = st.slider("Rows to load", 4, 20, 8, step=4)

# --- MAIN DASHBOARD (TABBED INTERFACE) ---
tab_feed, tab_bot = st.tabs(["🍿 AI Recommendations", "✨ PopGuru Agent"])

# ==========================================
# TAB 1: NEURAL NETWORK RECOMMENDATIONS
# ==========================================
with tab_feed:
    st.markdown(
        "<h2 style='font-weight: 800; margin-bottom: 0;'>Top Picks for You</h2>",
        unsafe_allow_html=True,
    )
    st.caption("Curated by Two-Stage Neural Network (SASRec + DCN-V2)")
    st.write("")  # Spacer

    if st.button("▶ Generate Feed", use_container_width=True):
        if not selected_movies:
            st.warning("Please add movies to your history in the sidebar first.")
        else:
            history_ids = [movie_to_id[m] for m in selected_movies]

            with st.spinner("Analyzing neural sequences and ranking candidates..."):
                try:
                    resp = requests.post(
                        API_URL,
                        json={"user_history_ids": history_ids, "k": k_items},
                        timeout=10,
                    )

                    if resp.status_code == 200:
                        data = resp.json()["recommendations"]

                        st.write("")
                        cols = st.columns(4)

                        for i, item in enumerate(data):
                            with cols[i % 4]:
                                mid = item["movie_id"]
                                meta = movie_meta.get(mid, {})
                                imdb_id = meta.get("imdbId")
                                raw_genre = meta.get("genres", "Unknown")
                                genre = (
                                    raw_genre.replace("|", " • ")
                                    if raw_genre != "(no genres listed)"
                                    else "Drama"
                                )

                                if imdb_id:
                                    clean_id = str(imdb_id).replace("tt", "")
                                    link = f"https://www.imdb.com/title/tt{clean_id}/"
                                else:
                                    link = "#"

                                rating = max(0.5, min(5.0, float(item["score"])))

                                clean_title = re.sub(
                                    r"[^a-zA-Z0-9 ]", "", item["title"].split(" (")[0]
                                )
                                url_title = urllib.parse.quote(clean_title)
                                poster_url = f"https://placehold.co/500x750/111111/E50914/png?text={url_title}"

                                color_hash = sum(ord(c) for c in clean_title) % 360
                                fallback_color = f"hsl({color_hash}, 20%, 15%)"

                                st.markdown(
                                    f"""
                                <a href="{link}" target="_blank" style="text-decoration: none; color: inherit;">
                                    <div class="netflix-card" style="background-color: {fallback_color}; background-image: url('{poster_url}'); background-size: cover; background-position: center;">
                                        <div class="netflix-card-overlay">
                                            <div>
                                                <span class="match-score">{rating:.1f} ★ Match</span>
                                                <span class="imdb-badge">IMDb</span>
                                            </div>
                                            <div class="movie-title">{item['title']}</div>
                                            <div class="movie-genre">{genre}</div>
                                        </div>
                                    </div>
                                </a>
                                """,
                                    unsafe_allow_html=True,
                                )

                    else:
                        st.error(f"Backend Error: {resp.text}")

                except requests.exceptions.ConnectionError:
                    st.error(
                        "Backend is offline. Ensure your FastAPI server is running (`python src/04_inference.py`)."
                    )

# ==========================================
# TAB 2: POPGURU LLM AGENT
# ==========================================
with tab_bot:
    st.markdown(
        "<h2 style='font-weight: 800; margin-bottom: 0;'>PopGuru: AI Entertainment Insider</h2>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Powered by Gemini 2.5 Flash API — Ask about the latest blockbusters, ratings, trending reels, or box office hits!"
    )

    # --- UPDATED: ACTIONABLE AI SYSTEM PROMPT ---
    SYSTEM_PROMPT = """
    You are 'PopGuru', an elite, trendy AI entertainment expert. You specialize in Indian and Global pop culture, including Bollywood, Tollywood, Hollywood movies, trending Spotify music, viral Instagram reels, actor gossips, reviews, and box office ratings. You know the absolute latest releases (2024, 2025, 2026).

    CRITICAL GUARDRAILS & ACTIONABLE LINKS:
    1. DOMAIN STRICTNESS: You MUST ONLY answer questions related to entertainment, media, movies, actors, music, and pop culture. Strictly decline anything else.
    2. TONE: Use a fun, trendy tone. Use a bit of Hinglish if it fits naturally, but remain professional.
    3. ACTIONABLE LINKS (MANDATORY): Whenever you recommend a movie, song, or trailer, you MUST provide a clickable Markdown link so the user can take action immediately. 
       - For movies IN THEATERS right now: Provide a BookMyShow link formatted like `[🎟️ Book Tickets on BookMyShow](https://in.bookmyshow.com/explore/movies)`
       - For STREAMING movies: Provide a JustWatch search link formatted like `[🍿 Find where to stream on JustWatch](https://www.justwatch.com/in/search?q=<Movie_Name_Here>)` or a direct platform link if you are 100% sure.
       - For SONGS or TRAILERS: Provide a YouTube search link formatted like `[🎵 Listen on YouTube](https://www.youtube.com/results?search_query=<Song_or_Trailer_Name_Here>)`
    Make sure your links stand out on separate lines!
    """

    chat_container = st.container(height=450)
    for message in st.session_state.chat_history:
        with chat_container.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(
        "Ask PopGuru (e.g., 'What are the reviews for the latest action movie?')...",
        key="popguru_chat",
    ):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with chat_container.chat_message("user"):
            st.markdown(prompt)

        with chat_container.chat_message("assistant"):
            with st.spinner("PopGuru is thinking..."):
                try:
                    full_prompt = f"{SYSTEM_PROMPT}\n\nUser Query: {prompt}"
                    response = model_gemini.generate_content(full_prompt)

                    st.markdown(response.text)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": response.text}
                    )
                except Exception as e:
                    st.error(
                        f"⚠️ API Error. Ensure your Gemini API Key is valid. Details: {e}"
                    )
