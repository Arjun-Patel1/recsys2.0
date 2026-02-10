# File: src/frontend.py
import streamlit as st
import requests
import polars as pl
import config

# --- Configuration ---
import os

# ... existing imports ...

# UPDATED CONFIGURATION
# If running in Docker, use the environment variable. Otherwise default to localhost.
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/recommend")
#API_URL = "http://127.0.0.1:8000/recommend"
st.set_page_config(page_title="ScaleRec Movie System", layout="wide")

# --- Load Data (Cached for Speed) ---
@st.cache_data
def load_movies():
    # We need the map to convert Title -> ID for the API
    # and ID -> Title for display
    df_movies = pl.read_csv(config.MOVIES_PATH)
    # Create a dictionary for quick lookup: Title -> ID
    movie_dict = dict(zip(df_movies["title"], df_movies["movieId"]))
    return movie_dict

movie_map = load_movies()

# --- Sidebar: User Controls ---
st.sidebar.title("🎬 ScaleRec Control")
st.sidebar.write("Simulate a user's watch history here.")

# Multiselect for Watch History
selected_movies = st.sidebar.multiselect(
    "Select movies you 'watched':",
    options=list(movie_map.keys()),
    default=["Toy Story (1995)", "Jumanji (1995)"] # Default example
)

# Number of recommendations
k_items = st.sidebar.slider("How many recommendations?", 5, 20, 10)

# --- Main Page ---
st.title("🍿 ScaleRec: Tier-1 Recommendation Engine")
st.markdown("""
This interface is a **Streamlit Frontend** connected to a **FastAPI Backend**.  
It uses a **Two-Tower Neural Network (SASRec)** + **Genome Embeddings** to find movies semantically related to your sequence.
""")

st.divider()

# --- Logic ---
if st.button("🚀 Get Recommendations", type="primary"):
    if not selected_movies:
        st.warning("Please select at least one movie in the sidebar!")
    else:
        # 1. Convert Titles to IDs
        history_ids = [movie_map[title] for title in selected_movies]
        
        # 2. Prepare Payload
        payload = {
            "user_history_ids": history_ids,
            "k": k_items
        }
        
        # 3. Call Backend API
        with st.spinner("Connecting to Neural Engine..."):
            try:
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    recs = data["recommendations"]
                    
                    st.success(f"Found {len(recs)} recommendations based on {len(history_ids)} interactions.")
                    
                    # 4. Display Results in a Grid
                    cols = st.columns(3) # 3 columns grid
                    for i, rec in enumerate(recs):
                        col = cols[i % 3] # Cycle through columns
                        with col:
                            # Card-like display
                            st.markdown(f"### 🎥 {rec['title']}")
                            st.caption(f"Confidence Score: {rec['score']:.4f}")
                            st.progress(min(rec['score'], 1.0)) # Visual bar for score
                            st.markdown("---")
                            
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("🚨 Connection Failed! Is the FastAPI Backend running?")
                st.info("Run `python src/04_inference.py` in a separate terminal.")

# --- Debugging Info (Optional) ---
with st.expander("🛠️ Developer Logs"):
    st.write("Selected IDs:", [movie_map[m] for m in selected_movies])
    st.write("API Endpoint:", API_URL)