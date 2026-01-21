# app.py
import streamlit as st
import requests
API_URL = "http://api:8000/recommend"

#local run API_URL = "http://127.0.0.1:8000/recommend"

st.set_page_config(page_title="Movie Recommendation System")

st.title("🎬 Movie Recommendation System")
st.caption("LightGCN + FAISS + Time-aware Ranking + LLM")

title = st.text_input("Enter a movie title")
k = st.slider("Number of recommendations", 5, 20, 10)

if st.button("Find recommendations"):
    if not title.strip():
        st.warning("Please enter a movie title.")
    else:
        with st.spinner("Finding best recommendations..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"title": title, "k": k},
                    timeout=300,
                )
                data = response.json()

                for i, r in enumerate(data["results"], 1):
                    st.markdown(f"### {i}. {r['title']}")
                    st.write(f"Score: `{r['score']}`")
                    st.caption(f"🧠 {r['reason']}")

            except Exception as e:
                st.error(f"API connection failed: {e}")
