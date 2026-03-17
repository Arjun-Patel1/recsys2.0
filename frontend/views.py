import sys
import os
import requests
import re
import urllib.parse
import polars as pl

# --- 1. PATH RESOLUTION (Solves config ModuleNotFoundError) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

import config

# --- 2. DJANGO & AUTH IMPORTS ---
from django.shortcuts import render, redirect
from django.contrib.auth import login, logout
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.decorators import login_required

# --- 3. GEMINI 2026 SDK IMPORT ---
from google import genai 

# --- 4. API & ENV CONFIGURATION ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = None
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)

# Pointing to FastAPI on Port 8001
API_URL = os.getenv("API_URL", "http://127.0.0.1:8001/recommend")
LINKS_CSV_PATH = os.path.join(BASE_DIR, 'processed_data', 'links.csv')

# ==========================================
# ENTERPRISE IN-MEMORY CACHING
# Eliminates disk I/O lag for massive datasets
# ==========================================
_CACHED_MOVIE_LIST = None
_CACHED_IMDB_MAP = None

def get_movie_data():
    global _CACHED_MOVIE_LIST, _CACHED_IMDB_MAP
    
    # Cache the 60,000+ Movies List
    if _CACHED_MOVIE_LIST is None:
        try:
            df_movies = pl.read_csv(config.MOVIES_PATH)
            _CACHED_MOVIE_LIST = df_movies.to_dicts()
            print("✅ Successfully cached movies.csv in memory.")
        except Exception as e:
            print(f"❌ Error loading movies: {e}")
            _CACHED_MOVIE_LIST = []
            
    # Cache the IMDb Links Mapping
    if _CACHED_IMDB_MAP is None:
        try:
            # dtypes={"imdbId": pl.Utf8} prevents Pandas/Polars from dropping leading zeros
            df_links = pl.read_csv(LINKS_CSV_PATH, dtypes={"imdbId": pl.Utf8})
            _CACHED_IMDB_MAP = dict(zip(df_links["movieId"], df_links["imdbId"]))
            print("✅ Successfully cached links.csv in memory.")
        except Exception as e:
            print(f"❌ Error loading links: {e}")
            _CACHED_IMDB_MAP = {}
            
    return _CACHED_MOVIE_LIST, _CACHED_IMDB_MAP

# ==========================================
# AUTHENTICATION VIEWS
# ==========================================
def register_view(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('dashboard')
    else: 
        form = UserCreationForm()
    return render(request, 'frontend/register.html', {'form': form})

def login_view(request):
    if request.method == "POST":
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('dashboard')
    else: 
        form = AuthenticationForm()
    return render(request, 'frontend/login.html', {'form': form})

def logout_view(request):
    if request.method == "POST":
        logout(request)
        return redirect('login')

# ==========================================
# MAIN DASHBOARD VIEW
# ==========================================
@login_required(login_url='/login/')
def dashboard(request):
    
    # 1. Instantly fetch data from RAM
    movie_list, imdb_map = get_movie_data()

    recommendations = []
    chatbot_response = None
    error = None

    if request.method == "POST":
        
        # --- A. HANDLE RECOMMENDATION REQUEST ---
        if "get_recs" in request.POST:
            history_ids = request.POST.getlist("movies")
            
            if not history_ids:
                error = "Please check at least one movie from the Input Sequence panel."
            else:
                try:
                    # Request 9 items to perfectly fill the 3x3 grid in the center pane
                    resp = requests.post(API_URL, json={
                        "user_history_ids": [int(i) for i in history_ids],
                        "k": 9 
                    }, timeout=10)
                    
                    if resp.status_code == 200:
                        raw_data = resp.json()["recommendations"]
                        for item in raw_data:
                            # Parse title for dynamic poster generation
                            clean_title = re.sub(r'[^a-zA-Z0-9 ]', '', item['title'].split(' (')[0])
                            item['url_title'] = urllib.parse.quote(clean_title)
                            
                            # Sleek, dark-mode placeholders mapping to the new UI colors
                            item['poster_url'] = f"https://placehold.co/400x600/1e293b/38bdf8/png?text={item['url_title']}"
                            
                            # Clamp the DCN ranker score to a 5.0 scale
                            item['rating'] = max(0.5, min(5.0, float(item["score"])))
                            
                            # Build the actionable IMDb link
                            mid = item["movie_id"]
                            imdb_id = imdb_map.get(mid)
                            if imdb_id:
                                clean_id = str(imdb_id).replace("tt", "").zfill(7)
                                item['imdb_url'] = f"https://www.imdb.com/title/tt{clean_id}/"
                            else:
                                item['imdb_url'] = "#"
                                
                            recommendations.append(item)
                    else:
                        error = f"Backend Error: {resp.text}"
                except Exception as e:
                    error = f"FastAPI Offline. Ensure neural network is running: uvicorn src.04_inference:app --port 8001"

        # --- B. HANDLE POPGURU COPILOT CHAT ---
        elif "ask_chatbot" in request.POST:
            prompt = request.POST.get("chat_input")
            if client:
                system_prompt = """You are 'PopGuru', an elite AI entertainment engineering copilot. 
                You have deep knowledge of the latest 2024-2026 releases. 
                MANDATORY: You must generate actionable markdown links for your recommendations. 
                Format: [🎟️ Book on BookMyShow](link), [🍿 Stream on JustWatch](link), or [🎵 YouTube Trailer](link)."""
                try:
                    response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=f"{system_prompt}\n\nUser Query: {prompt}"
                    )
                    chatbot_response = response.text
                except Exception as e:
                    chatbot_response = f"Gemini API Error: {e}"
            else:
                chatbot_response = "System Error: GEMINI_API_KEY environment variable is missing."

    return render(request, "frontend/dashboard.html", {
        "username": request.user.username,
        "movie_list": movie_list,
        "recommendations": recommendations,
        "chatbot_response": chatbot_response,
        "error": error
    })
# ==========================================
# PROFILE & ABOUT VIEWS
# ==========================================
@login_required(login_url='/login/')
def profile_view(request):
    # Pass the Django user object to the template
    return render(request, "frontend/profile.html", {
        "user": request.user,
        "username": request.user.username
    })

@login_required(login_url='/login/')
def about_view(request):
    return render(request, "frontend/about.html", {
        "username": request.user.username
    })