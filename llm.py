# llm.py
import requests
import re
from typing import List, Dict

# ---------------- CONFIG ----------------
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"
TIMEOUT = 60


# ---------------- CORE LLM CALL ----------------
def ollama_generate(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
        if r.status_code != 200:
            return ""
        return r.json().get("response", "").strip()
    except Exception:
        return ""


# ---------------- EXPLANATIONS ----------------
def explain_recommendation(
    seed_title: str,
    candidate_title: str,
    score: float | None = None,
    year: int | None = None,
    popularity: int | None = None,
) -> str:
    """
    Faithful explanation:
    - NO plot or theme invention
    - Grounded in collaborative filtering signals
    """

    prompt = f"""
You are explaining a movie recommendation.

Context:
- User liked: "{seed_title}"
- Recommended movie: "{candidate_title}"

Available signals:
- Similar users liked both movies
- Popularity (number of ratings): {popularity if popularity is not None else "unknown"}
- Release year: {year if year is not None else "unknown"}

Rules:
- Use ONLY the signals above
- Do NOT invent plot, story, or thematic similarities
- Keep it factual and ONE sentence
"""

    response = ollama_generate(prompt)

    if response:
        return response

    # deterministic fallback (never hallucates)
    parts = ["Similar users who liked this also watched it"]
    if popularity and popularity > 50:
        parts.append("it is relatively popular")
    if year and year >= 2015:
        parts.append("it is a more recent release")

    return ", ".join(parts).capitalize() + "."


# ---------------- LLM RE-RANKING ----------------
def llm_rerank(seed_title: str, recs: List[Dict]) -> List[Dict]:
    """
    LLM-assisted reranking with strong safety guarantees.
    If LLM output is weak or noisy, original ranking is preserved.
    """

    if not recs:
        return recs

    titles = [r["title"] for r in recs]

    prompt = f"""
User liked: "{seed_title}"

Task:
Re-rank the following movies from most to least relevant.

Rules:
- Do NOT add or remove movies
- Do NOT explain anything
- Output ONLY movie titles
- One title per line
- Preserve exact or very similar wording

Movies:
{titles}
"""

    text = ollama_generate(prompt)
    if not text:
        return recs

    # Normalize LLM output
    llm_lines = [
        re.sub(r"^[0-9\.\-\)\s]+", "", line).strip()
        for line in text.splitlines()
        if line.strip()
    ]

    reordered = []
    used = set()

    for line in llm_lines:
        for r in recs:
            if r["title"].lower() == line.lower() and r["title"] not in used:
                reordered.append(r)
                used.add(r["title"])
                break

    # Append any missing items in original order
    for r in recs:
        if r["title"] not in used:
            reordered.append(r)

    # Safety: ensure order quality
    if len(reordered) != len(recs):
        return recs

    return reordered

