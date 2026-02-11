# File: src/config.py
import os

# Base Paths
PROJECT_ROOT = r"C:\Users\arjun\Downloads\MOVIE_rec_sys"
DATA_DIR = os.path.join(PROJECT_ROOT, "ml-25m")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "processed_data")

# Make sure processed folder exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Raw Data Paths
RATINGS_PATH = os.path.join(DATA_DIR, "ratings.csv")
MOVIES_PATH = os.path.join(DATA_DIR, "movies.csv")
GENOME_SCORES_PATH = os.path.join(DATA_DIR, "genome-scores.csv")
LINKS_PATH = os.path.join(PROCESSED_DIR, "links.csv")

# Processed Paths (We save these as Parquet/Numpy for speed)
TRAIN_DATA_PATH = os.path.join(PROCESSED_DIR, "train_sequences.parquet")
VAL_DATA_PATH = os.path.join(PROCESSED_DIR, "val_sequences.parquet")
MOVIE_MAP_PATH = os.path.join(PROCESSED_DIR, "movie_id_map.parquet")
GENOME_EMBEDDINGS_PATH = os.path.join(PROCESSED_DIR, "genome_embeddings.npy")

# Model Hyperparameters (Tier-1: Centralized Config)
MAX_LEN = 50           # Context window (Last 50 movies watched)
MIN_INTERACTIONS = 5   # Filter out inactive users (noise reduction)
EMBEDDING_DIM = 128    # Size of movie/user vectors
NUM_HEADS = 4          # Transformer heads
NUM_BLOCKS = 2         # Transformer depth
DROPOUT = 0.1
BATCH_SIZE = 128
EPOCHS = 10