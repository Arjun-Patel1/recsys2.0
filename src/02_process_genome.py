# File: src/02_process_genome.py
import polars as pl
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import config

def process_genome():
    print("Processing Genome Scores...")
    
    # 1. Load Data
    genome = pl.read_csv(config.GENOME_SCORES_PATH)
    movie_map = pl.read_parquet(config.MOVIE_MAP_PATH)
    
    # 2. Pivot: Convert to Matrix (Rows=Movies, Cols=Tags)
    # Shape: [Num_Movies, 1128]
    print("Pivoting Genome Table (this might take a moment)...")
    genome_pivot = genome.pivot(
        index="movieId", 
        columns="tagId", 
        values="relevance"
    ).fill_null(0) # Fill missing tags with 0 relevance
    
    # 3. Align with our Mapped Movie IDs
    # We only care about movies that actually exist in our training data
    # Join with movie_map
    genome_joined = movie_map.join(
        genome_pivot, 
        left_on="original_movieId", 
        right_on="movieId", 
        how="left"
    ).fill_null(0) # Movies with no genome data get 0 vectors (Cold Start handling)
    
    # Extract just the tag columns (drop id columns)
    # Assuming tag columns start after 'mapped_movieId'
    tag_columns = [col for col in genome_joined.columns if col not in ["original_movieId", "mapped_movieId", "movieId"]]
    genome_matrix = genome_joined.select(tag_columns).to_numpy()
    
    print(f"Original Genome Matrix Shape: {genome_matrix.shape}")
    
    # 4. Dimensionality Reduction (PCA)
    # 1128 dims is too big for a side-feature. 
    # We compress it to 64 dims to capture the "Essence" of the movie.
    print("Fitting PCA...")
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(genome_matrix)
    
    pca = PCA(n_components=64) # Compress to 64 latent features
    reduced_matrix = pca.fit_transform(scaled_matrix)
    
    print(f"Reduced Matrix Shape: {reduced_matrix.shape}")
    print(f"Explained Variance: {sum(pca.explained_variance_ratio_):.2f}")

    # 5. Add Padding Vector
    # Our mapped_ids start at 1. Index 0 is padding.
    # So we need to insert a row of zeros at the top for index 0.
    final_embeddings = np.vstack([np.zeros(64), reduced_matrix])
    
    # Save as Numpy Array
    np.save(config.GENOME_EMBEDDINGS_PATH, final_embeddings)
    print("Genome Embeddings Saved!")

if __name__ == "__main__":
    process_genome()