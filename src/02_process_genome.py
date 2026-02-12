
import polars as pl
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import config

def process_genome():
    print("Processing Genome Scores...")
    
    genome = pl.read_csv(config.GENOME_SCORES_PATH)
    movie_map = pl.read_parquet(config.MOVIE_MAP_PATH)
    
    print("Pivoting Genome Table (this might take a moment)...")
    genome_pivot = genome.pivot(
        index="movieId", 
        columns="tagId", 
        values="relevance"
    ).fill_null(0)
    
    genome_joined = movie_map.join(
        genome_pivot, 
        left_on="original_movieId", 
        right_on="movieId", 
        how="left"
    ).fill_null(0)
    
    tag_columns = [col for col in genome_joined.columns if col not in ["original_movieId", "mapped_movieId", "movieId"]]
    genome_matrix = genome_joined.select(tag_columns).to_numpy()
    
    print(f"Original Genome Matrix Shape: {genome_matrix.shape}")
    
    print("Fitting PCA...")
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(genome_matrix)
    
    pca = PCA(n_components=64)
    reduced_matrix = pca.fit_transform(scaled_matrix)
    
    print(f"Reduced Matrix Shape: {reduced_matrix.shape}")
    print(f"Explained Variance: {sum(pca.explained_variance_ratio_):.2f}")

    final_embeddings = np.vstack([np.zeros(64), reduced_matrix])
    
    np.save(config.GENOME_EMBEDDINGS_PATH, final_embeddings)
    print("Genome Embeddings Saved!")

if __name__ == "__main__":
    process_genome()
