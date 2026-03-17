# File: src/03_train.py Self-Attentive Sequential Recommendation
import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
# --------------------------

import numpy as np
import polars as pl
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

import config
from models import SASRec

# Set Random Seed for Reproducibility
tf.random.set_seed(42)
np.random.seed(42)


def get_dataset(file_path, batch_size=512, max_len=50, sample_fraction=0.15):
    """
    Loads Parquet data and creates a TensorFlow Dataset.
    Implements the "Next Item Prediction" logic with aggressive subsampling.
    """
    print(f"Loading data from {file_path}...")
    df = pl.read_parquet(file_path)

    # --- SPEED UP FIX ---
    # We take a 15% random sample of users.
    # This reduces training time from 7 hours to ~10-15 minutes on a CPU.
    if sample_fraction < 1.0:
        print(
            f"Subsampling data to {sample_fraction * 100}% for fast local training..."
        )
        df = df.sample(fraction=sample_fraction, seed=42)
    # --------------------

    # Convert Polars list column to List of Lists
    sequences = df["sequence"].to_list()

    # Pad Sequences (Pre-padding with 0s)
    # If a user watched [1, 2], and max_len=5, it becomes [0, 0, 0, 1, 2]
    sequences_padded = pad_sequences(
        sequences, maxlen=max_len, padding="pre", truncating="pre"
    )

    # Create Inputs (x) and Targets (y)
    # Task: Given [A, B, C], predict [B, C, D]
    x_data = sequences_padded[:, :-1]  # All except last item
    y_data = sequences_padded[:, 1:]  # All except first item

    # Convert to Tensor
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))

    # Optimization: Shuffle, Batch, and Prefetch
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache()  # Keep in memory
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def main():
    # 1. Load Genome Embeddings (The SOTA Feature)
    print("Loading Genome Embeddings...")
    if os.path.exists(config.GENOME_EMBEDDINGS_PATH):
        genome_embeddings = np.load(config.GENOME_EMBEDDINGS_PATH)
        print(f"Genome Embeddings Loaded. Shape: {genome_embeddings.shape}")
    else:
        print("WARNING: Genome Embeddings not found. Training without side-info.")
        genome_embeddings = None

    # 2. Prepare Datasets (Overriding config batch size for speed)
    FAST_BATCH_SIZE = 512
    train_ds = get_dataset(
        config.TRAIN_DATA_PATH,
        FAST_BATCH_SIZE,
        config.MAX_LEN + 1,
        sample_fraction=0.15,
    )
    val_ds = get_dataset(
        config.VAL_DATA_PATH, FAST_BATCH_SIZE, config.MAX_LEN + 1, sample_fraction=0.15
    )

    # 3. Initialize Model
    # Get total movies from the movie map
    num_items = len(pl.read_parquet(config.MOVIE_MAP_PATH))

    print(f"Initializing SASRec Model for {num_items} items...")
    model = SASRec(
        num_items=num_items, config=config, genome_features=genome_embeddings
    )

    # 4. Compile Model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

    # 5. Callbacks (Early Stopping & Checkpointing)
    checkpoint_path = os.path.join(config.PROCESSED_DIR, "best_model.weights.h5")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
            verbose=1,
        ),
    ]

    # 6. Train (Capped at 5 epochs for speed)
    FAST_EPOCHS = 5
    print(f"Starting Training for {FAST_EPOCHS} epochs...")
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=FAST_EPOCHS, callbacks=callbacks
    )

    print("Training Complete. Model saved in Keras 2 format.")


if __name__ == "__main__":
    main()
