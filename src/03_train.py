# File: src/03_train.py
import os
import numpy as np
import polars as pl
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

import config
from models import SASRec

tf.random.set_seed(42)
np.random.seed(42)

def get_dataset(file_path, batch_size=128, max_len=50):
    print(f"Loading data from {file_path}...")
    df = pl.read_parquet(file_path)
    
    sequences = df["sequence"].to_list()
    sequences_padded = pad_sequences(sequences, maxlen=max_len, padding='pre', truncating='pre')
    
    x_data = sequences_padded[:, :-1]
    y_data = sequences_padded[:, 1:]
    
    dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
    dataset = dataset.shuffle(buffer_size=1024)
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def main():
    print("Loading Genome Embeddings...")
    if os.path.exists(config.GENOME_EMBEDDINGS_PATH):
        genome_embeddings = np.load(config.GENOME_EMBEDDINGS_PATH)
        print(f"Genome Embeddings Loaded. Shape: {genome_embeddings.shape}")
    else:
        print("WARNING: Genome Embeddings not found. Training without side-info.")
        genome_embeddings = None

    train_ds = get_dataset(config.TRAIN_DATA_PATH, config.BATCH_SIZE, config.MAX_LEN + 1)
    val_ds = get_dataset(config.VAL_DATA_PATH, config.BATCH_SIZE, config.MAX_LEN + 1)

    num_items = len(pl.read_parquet(config.MOVIE_MAP_PATH)) 
    
    print(f"Initializing SASRec Model for {num_items} items...")
    model = SASRec(
        num_items=num_items, 
        config=config, 
        genome_features=genome_embeddings
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])

    checkpoint_path = os.path.join(config.PROCESSED_DIR, "best_model.weights.h5")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            save_weights_only=True,
            monitor="val_loss",
            verbose=1
        )
    ]

    print("Starting Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=callbacks
    )
    
    print("Training Complete. Model saved.")

if __name__ == "__main__":
    main()
