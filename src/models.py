# File: src/models.py
import tensorflow as tf
from tensorflow.keras import layers, models

class SASRec(models.Model):
    def __init__(self, num_items, config, genome_features=None):
        super(SASRec, self).__init__()
        self.max_len = config.MAX_LEN
        self.d_model = config.EMBEDDING_DIM
        
        # 1. Item Embedding (Learned from interactions)
        self.item_embedding = layers.Embedding(
            input_dim=num_items + 1, 
            output_dim=self.d_model, 
            mask_zero=True
        )
        
        # 2. Positional Embedding (Learned time sequence)
        self.pos_embedding = layers.Embedding(
            input_dim=self.max_len, 
            output_dim=self.d_model
        )
        
        # 3. Genome Injection (The SOTA Boost)
        # If we passed genome features, we create a projection layer
        self.use_genome = genome_features is not None
        if self.use_genome:
            # Create a non-trainable constant constant tensor
            self.genome_matrix = tf.constant(genome_features, dtype=tf.float32)
            # Project 64-dim genome to 128-dim model space
            self.genome_projection = layers.Dense(self.d_model, activation="relu")

        # 4. Transformer Blocks
        self.blocks = []
        for _ in range(config.NUM_BLOCKS):
            self.blocks.append(
                layers.MultiHeadAttention(
                    num_heads=config.NUM_HEADS, 
                    key_dim=self.d_model, 
                    dropout=config.DROPOUT
                )
            )
            
        self.layer_norm1 = layers.LayerNormalization()
        self.layer_norm2 = layers.LayerNormalization()
        self.dropout = layers.Dropout(config.DROPOUT)
        
        # Final Output Layer
        self.output_layer = layers.Dense(num_items + 1)

    def call(self, inputs, training=False):
        # Inputs: (batch, seq_len)
        seq_len = tf.shape(inputs)[1]
        
        # A. Get Embeddings
        x = self.item_embedding(inputs) # (batch, seq, d_model)
        
        # B. Inject Genome Semantics (if available)
        if self.use_genome:
            # Look up genome vectors for the input movie IDs
            genome_emb = tf.nn.embedding_lookup(self.genome_matrix, inputs)
            genome_emb = self.genome_projection(genome_emb)
            # Add to learned embeddings (Residual connection style)
            x = x + genome_emb 

        # C. Add Positional Info
        positions = tf.range(start=0, limit=seq_len, delta=1)
        pos_emb = self.pos_embedding(positions)
        x = x + pos_emb
        
        # D. Causal Masking (Crucial for Sequential Recs)
        # Prevents position i from attending to position i+1
        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        
        # E. Transformer Layers
        for block in self.blocks:
            # Self-Attention
            attn_output = block(query=x, value=x, key=x, attention_mask=mask, training=training)
            # Add & Norm
            x = self.layer_norm1(x + attn_output)
            x = self.dropout(x, training=training)
            
        # We generally predict the next item for EVERY position in training
        logits = self.output_layer(x)
        
        return logits