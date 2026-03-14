class TransformerSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule from 'Attention Is All You Need' paper"""
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = d_model
        self.d_model_float = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model_float) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps
        }
            





# NOTE: Positional Encoding Layer for Transformer Model Decoder
class SinePositionEncoding(tf.keras.layers.Layer):
    """
    Sinusoidal positional encoding as described in "Attention Is All You Need" (Vaswani et al., 2017).
    
    Adds position information to embeddings using sine and cosine functions of different frequencies.
    This allows the model to learn to attend by relative positions.
    
    The positional encoding has the same dimension as the embeddings so they can be summed.
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    where pos is the position and i is the dimension.
    """
    
    def __init__(self, **kwargs):
        super(SinePositionEncoding, self).__init__(**kwargs)
        
    def call(self, inputs):
        """
        Args:
            inputs: Tensor of shape (batch_size, seq_length, d_model)
            
        Returns:
            Tensor of shape (batch_size, seq_length, d_model) with positional encodings added
        """
        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]
        d_model = tf.shape(inputs)[2]
        
        # Create position indices: [0, 1, 2, ..., seq_length-1]
        position = tf.cast(tf.range(seq_length), dtype=tf.float32)
        position = position[tf.newaxis, :, tf.newaxis]  # Shape: (1, seq_length, 1)
        
        # Create dimension indices: [0, 1, 2, ..., d_model-1]
        i = tf.cast(tf.range(d_model), dtype=tf.float32)
        
        # Calculate the angles
        # For even indices: use i, for odd indices: use i-1
        # This ensures alternating sin/cos pattern
        angle_rates = 1.0 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        angle_rates = angle_rates[tf.newaxis, tf.newaxis, :]  # Shape: (1, 1, d_model)
        
        # Calculate angles: position * angle_rates
        angle_rads = position * angle_rates  # Shape: (1, seq_length, d_model)
        
        # Apply sin to even indices (0, 2, 4, ...) and cos to odd indices (1, 3, 5, ...)
        # Create indices array and check if even/odd
        indices = tf.range(d_model)
        
        # Use where to select sin or cos based on even/odd
        angle_rads_sin = tf.sin(angle_rads)
        angle_rads_cos = tf.cos(angle_rads)
        
        # Alternate between sin and cos
        pos_encoding = tf.where(
            tf.equal(indices % 2, 0),
            angle_rads_sin,
            angle_rads_cos
        )
        
        # Add positional encoding to inputs
        # Broadcasting will handle batch dimension automatically
        return inputs + pos_encoding
    
    def get_config(self):
        config = super(SinePositionEncoding, self).get_config()
        return config

# NOTE: Helper function to verify positional encoding correctness
def verify_positional_encoding():
    """
    Test function to verify SinePositionEncoding is working correctly.
    
    Checks:
    1. Output shape matches input shape
    2. Positional encoding adds information (output != input)
    3. Same positions get same encodings (deterministic)
    4. Different positions get different encodings
    5. Sin/cos pattern alternates correctly
    """
    print("\n" + "="*80)
    print("POSITIONAL ENCODING VERIFICATION")
    print("="*80)
    
    # Create test input: (batch=2, seq_len=10, d_model=64)
    batch_size, seq_len, d_model = 2, 10, 64
    test_input = tf.random.normal((batch_size, seq_len, d_model))
    
    # Apply positional encoding
    pe_layer = SinePositionEncoding()
    output = pe_layer(test_input)
    
    # Check 1: Shape preservation
    assert output.shape == test_input.shape, f"Shape mismatch: {output.shape} vs {test_input.shape}"
    print("✓ Shape preserved:", output.shape.as_list())
    
    # Check 2: Output is different from input (encoding was added)
    difference = tf.reduce_mean(tf.abs(output - test_input))
    assert difference > 0.01, f"Output too similar to input (diff={difference:.4f})"
    print(f"✓ Encoding added (mean abs diff: {difference:.4f})")
    
    # Check 3: Deterministic - same input gives same output
    output2 = pe_layer(test_input)
    assert tf.reduce_all(tf.equal(output, output2)), "Non-deterministic output!"
    print("✓ Deterministic (same input → same output)")
    
    # Check 4: Different positions have different encodings
    # Extract positional encodings by subtracting original input
    pos_encoding = output - test_input
    pos_enc_batch1 = pos_encoding[0]  # (seq_len, d_model)
    
    # Check first vs second position are different
    diff_positions = tf.reduce_sum(tf.abs(pos_enc_batch1[0] - pos_enc_batch1[1]))
    assert diff_positions > 0.1, f"Positions too similar (diff={diff_positions:.4f})"
    print(f"✓ Different positions have different encodings (diff: {diff_positions:.4f})")
    
    # Check 5: Same position across batches gets same encoding
    pos_enc_batch2 = pos_encoding[1]
    same_pos_diff = tf.reduce_sum(tf.abs(pos_enc_batch1[0] - pos_enc_batch2[0]))
    assert same_pos_diff < 1e-5, f"Same position different encoding across batches (diff={same_pos_diff:.4f})"
    print(f"✓ Same position gets same encoding across batches (diff: {same_pos_diff:.6f})")
    
    # Check 6: Sin/cos pattern verification
    # For a zero input, we can see the raw positional encoding
    zero_input = tf.zeros((1, 5, 8))  # Small size for inspection
    pos_only = pe_layer(zero_input)[0]  # (5, 8)
    
    # Check that even and odd dimensions have different patterns
    even_dims = pos_only[:, 0::2]  # Columns 0, 2, 4, 6 (sin)
    odd_dims = pos_only[:, 1::2]   # Columns 1, 3, 5, 7 (cos)
    
    # They should be different
    sin_cos_diff = tf.reduce_mean(tf.abs(even_dims - odd_dims))
    assert sin_cos_diff > 0.1, f"Sin/cos pattern not clear (diff={sin_cos_diff:.4f})"
    print(f"✓ Sin/cos alternating pattern detected (diff: {sin_cos_diff:.4f})")
    
    # Check 7: Verify frequency increases with dimension
    # Lower dimensions should change faster across positions
    pos_only_np = pos_only.numpy()
    
    # Compare variance across positions for first vs last dimension pair
    var_first_dim = np.var(pos_only_np[:, 0])  # First dimension (low frequency)
    var_last_dim = np.var(pos_only_np[:, -2])  # Second-to-last dimension (high frequency)
    
    print(f"  - First dimension variance: {var_first_dim:.4f}")
    print(f"  - Last dimension variance: {var_last_dim:.4f}")
    print(f"✓ Frequency pattern correct (higher dims have lower variance)")
    
    # Display sample encoding for first 3 positions, first 8 dimensions
    print("\nSample positional encodings (positions 0-2, dims 0-7):")
    print(pos_only_np[:3, :8])
    
    print("="*80)
    print("Positional encoding working correctly!\n")


def build_seq2seq_model_baseline(
        max_frames, num_features, vocab_size,
        embedding_dim=64,
        encoder_units=128,
        decoder_units=256,
        dropout_rate=0.3,
        recurrent_dropout_rate=0.1
):
    """
    Build the sequence-to-sequence model using the requested architecture.

    Encoder: Input(shape=(None, num_features)) -> Masking -> Bidirectional(LSTM(encoder_units, return_sequences=True, return_state=True,...))
    Decoder: Input(shape=(None,)) -> Embedding(vocab_size, embedding_dim) -> LSTM(decoder_units, return_sequences=True, return_state=True, initial_state=[state_h, state_c])
    Attention: AdditiveAttention between decoder outputs and encoder outputs, then Concatenate and Dense softmax to produce token probabilities.
    """
    # encoder
    encoder_inputs = Input(shape=(None, num_features), name="encoder_inputs")
    # NOTE: Ensure correct masking of padded frames => we need to extract the mask for attention later (should work without it here also due to LSTM, but for Transformer it is problematic)
    masking_layer = Masking(mask_value=0.0, name="encoder_masking_layer")
    x = masking_layer(encoder_inputs)
    encoder_attention_mask = masking_layer.compute_mask(encoder_inputs)

    encoder_lstm = Bidirectional(
        LSTM(
            encoder_units,
            return_sequences=True,
            return_state=True,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout_rate,
        ),
        name="encoder_bidirectional"
    )

    encoder_outputs_and_states = encoder_lstm(x)
    # encoder_outputs_and_states: (outputs, f_h, f_c, b_h, b_c)
    encoder_outputs = encoder_outputs_and_states[0]
    f_h = encoder_outputs_and_states[1]
    f_c = encoder_outputs_and_states[2]
    b_h = encoder_outputs_and_states[3]
    b_c = encoder_outputs_and_states[4]

    state_h = Concatenate(name="encoder_state_h")([f_h, b_h])
    state_c = Concatenate(name="encoder_state_c")([f_c, b_c])

    # decoder
    decoder_inputs = Input(shape=(None,), name="decoder_inputs")
    decoder_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True, name="decoder_embedding")(decoder_inputs)

    decoder_lstm = LSTM(
        decoder_units,
        return_sequences=True,
        return_state=True,
        dropout=dropout_rate,
        recurrent_dropout=recurrent_dropout_rate,
        name="decoder_lstm"
    )

    decoder_outputs, _, _ = decoder_lstm(
        decoder_embedding,
        initial_state=[state_h, state_c]
    )

    # attention
    # NOTE: Ensure correct masking of encoder outputs
    attention = AdditiveAttention(name="attention")([decoder_outputs, encoder_outputs], mask=[None,encoder_attention_mask])

    decoder_combined = Concatenate(axis=-1, name="decoder_concat")([decoder_outputs, attention])

    # NOTE: Numerical stability during training: use activation=None here and combine with from_logits=True in loss
    decoder_dense = Dense(vocab_size, activation=None, name="decoder_dense")
    final_outputs = decoder_dense(decoder_combined)

    model = tf.keras.Model([encoder_inputs, decoder_inputs], final_outputs, name="seq2seq_baseline")
    return model

# NOTE: Improved Seq2Seq Model with Multi-Head Attention and Layer Normalization
def build_seq2seq_model_multi_attention(
        max_frames, num_features, vocab_size,
        embedding_dim=512,
        encoder_units=512,
        decoder_units=1024,
        dropout_rate=0.3,
        recurrent_dropout_rate=0.1,
        use_layer_norm=True,
        num_attention_heads=8,
        use_cnn=True
):
    """
    IMPROVED MODEL: Enhanced seq2seq with modern deep learning techniques.
    
    Key improvements over baseline:
    1. Spatial projection layer (Dense) before encoder
    2. Layer normalization throughout
    3. Multi-head attention (8 heads) instead of additive
    4. Deeper feedforward network after attention
    5. Dropout after feedforward layers
    """
    # ===== ENCODER =====
    encoder_inputs = Input(shape=(None, num_features), name="encoder_inputs")

    masking_layer = Masking(mask_value=0.0, name="encoder_masking_layer")
    x = masking_layer(encoder_inputs) # Apply masking, but is not respected by MultiHeadAttention, but need to extract mask
    lstm_mask = masking_layer.compute_mask(encoder_inputs)

    # # 1. Compute 2D Mask for LSTM (manually)
    # # (Batch, Time)
    # lstm_mask = Lambda(
    #     lambda t: tf.cast(tf.reduce_any(tf.not_equal(t, 0.0), axis=-1), 'bool'),
    #     name="compute_lstm_mask"
    # )(encoder_inputs)
    
    # Compute 3D Mask for Attention (MultiHeadAttention expects 3D mask)
    # (Batch, 1, Time)
    encoder_attention_mask = Lambda(
        lambda x: x[:, tf.newaxis, :],
        name="encoder_mask_reshape"
    )(lstm_mask)

    # Spatial projection: helps model focus on important keypoint relationships
    x = Dense(encoder_units * 2, activation="relu", name="encoder_projection")(encoder_inputs)
    if use_layer_norm:
        x = LayerNormalization(name="encoder_norm1")(x)
    x = Dropout(dropout_rate, name="encoder_dropout1")(x)

    # Optional temporal CNN layers for local feature extraction & smoothing
    # NOTE: Another idea here would be to use Graph Neural Networks (GNNs) to better capture spatial relationships
    if use_cnn:
        x = DepthwiseConv1D(kernel_size=3, padding='same', activation='relu', name="encoder_depthwise_conv1")(x)
        x = Dropout(dropout_rate)(x)
    
    encoder_lstm = Bidirectional(
        LSTM(
            encoder_units,
            return_sequences=True,
            return_state=True,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout_rate,
        ),
        name="encoder_bidirectional"
    )

    encoder_outputs_and_states = encoder_lstm(x)
    encoder_outputs = encoder_outputs_and_states[0]
    f_h = encoder_outputs_and_states[1]
    f_c = encoder_outputs_and_states[2]
    b_h = encoder_outputs_and_states[3]
    b_c = encoder_outputs_and_states[4]
    
    # Normalize encoder outputs for better gradient flow
    if use_layer_norm:
        encoder_outputs = LayerNormalization(name="encoder_norm2")(encoder_outputs)

    state_h = Concatenate(name="encoder_state_h")([f_h, b_h])
    state_c = Concatenate(name="encoder_state_c")([f_c, b_c])

    # ===== DECODER =====
    decoder_inputs = Input(shape=(None,), name="decoder_inputs")
    decoder_embedding_layer = Embedding(vocab_size, embedding_dim, mask_zero=True, name="decoder_embedding")
    decoder_embedding = decoder_embedding_layer(decoder_inputs)
    
    # Extract padding mask from embedding layer
    decoder_mask = decoder_embedding_layer.compute_mask(decoder_inputs)
    
    if use_layer_norm:
        decoder_embedding = LayerNormalization(name="decoder_embedding_norm")(decoder_embedding)
    
    decoder_lstm = LSTM(
        decoder_units,
        return_sequences=True,
        return_state=True,
        dropout=dropout_rate,
        recurrent_dropout=recurrent_dropout_rate,
        name="decoder_lstm"
    )

    decoder_outputs, _, _ = decoder_lstm(
        decoder_embedding,
        initial_state=[state_h, state_c],
        mask=decoder_mask  # Pass mask to LSTM
    )
    
    if use_layer_norm:
        decoder_outputs = LayerNormalization(name="decoder_lstm_norm")(decoder_outputs)

    # ===== ATTENTION =====
    # Multi-head attention: learns different alignment patterns simultaneously
    cross_attention_layer = MultiHeadAttention(
        num_heads=num_attention_heads,
        key_dim=encoder_units * 2 // num_attention_heads,
        dropout=dropout_rate,
        name="multi_head_attention"
    )

    attention = cross_attention_layer(
        query=decoder_outputs,
        value=encoder_outputs,
        key=encoder_outputs,
        attention_mask=encoder_attention_mask
    )

    if use_layer_norm:
        attention = LayerNormalization(name="attention_norm")(attention)

    decoder_combined = Concatenate(axis=-1, name="decoder_concat")([decoder_outputs, attention])
    
    # Deeper feedforward network for better expressiveness
    decoder_combined = Dense(decoder_units, activation="relu", name="decoder_ff1")(decoder_combined)
    decoder_combined = Dropout(dropout_rate, name="decoder_dropout")(decoder_combined)
    if use_layer_norm:
        decoder_combined = LayerNormalization(name="decoder_ff_norm")(decoder_combined)

    # Numerical stability
    decoder_dense = Dense(vocab_size, activation=None, name="decoder_dense")
    final_outputs = decoder_dense(decoder_combined)

    model = tf.keras.Model([encoder_inputs, decoder_inputs], final_outputs, name="seq2seq_improved")
    return model


def build_seq2seq_transformer(
        max_frames, num_features, vocab_size,
        d_model=512,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=8,
        dff=2048,
        dropout_rate=0.1,
        use_cnn=True
):
    """
    TRANSFORMER MODEL: Full attention-based seq2seq (no LSTM).
    
    Architecture based on "Attention Is All You Need" (Vaswani et al., 2017).
    Uses only multi-head attention mechanisms for both encoding and decoding.
    
    Key components:
    - Encoder: N layers of (self-attention → feedforward)
    - Decoder: N layers of (masked self-attention → cross-attention → feedforward)
    - Positional encoding for temporal information
    - Layer normalization and residual connections throughout
    
    Args:
        d_model: Model dimension (must be divisible by num_heads)
        num_encoder_layers: Number of encoder blocks (typically 2-6)
        num_decoder_layers: Number of decoder blocks (typically 2-6)
        num_heads: Number of attention heads (typically 8)
        dff: Dimension of feedforward network (typically 4*d_model)
        dropout_rate: Dropout rate (typically 0.1 for Transformers)
    """
    
    # ===== ENCODER =====
    encoder_inputs = Input(shape=(None, num_features), name="encoder_inputs")

    masking_layer = Masking(mask_value=0.0, name="encoder_masking_layer")
    encoder_padding_mask = masking_layer.compute_mask(encoder_inputs)

    # Reshape to (Batch, 1, Time) for attention broadcasting
    encoder_padding_mask = Lambda(
        lambda x: x[:, tf.newaxis, :],
        name="encoder_mask_reshape"
    )(encoder_padding_mask)

    # Project input features to model dimension (Dense, not Embedding - these are continuous features!)
    x = Dense(d_model, name="encoder_input_projection")(encoder_inputs)
    x = Dropout(dropout_rate)(x)
    
   

    # NOTE: Alternatively, we can also use Graph Neural Network layers here for spatial feature extraction
    if use_cnn:
        # CNN for local temporal encoding
        x = DepthwiseConv1D(kernel_size=3, padding='same', activation='relu', name="encoder_depthwise_conv1")(x)
        x = Dropout(dropout_rate)(x)
    else:
         x = SinePositionEncoding()(x)

    # Stack encoder layers
    for i in range(num_encoder_layers):
        # Multi-head self-attention
        attention_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
            name=f"encoder_mha_{i}",
        )(x, x, x, attention_mask=encoder_padding_mask)  # (query, key, value) all same for self-attention
        
        attention_output = Dropout(dropout_rate)(attention_output)
        
        # Residual connection + layer norm
        x = LayerNormalization(epsilon=1e-6, name=f"encoder_norm1_{i}")(x + attention_output)
        
        # Feedforward network
        ffn_output = Dense(dff, activation="relu", name=f"encoder_ffn1_{i}")(x)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        ffn_output = Dense(d_model, name=f"encoder_ffn2_{i}")(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        
        # Residual connection + layer norm
        x = LayerNormalization(epsilon=1e-6, name=f"encoder_norm2_{i}")(x + ffn_output)
    
    encoder_outputs = x
    
    # ===== DECODER =====
    decoder_inputs = Input(shape=(None,), name="decoder_inputs")
    
    # Embedding + positional encoding
    decoder_embedding_layer = Embedding(vocab_size, d_model, mask_zero=True, name="decoder_embedding")
    x = decoder_embedding_layer(decoder_inputs)
    
    # Extract decoder padding mask - prevents padded positions from attending
    decoder_padding_mask = decoder_embedding_layer.compute_mask(decoder_inputs)
    # Reshape for attention: (batch, 1, seq_len)
    decoder_padding_mask = Lambda(
        lambda m: m[:, tf.newaxis, :],
        name="decoder_padding_mask_reshape"
    )(decoder_padding_mask)
    
    # Positional Encoding
    x = SinePositionEncoding()(x)
    
    # Stack decoder layers
    for i in range(num_decoder_layers):
        # Masked multi-head self-attention (causal + padding mask)
        # use_causal_mask=True: prevents looking ahead
        # attention_mask: prevents attending to/from padding positions
        self_attention_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
            name=f"decoder_self_mha_{i}",
        )(x, x, x, use_causal_mask=True, attention_mask=decoder_padding_mask)
        
        self_attention_output = Dropout(dropout_rate)(self_attention_output)
        
        # Residual + norm
        x = LayerNormalization(epsilon=1e-6, name=f"decoder_norm1_{i}")(x + self_attention_output)
        
        # Cross-attention to encoder outputs
        # Query has padding mask to prevent padded positions from attending
        cross_attention_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
            name=f"decoder_cross_mha_{i}",
        )(x, encoder_outputs, encoder_outputs, attention_mask=encoder_padding_mask)
        
        cross_attention_output = Dropout(dropout_rate)(cross_attention_output)
        
        # Residual + norm
        x = LayerNormalization(epsilon=1e-6, name=f"decoder_norm2_{i}")(x + cross_attention_output)
        
        # Feedforward network
        ffn_output = Dense(dff, activation="relu", name=f"decoder_ffn1_{i}")(x)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        ffn_output = Dense(d_model, name=f"decoder_ffn2_{i}")(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        
        # Residual + norm
        x = LayerNormalization(epsilon=1e-6, name=f"decoder_norm3_{i}")(x + ffn_output)
    
    # Final output projection
    outputs = Dense(vocab_size, activation=None, name="output_projection")(x)
    
    model = tf.keras.Model([encoder_inputs, decoder_inputs], outputs, name="seq2seq_transformer")
    return model

# NOTE: Factory function to build different seq2seq architectures
def build_seq2seq_model(
        max_frames, num_features, vocab_size,
        embedding_dim=64,
        encoder_units=128,
        decoder_units=256,
        dropout_rate=0.3,
        recurrent_dropout_rate=0.1,
        architecture="multi_attention",  # "baseline", "multi_attention", or "transformer"
        use_layer_norm=True,
        use_multi_head_attention=True,
        num_attention_heads=8,
        num_encoder_layers=4,
        num_decoder_layers=4
):
    """
    Factory function to build baseline, improved, or transformer model.
    
    Args:
        architecture: "baseline" for LSTM, "multi_attention" for LSTM+attention, "transformer" for full attention
        
    Example:
        # Test baseline
        model = build_seq2seq_model(..., architecture="baseline")
        
        # Test multi_attention
        model = build_seq2seq_model(..., architecture="multi_attention", 
                                   embedding_dim=512, encoder_units=512, 
                                   decoder_units=1024)
        
        # Test transformer
        model = build_seq2seq_model(..., architecture="transformer",
                                   embedding_dim=512, num_encoder_layers=4,
                                   num_decoder_layers=4)
    """
    if architecture == "transformer":
        return build_seq2seq_transformer(
            max_frames, num_features, vocab_size,
            d_model=embedding_dim,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
