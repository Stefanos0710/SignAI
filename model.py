"""
### SignAI - Sign Language translater ###

## Model ##

-------------------------------------------------------------
The `seq2seq_model` is an encoder-decoder model implemented for sequence-to-sequence learning tasks,
such as translating a sequence of sign language keypoints into words. The model consists of the following components:

1. **Encoder**:
    - Takes an input sequence (e.g., sign language keypoints over time).
    - Applies two dense layers with ReLU activation and L1 regularization for feature extraction and preprocessing.
    - Includes a Dropout layer for regularization (30% dropout to avoid overfitting).
    - A single LSTM layer processes the sequence data, outputting the final hidden states.

2. **Decoder**:
    - Receives the target sequence as input (e.g., token IDs for words).
    - Reshapes the input into a 3D tensor for compatibility with the LSTM layer.
    - An LSTM layer processes the input sequence, initialized with the encoder's final hidden states.
    - A Dense layer with a Softmax activation generates probabilities for each token in the output vocabulary.

3. **Model**:
    - Combines the encoder and decoder into a single model that maps input sequences to output sequences.

The flexible architecture allows adjustments (e.g., number of layers, dropout rates) for experimentation and optimization.

-------------------------------------------------------------

# New from old version:
    - 2 hiddenlayer (should add maybe more. Have to test)
    - hidden layer neurons: before = 2048 ; after = 512 - 1024
    - Have to ask Professor if it´s good idea: (dropout "30%")
    - have to think

## Metadata
- **Author**: Stefanos Koufogazos Loukianov
- **Original Creation Date**:  2025/1/24
- **Last Update**: 2025/04/26
- **License**: MIT License
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Reshape
from tensorflow.keras.regularizers import l1

def build_seq2seq_model(input_sequence_length, input_feature_dim, target_vocab_size, hidden_dim,
                        target_sequence_length):
    """
    Parameters:
    - input_sequence_length: Number of frames per sequence (e.g., 20 for 1 second at 20 FPS)
    - input_feature_dim: Number of features per frame (e.g., keypoints = 10)
    - target_vocab_size: Number of possible output tokens (e.g., words or subwords)
    - hidden_dim: Number of LSTM cells in the encoder and decoder
    - target_sequence_length: Length of the target sequence (e.g., number of words in the target sentence)
    """

    """ -------------------- Encoder -------------------- """
    # Input: Sequence with shape (sequence length, number of features per frame)
    encoder_inputs = Input(shape=(input_sequence_length, input_feature_dim), name="encoder_inputs")

    # Two Dense layers for preprocessing and nonlinear projection
    hidden_layer1 = Dense(512, activation="relu",  kernel_regularizer=l1(0.01), name="encoder_dense_1")(encoder_inputs)
    hidden_layer2 = Dense(512, activation="relu",  kernel_regularizer=l1(0.01), name="encoder_dense_2")(hidden_layer1)

    # Regularization with dropout layer
    dropout_layer = Dropout(0.3, name="encoder_dropout")(hidden_layer2)

    # LSTM for sequence processing, returns the final hidden states
    encoder_lstm, state_h, state_c = LSTM(
        hidden_dim, return_state=True,  kernel_regularizer=l1(0.01), name="encoder_lstm")(dropout_layer)
    encoder_states = [state_h, state_c]

    """ -------------------- Decoder -------------------- """
    # Decoder receives the target sequence (e.g., token IDs)
    # Decoder Input Shape: (batch_size, target_sequence_length)
    decoder_inputs = Input(shape=(None,), name="decoder_inputs") # None = dynamic seq len

    decoder_reshaped = Reshape((-1, 1), name="decoder_reshape")(decoder_inputs)

    decoder_lstm = LSTM(
        hidden_dim, return_sequences=True, return_state=True, kernel_regularizer=l1(0.01), name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_reshaped, initial_state=encoder_states)

    decoder_dense = Dense(target_vocab_size, activation="softmax", name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)


    # # Decoder receives the target sequence (e.g., token IDs)
    # # Decoder Input Shape: (batch_size, target_sequence_length)
    # decoder_inputs = Input(shape=(target_sequence_length,), name="decoder_inputs")
    #
    # # Keras operation to make decoder_inputs 3D
    # decoder_inputs = Reshape((target_sequence_length, 1))(decoder_inputs)
    #
    # # Classic LSTM with the initial state from the encoder
    # decoder_lstm = LSTM(
    #     hidden_dim, return_sequences=True, return_state=True, kernel_regularizer=l1(0.01), name="decoder_lstm")
    #
    # # Apply the decoder LSTM with the initial state from the encoder
    # decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    #
    # # Dense layer to output token probabilities (softmax)
    # decoder_dense = Dense(target_vocab_size, activation="softmax", name="decoder_dense")
    # decoder_outputs = decoder_dense(decoder_outputs)

    """ -------------------- Model -------------------- """
    # Combines the encoder and decoder paths into a complete system
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name="seq2seq_model")

    return model



# ======================================================== #
# ||             OLD VERSIONS OF MODELS                 || #
# ======================================================== #

# --------------------- v2 -------------------- #
"""

def build_seq2seq_model(input_sequence_length, input_feature_dim, target_vocab_size, hidden_dim,
                        target_sequence_length):
    '''
    Parameters:
    - input_sequence_length: Number of frames per sequence (e.g., 20 for 1 second at 20 FPS)
    - input_feature_dim: Number of features per frame (e.g., keypoints = 10)
    - target_vocab_size: Number of possible output tokens (e.g., words or subwords)
    - hidden_dim: Number of LSTM cells in the encoder and decoder
    - target_sequence_length: Length of the target sequence (e.g., number of words in the target sentence)
    '''

    ''' -------------------- Encoder -------------------- '''
    # Input: Sequence with shape (sequence length, number of features per frame)
    encoder_inputs = Input(shape=(input_sequence_length, input_feature_dim), name="encoder_inputs")

    # Two Dense layers for preprocessing and nonlinear projection
    hidden_layer1 = Dense(512, activation="relu", name="encoder_dense_1")(encoder_inputs)
    hidden_layer2 = Dense(512, activation="relu", name="encoder_dense_2")(hidden_layer1)

    # Regularization with dropout layer
    dropout_layer = Dropout(0.3, name="encoder_dropout")(hidden_layer2)

    # LSTM for sequence processing, returns the final hidden states
    encoder_lstm, state_h, state_c = LSTM(
        hidden_dim, return_state=True, name="encoder_lstm")(dropout_layer)
    encoder_states = [state_h, state_c]

    ''' -------------------- Decoder -------------------- '''
    # Decoder receives the target sequence (e.g., token IDs)
    # Decoder Input Shape: (batch_size, target_sequence_length)
    decoder_inputs = Input(shape=(target_sequence_length,), name="decoder_inputs")

    # Keras operation to make decoder_inputs 3D
    decoder_inputs = Reshape((target_sequence_length, 1))(decoder_inputs)

    # Classic LSTM with the initial state from the encoder
    decoder_lstm = LSTM(
        hidden_dim, return_sequences=True, return_state=True, name="decoder_lstm")

    # Apply the decoder LSTM with the initial state from the encoder
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    # Dense layer to output token probabilities (softmax)
    decoder_dense = Dense(target_vocab_size, activation="softmax", name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)

    ''' -------------------- Model -------------------- '''
    # Combines the encoder and decoder paths into a complete system
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name="seq2seq_model")

    return model
    
    
    Epoch 1/10
285/285 ━━━━━━━━━━━━━━━━━━━━ 364s 1s/step - accuracy: 0.6689 - loss: 1.1944 - val_accuracy: 0.4737 - val_loss: 5.4949
Epoch 2/10
285/285 ━━━━━━━━━━━━━━━━━━━━ 318s 1s/step - accuracy: 0.8615 - loss: 0.3900 - val_accuracy: 0.4737 - val_loss: 6.0360
Epoch 3/10
285/285 ━━━━━━━━━━━━━━━━━━━━ 294s 1s/step - accuracy: 0.8648 - loss: 0.3527 - val_accuracy: 0.5438 - val_loss: 6.3716
Epoch 4/10
285/285 ━━━━━━━━━━━━━━━━━━━━ 261s 916ms/step - accuracy: 0.8659 - loss: 0.3433 - val_accuracy: 0.5438 - val_loss: 6.5115
Epoch 5/10
285/285 ━━━━━━━━━━━━━━━━━━━━ 263s 924ms/step - accuracy: 0.8663 - loss: 0.3380 - val_accuracy: 0.5438 - val_loss: 6.5165
Epoch 6/10
285/285 ━━━━━━━━━━━━━━━━━━━━ 272s 954ms/step - accuracy: 0.8656 - loss: 0.3370 - val_accuracy: 0.5438 - val_loss: 6.5865
Epoch 7/10
285/285 ━━━━━━━━━━━━━━━━━━━━ 287s 1s/step - accuracy: 0.8639 - loss: 0.3359 - val_accuracy: 0.5438 - val_loss: 6.5871
Epoch 8/10
285/285 ━━━━━━━━━━━━━━━━━━━━ 299s 1s/step - accuracy: 0.8664 - loss: 0.3369 - val_accuracy: 0.5087 - val_loss: 6.7552
Epoch 9/10
285/285 ━━━━━━━━━━━━━━━━━━━━ 291s 1s/step - accuracy: 0.8664 - loss: 0.3348 - val_accuracy: 0.5087 - val_loss: 6.8439
Epoch 10/10
285/285 ━━━━━━━━━━━━━━━━━━━━ 264s 924ms/step - accuracy: 0.8653 - loss: 0.3344 - val_accuracy: 0.5087 - val_loss: 7.0562
     
"""


# --------------------- v1 -------------------- #
"""
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

def old_build_seq2seq_model(input_vocab_size, target_vocab_size, embedding_dim, hidden_dim):
    # Encodere
    encoder_inputs = Input(shape=(None,), name="encoder_inputs")
    encoder_embedding = Embedding(input_vocab_size, embedding_dim, name="encoder_embedding")(encoder_inputs)
    encoder_lstm, state_h, state_c = LSTM(hidden_dim, return_state=True, name="encoder_lstm")(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(None,), name="decoder_inputs")
    decoder_embedding = Embedding(target_vocab_size, embedding_dim, name="decoder_embedding")(decoder_inputs)
    decoder_lstm = LSTM(hidden_dim, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = Dense(target_vocab_size, activation="softmax", name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)

    # Model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name="seq2seq_model")
    return model
"""
