from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

def build_seq2seq_model(input_vocab_size, target_vocab_size, embedding_dim, hidden_dim):
    # Encoder
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
