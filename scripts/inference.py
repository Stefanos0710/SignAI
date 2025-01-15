from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
import numpy as np


# Encoder-Decoder-Inferenzmodelle definieren
def build_inference_models(seq2seq_model, hidden_dim):
    # Encoder-Modell
    encoder_inputs = seq2seq_model.input[0]  # encoder_inputs
    encoder_embedding = seq2seq_model.get_layer("encoder_embedding").output
    _, state_h, state_c = seq2seq_model.get_layer("encoder_lstm").output
    encoder_states = [state_h, state_c]
    encoder_model = Model(encoder_inputs, encoder_states)

    # Decoder-Modell
    decoder_inputs = seq2seq_model.input[1]  # decoder_inputs
    decoder_state_input_h = Input(shape=(hidden_dim,), name="decoder_state_input_h")
    decoder_state_input_c = Input(shape=(hidden_dim,), name="decoder_state_input_c")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_embedding = seq2seq_model.get_layer("decoder_embedding")(decoder_inputs)
    decoder_lstm = seq2seq_model.get_layer("decoder_lstm")
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h, state_c]
    decoder_dense = seq2seq_model.get_layer("decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    return encoder_model, decoder_model


# Modell laden
seq2seq_model = load_model("seq2seq_model.h5")
encoder_model, decoder_model = build_inference_models(seq2seq_model, hidden_dim=512)


# Übersetzung
def translate_sequence(input_seq, encoder_model, decoder_model, input_tokenizer, target_tokenizer, max_len=10):
    # Eingabe encodieren
    states_value = encoder_model.predict(input_seq)

    # Starttoken
    target_seq = np.array([[target_tokenizer.word_index["<sos>"]]])
    stop_condition = False
    decoded_sentence = []

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = target_tokenizer.index_word[sampled_token_index]
        decoded_sentence.append(sampled_word)

        if sampled_word == "<eos>" or len(decoded_sentence) >= max_len:
            stop_condition = True

        # Update der Zielsequenz und Zustände
        target_seq = np.array([[sampled_token_index]])
        states_value = [h, c]

    return " ".join(decoded_sentence)


# Beispiel
if __name__ == "__main__":
    # Beispiel-Tokenizers
    input_tokenizer = {"word_index": {"<sos>": 1, "hello": 2, "world": 3},
                       "index_word": {1: "<sos>", 2: "hello", 3: "world"}}
    target_tokenizer = {"word_index": {"<sos>": 1, "<eos>": 2, "hi": 3},
                        "index_word": {1: "<sos>", 2: "<eos>", 3: "hi"}}

    input_seq = np.array([[2, 3, 0]])  # Beispiel-Eingabesequenz
    translation = translate_sequence(input_seq, encoder_model, decoder_model, input_tokenizer, target_tokenizer)
    print("Übersetzung:", translation)
