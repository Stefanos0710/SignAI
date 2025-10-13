import os
import json
import csv
import logging
import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Reshape

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_json = json.load(f)
    return tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)


def load_and_prepare_data(csv_file_path):
    samples = []

    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        # Collect all frames
        all_frames = []
        for row in reader:
            # Extract keypoints
            keypoints = []

            # Pose Keypoints (33 points)
            for i in range(33):
                keypoints.extend([float(row[f"pose_{i}_x"]), float(row[f"pose_{i}_y"])])

            # Hand Keypoints (42 points)
            for i in range(42):
                keypoints.extend([float(row[f"hand_{i}_x"]), float(row[f"hand_{i}_y"])])

            # Face Keypoints (468 points)
            for i in range(468):
                keypoints.extend([float(row[f"face_{i}_x"]), float(row[f"face_{i}_y"])])

            all_frames.append(np.array(keypoints, dtype=np.float32))

        # Average over all frames
        average_keypoints = np.mean(all_frames, axis=0)

        # Normalization
        min_val = average_keypoints.min()
        max_val = average_keypoints.max()
        range_val = max_val - min_val if max_val > min_val else 1.0
        normalized_keypoints = (average_keypoints - min_val) / range_val

        # Reshape for the model
        sample = np.expand_dims(normalized_keypoints, axis=0)  # Add batch dimension
        sample = np.expand_dims(sample, axis=1)  # Add sequence dimension

    return sample


def build_inference_models(training_model):
    # Encoder
    encoder_inputs = Input(shape=(1, 1086), name="encoder_inputs")

    # Encoder Layers
    encoder_norm = training_model.get_layer("encoder_norm")(encoder_inputs)
    encoder_dense_1 = training_model.get_layer("encoder_dense_1")(encoder_norm)
    batch_norm_1 = training_model.get_layer("batch_norm_1")(encoder_dense_1)
    dropout_1 = training_model.get_layer("dropout_1")(batch_norm_1)
    encoder_dense_2 = training_model.get_layer("encoder_dense_2")(dropout_1)
    batch_norm_2 = training_model.get_layer("batch_norm_2")(encoder_dense_2)
    dropout_2 = training_model.get_layer("dropout_2")(batch_norm_2)

    # Bidirectional LSTM
    bidirectional = training_model.get_layer("bidirectional")
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = bidirectional(dropout_2)

    # States zusammenführen
    state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
    state_c = tf.keras.layers.Concatenate()([forward_c, backward_c])

    # Encoder Model
    encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])

    # Decoder
    decoder_inputs = Input(shape=(1,), name="decoder_inputs")
    decoder_state_input_h = Input(shape=(512,), name="decoder_state_input_h")
    decoder_state_input_c = Input(shape=(512,), name="decoder_state_input_c")
    encoder_outputs_input = Input(shape=(1, 512), name="encoder_outputs_input")

    # Decoder Layers
    decoder_embedding = training_model.get_layer("decoder_embedding")(decoder_inputs)

    # LSTM Layer
    decoder_lstm = training_model.get_layer("decoder_lstm")
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding,
        initial_state=[decoder_state_input_h, decoder_state_input_c]
    )

    # Attention
    attention = training_model.get_layer("multi_head_attention")(
        decoder_outputs, encoder_outputs_input
    )

    # Add & Norm
    attention_output = tf.keras.layers.Add()([decoder_outputs, attention])
    normalized = tf.keras.layers.LayerNormalization()(attention_output)

    # Dense Layers
    dense = training_model.get_layer("decoder_dense")(normalized)
    decoder_outputs = training_model.get_layer("decoder_output")(dense)

    # Decoder Model
    decoder_model = Model(
        inputs=[decoder_inputs, decoder_state_input_h, decoder_state_input_c, encoder_outputs_input],
        outputs=[decoder_outputs, state_h, state_c]
    )

    return encoder_model, decoder_model


def decode_sequence(encoder_input_data, encoder_model, decoder_model, tokenizer):
    # Encoder prediction
    encoder_outputs, state_h, state_c = encoder_model.predict(encoder_input_data, verbose=0)

    # Decoder setup
    target_seq = np.array([[tokenizer.word_index['<start>']]])

    # Decoder prediction
    output_tokens, _, _ = decoder_model.predict(
        [target_seq, state_h, state_c, encoder_outputs],
        verbose=0
    )

    # Calculate softmax probabilities
    probabilities = tf.nn.softmax(output_tokens[0, -1, :]).numpy()

    # Find the word with the highest probability
    predicted_token_index = np.argmax(probabilities)

    # Store probabilities for all words
    all_probabilities = {}
    for word, index in tokenizer.word_index.items():
        if word not in ['<start>', '<end>', '<unk>']:
            all_probabilities[word] = probabilities[index] * 100

    # Find the predicted word
    predicted_word = None
    for word, index in tokenizer.word_index.items():
        if index == predicted_token_index and word not in ['<start>', '<end>', '<unk>']:
            predicted_word = word
            break

    return predicted_word, all_probabilities


def main_inference(model_path):
    try:
        start_time = time.time()

        # Load model and tokenizer
        model_load_start = time.time()
        training_model = load_model(model_path)
        tokenizer = load_tokenizer("../tokenizers/gloss_tokenizer.json")
        model_load_time = time.time() - model_load_start

        # Build inference models
        build_start = time.time()
        encoder_model, decoder_model = build_inference_models(training_model)
        build_time = time.time() - build_start
        print("Models successfully created")

        if os.path.exists("../data/live/live_dataset.csv"):
            try:
                # Load new data (with average over all frames)
                data_load_start = time.time()
                encoder_input_data = load_and_prepare_data("../data/live/live_dataset.csv")
                data_load_time = time.time() - data_load_start

                # Make prediction for the entire sequence
                inference_start = time.time()
                predicted_word, probabilities = decode_sequence(
                    encoder_input_data,
                    encoder_model,
                    decoder_model,
                    tokenizer
                )
                inference_time = time.time() - inference_start

                if predicted_word:
                    # Sort predictions by probability
                    sorted_predictions = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                    top_predictions = sorted_predictions[:5]  # Top 5

                    confidence = probabilities.get(predicted_word, 0.0)
                    total_time = time.time() - start_time

                    print(f"\nPrediction: {predicted_word.upper()}")
                    print(f"Confidence: {confidence:.2f}%")
                    print("\nTop 5 Predictions:")
                    for word, prob in top_predictions:
                        print(f"{word.upper()}: {prob:.2f}%")
                    print(f"\nTotal time: {total_time:.3f}s")
                    print("-" * 50)

                    # Return detailed results
                    return {
                        'predicted_word': predicted_word,
                        'confidence': round(confidence, 2),
                        'top_predictions': [
                            {'word': word, 'confidence': round(prob, 2)}
                            for word, prob in top_predictions
                        ],
                        'timing': {
                            'total_time': round(total_time, 3),
                            'model_load_time': round(model_load_time, 3),
                            'build_time': round(build_time, 3),
                            'data_load_time': round(data_load_time, 3),
                            'inference_time': round(inference_time, 3)
                        }
                    }

            except Exception as e:
                print(f"Error during processing: {str(e)}")
                return None

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise
    return None
