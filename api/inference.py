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

# Import resource_path helper
try:
    from app.resource_path import resource_path, writable_path
except Exception:
    # Fallback implementations
    import sys
    def resource_path(relative_path):
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, relative_path)
        return os.path.join(os.path.abspath('.'), relative_path)
    def writable_path(relative_path):
        if getattr(sys, 'frozen', False) and hasattr(sys, 'executable'):
            base = os.path.dirname(sys.executable)
        else:
            base = os.path.abspath('.')
        full = os.path.join(base, relative_path)
        parent = os.path.dirname(full)
        if parent:
            os.makedirs(parent, exist_ok=True)
        return full


def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_json = json.load(f)
    return tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)


FEATURES_PER_FRAME = 1086  # 33 pose + 42 hand + 468 face -> (543 * 2)

def load_and_prepare_data(csv_file_path):
    samples = []

    if not os.path.exists(csv_file_path) or os.path.getsize(csv_file_path) == 0:
        logging.error(f"CSV not found or empty: {csv_file_path}")
        # Return zeros to keep pipeline alive
        sample = np.zeros((1, 1, FEATURES_PER_FRAME), dtype=np.float32)
        return sample

    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        # Validate expected columns exist
        expected_cols = []
        expected_cols += [f"pose_{i}_x" for i in range(33)] + [f"pose_{i}_y" for i in range(33)]
        expected_cols += [f"hand_{i}_x" for i in range(42)] + [f"hand_{i}_y" for i in range(42)]
        expected_cols += [f"face_{i}_x" for i in range(468)] + [f"face_{i}_y" for i in range(468)]

        use_fallback_numeric = False
        if not reader.fieldnames or any(col not in reader.fieldnames for col in expected_cols):
            logging.error("CSV is missing expected keypoint columns; attempting numeric fallback parser.")
            use_fallback_numeric = True

        if not use_fallback_numeric:
            # Collect all frames via DictReader
            all_frames = []
            for row in reader:
                try:
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

                    if len(keypoints) == FEATURES_PER_FRAME:
                        all_frames.append(np.array(keypoints, dtype=np.float32))
                    else:
                        logging.warning(f"Skipping frame with invalid feature count: {len(keypoints)}")
                except Exception:
                    # Skip malformed rows
                    continue

            if len(all_frames) == 0:
                logging.error("No valid frames parsed from CSV; returning zeros sample.")
                return np.zeros((1, 1, FEATURES_PER_FRAME), dtype=np.float32)

            # Average over all frames
            average_keypoints = np.mean(all_frames, axis=0)
        else:
            # Fallback: numeric CSV reader; pick last 1086 numeric values per row
            file.seek(0)
            raw_reader = csv.reader(file)
            all_frames = []
            header_skipped = False
            for row in raw_reader:
                # Skip empty rows
                if not row:
                    continue
                # Try to detect header (non-numeric tokens)
                if not header_skipped:
                    try:
                        # Attempt to parse first row as floats; if fails, treat as header
                        [float(x) for x in row]
                    except Exception:
                        header_skipped = True
                        continue
                    header_skipped = True
                try:
                    values = []
                    for x in row:
                        try:
                            values.append(float(x))
                        except Exception:
                            # Non-numeric tokens become NaN; filter later
                            continue
                    if len(values) >= FEATURES_PER_FRAME:
                        frame_vals = np.array(values[-FEATURES_PER_FRAME:], dtype=np.float32)
                        all_frames.append(frame_vals)
                except Exception:
                    continue

            if len(all_frames) == 0:
                logging.error("Fallback parser found no frames; returning zeros sample.")
                return np.zeros((1, 1, FEATURES_PER_FRAME), dtype=np.float32)

            average_keypoints = np.mean(all_frames, axis=0)

        # Normalization (min-max per sample)
        min_val = float(np.min(average_keypoints))
        max_val = float(np.max(average_keypoints))
        range_val = (max_val - min_val) if max_val > min_val else 1.0
        normalized_keypoints = (average_keypoints - min_val) / range_val

        # Reshape for the model -> (batch, timesteps, features)
        sample = normalized_keypoints.astype(np.float32)[None, None, :]

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
    start_index = tokenizer.word_index.get('<start>')
    if start_index is None:
        # Fallback: use most common token index 1
        start_index = 1
    target_seq = np.array([[start_index]])

    # Decoder prediction
    output_tokens, _, _ = decoder_model.predict(
        [target_seq, state_h, state_c, encoder_outputs],
        verbose=0
    )

    # Calculate softmax probabilities
    probabilities = tf.nn.softmax(output_tokens[0, -1, :]).numpy()

    # Predicted token index
    predicted_token_index = int(np.argmax(probabilities))

    # Build word->prob mapping (skip special tokens when possible)
    all_probabilities = {}
    vocab_size = probabilities.shape[-1]
    for word, index in tokenizer.word_index.items():
        if index < vocab_size and word not in ['<start>', '<end>', '<unk>']:
            all_probabilities[word] = float(probabilities[index] * 100.0)

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

        # Use resource_path for tokenizer
        tokenizer_path = resource_path(os.path.join("tokenizers", "gloss_tokenizer.json"))
        if not os.path.exists(tokenizer_path):
            # Fallback to relative path for development
            tokenizer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tokenizers", "gloss_tokenizer.json"))

        tokenizer = load_tokenizer(tokenizer_path)
        model_load_time = time.time() - model_load_start

        # Build inference models
        build_start = time.time()
        encoder_model, decoder_model = build_inference_models(training_model)
        build_time = time.time() - build_start
        print("Models successfully created")

        # Use writable_path for live data
        csv_path = writable_path(os.path.join("data", "live", "live_dataset.csv"))
        if not os.path.exists(csv_path):
            # Fallback to relative path for development
            csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "live", "live_dataset.csv"))

        if os.path.exists(csv_path):
            # Check if CSV file is not empty
            csv_size = os.path.getsize(csv_path)
            if csv_size == 0:
                print(f"✗ Error: CSV file is empty: {csv_path}")
                return {
                    'predicted_word': None,
                    'error': 'CSV file is empty'
                }

            # Log CSV file info
            csv_modified_time = os.path.getmtime(csv_path)
            import datetime
            csv_modified_str = datetime.datetime.fromtimestamp(csv_modified_time).strftime('%Y-%m-%d %H:%M:%S')
            print(f"→ Reading CSV: {csv_path}")
            print(f"  Size: {csv_size} bytes")
            print(f"  Last modified: {csv_modified_str}")

            try:
                # Load new data (with average over all frames)
                data_load_start = time.time()
                encoder_input_data = load_and_prepare_data(csv_path)
                data_load_time = time.time() - data_load_start

                # Sanity check input shape
                if encoder_input_data.shape != (1, 1, FEATURES_PER_FRAME):
                    raise ValueError(
                        f"Prepared data has wrong shape: {encoder_input_data.shape}, expected (1, 1, {FEATURES_PER_FRAME})"
                    )

                # Make prediction for the entire sequence
                inference_start = time.time()
                predicted_word, probabilities = decode_sequence(
                    encoder_input_data,
                    encoder_model,
                    decoder_model,
                    tokenizer
                )
                inference_time = time.time() - inference_start

                total_time = time.time() - start_time

                if predicted_word:
                    # Sort predictions by probability
                    sorted_predictions = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
                    top_predictions = sorted_predictions[:5]  # Top 5

                    confidence = probabilities.get(predicted_word, 0.0)

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
                else:
                    return {
                        'predicted_word': None,
                        'confidence': 0.0,
                        'top_predictions': [],
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
