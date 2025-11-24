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
        tokenizer_json_str = f.read()
    # if there is already a dic, provide string
    if isinstance(tokenizer_json_str, dict):
        tokenizer_json_str = json.dumps(tokenizer_json_str)
    return tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json_str)

# FEATURES_PER_FRAME = 1086  # 33 pose + 42 hand + 468 face -> (543 * 2)
FEATURES_PER_FRAME = 151

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

        fieldnames = reader.fieldnames
        # when no header there, use fallback parser
        if not fieldnames:
            logging.error("CSV has no header; attempting numeric fallback parser.")
            use_fallback_numeric = True
        else:
            # if any colum is missing, fill out with zeros
            missing = [col for col in expected_cols if col not in fieldnames]
            if missing:
                logging.warning(f"CSV is missing {len(missing)} expected keypoint columns; missing columns will be padded with zeros.")
            use_fallback_numeric = False

        if not use_fallback_numeric:
            # read frames and add missing columns with zeros
            all_frames = []
            for row in reader:
                try:
                    keypoints = []
                    for col in expected_cols:
                        val = row.get(col, None)
                        if val is None or val == "":
                            keypoints.append(0.0)
                        else:
                            try:
                                keypoints.append(float(val))
                            except Exception:
                                keypoints.append(0.0)

                    if len(keypoints) == FEATURES_PER_FRAME:
                        all_frames.append(np.array(keypoints, dtype=np.float32))
                    else:
                        logging.warning(f"Skipping frame with invalid feature count: {len(keypoints)}")
                except Exception:
                    # skip malformed rows
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


def _get_layer_by_candidates(model, candidates):
    for name in candidates:
        try:
            return model.get_layer(name)
        except Exception:
            continue
    # try substring match
    for layer in model.layers:
        lname = getattr(layer, 'name', '')
        for cand in candidates:
            if cand and cand.lower() in lname.lower():
                return layer
    return None


def build_inference_models(training_model, vocab_size=None):
    """
    Build encoder and decoder inference models from a trained model.
    This function is defensive: it tries multiple candidate layer names and falls back
    to layers discovered in the loaded training_model if exact names differ.
    """
    try:
        layer_names = [getattr(l, 'name', str(l)) for l in training_model.layers]
        logging.info(f"[Inference.build] Loaded model layers: {layer_names}")
    except Exception:
        pass

    # determine encoder input shape dynamically if possible
    encoder_input_shape = None
    try:
        enc_input_layer = training_model.get_layer('encoder_inputs')
        # Keras input shape is (None, timesteps, features) or similar
        encoder_input_shape = tuple(dim for dim in enc_input_layer.input_shape[1:])
    except Exception:
        #try to deduce from model.input_shape
        try:
            if isinstance(training_model.input_shape, (list, tuple)):
                shape = training_model.input_shape
                # if model has multiple inputs, try to find one that looks like encoder (contains 'encoder' in name)
                encoder_input_shape = (1, FEATURES_PER_FRAME)
        except Exception:
            encoder_input_shape = (1, FEATURES_PER_FRAME)

    # creatae encoder input tensor
    encoder_inputs = Input(shape=encoder_input_shape, name="encoder_inputs")

    # try to apply masking layer if exists
    masking_layer = _get_layer_by_candidates(training_model, ['encoder_masking', 'masking'])
    x = encoder_inputs
    if masking_layer is not None:
        try:
            x = masking_layer(x)
        except Exception:
            # If layer expects different config, skip it
            x = encoder_inputs

    # try to find a normalization dense stack

    encoder_norm_layer = _get_layer_by_candidates(training_model, ['encoder_norm', 'layer_norm', 'batch_norm_1', 'batch_norm'])
    if encoder_norm_layer is not None:
        try:
            x = encoder_norm_layer(x)
        except Exception:
            pass

    # find bidirectional layer
    bidir = _get_layer_by_candidates(training_model, ['encoder_bidirectional', 'bidirectional', 'bidirectional_1', 'bidirectional_layer', 'bidirectional_lstm', 'bidirectional_wrapper', 'bidirectional'])
    if bidir is None:
        # try generic LSTM or GRU as fallback
        bidir = _get_layer_by_candidates(training_model, ['encoder_lstm', 'lstm', 'encoder_rnn'])

    if bidir is None:
        raise ValueError('Could not find encoder bidirectional (LSTM) layer in the trained model.')

    # apply bidirectional layer to x. Many Bidirectional wrappers return different outputs; handle common cases.
    try:
        bidir_out = bidir(x)
    except Exception:
        # as a last resort, call bidir on the raw encoder_inputs
        bidir_out = bidir(encoder_inputs)

    # bidir_out may be a tensor or a tuple (outputs, forward_h, forward_c, backward_h, backward_c)
    if isinstance(bidir_out, (list, tuple)) and len(bidir_out) >= 5:
        encoder_outputs = bidir_out[0]
        forward_h = bidir_out[1]
        forward_c = bidir_out[2]
        backward_h = bidir_out[3]
        backward_c = bidir_out[4]
    else:
        # if  the Bidirectional wrapper returns a single tensor try to retrieve state layers from model
        encoder_outputs = bidir_out
        # try to find state tensors as model layers
        fh_layer = _get_layer_by_candidates(training_model, ['encoder_state_h', 'state_h', 'forward_h'])
        fc_layer = _get_layer_by_candidates(training_model, ['encoder_state_c', 'state_c', 'forward_c'])
        bh_layer = _get_layer_by_candidates(training_model, ['backward_h', 'back_h'])
        bc_layer = _get_layer_by_candidates(training_model, ['backward_c', 'back_c'])
        try:
            forward_h = fh_layer(encoder_outputs) if fh_layer is not None else None
            forward_c = fc_layer(encoder_outputs) if fc_layer is not None else None
            backward_h = bh_layer(encoder_outputs) if bh_layer is not None else None
            backward_c = bc_layer(encoder_outputs) if bc_layer is not None else None
        except Exception:
            # if it cannot get states
            encoder_dim = int(encoder_outputs.shape[-1]) if encoder_outputs.shape[-1] is not None else 512
            forward_h = tf.zeros_like(tf.reshape(encoder_outputs[:, 0, :], (-1, encoder_dim)))
            forward_c = tf.zeros_like(forward_h)
            backward_h = tf.zeros_like(forward_h)
            backward_c = tf.zeros_like(forward_h)

    # combine states
    try:
        state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
        state_c = tf.keras.layers.Concatenate()([forward_c, backward_c])
    except Exception:
        # if concatenation fails, try to use available states
        if forward_h is not None and backward_h is not None:
            state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
            state_c = tf.keras.layers.Concatenate()([forward_c, backward_c])
        else:
            # use zeros
            enc_dim = int(encoder_outputs.shape[-1]) if encoder_outputs.shape[-1] is not None else 512
            state_h = tf.zeros((tf.shape(encoder_outputs)[0], enc_dim))
            state_c = tf.zeros_like(state_h)

    # Encoder Model
    encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])

    # ------------------ Decoder ------------------
    # Find decoder LSTM and embedding layers
    decoder_embedding_layer = _get_layer_by_candidates(training_model, ['decoder_embedding', 'embedding'])
    decoder_lstm_layer = _get_layer_by_candidates(training_model, ['decoder_lstm', 'lstm_decoder', 'decoder_lstm'])
    attention_layer = _get_layer_by_candidates(training_model, ['attention', 'multi_head_attention', 'dot_attention'])
    decoder_dense_layer = _get_layer_by_candidates(training_model, ['decoder_output', 'decoder_dense', 'dense'])

    # Infer decoder units
    decoder_units = None
    if decoder_lstm_layer is not None:
        try:
            decoder_units = int(getattr(decoder_lstm_layer, 'units', None) or getattr(decoder_lstm_layer.cell, 'units', None))
        except Exception:
            decoder_units = 512
    else:
        decoder_units = 512

    decoder_inputs = Input(shape=(1,), name='decoder_inputs')
    decoder_state_input_h = Input(shape=(decoder_units,), name='decoder_state_input_h')
    decoder_state_input_c = Input(shape=(decoder_units,), name='decoder_state_input_c')
    # encoder outputs shape: (batch, timesteps, enc_dim)
    enc_out_dim = int(encoder_outputs.shape[-1]) if encoder_outputs.shape[-1] is not None else decoder_units
    encoder_outputs_input = Input(shape=(None, enc_out_dim), name='encoder_outputs_input')

    # Embedding
    if decoder_embedding_layer is not None:
        try:
            decoder_embedding = decoder_embedding_layer(decoder_inputs)
        except Exception:
            # Fallback: use a simple dense to project to decoder_units
            decoder_embedding = Dense(decoder_units, activation='relu')(decoder_inputs)
    else:
        decoder_embedding = Dense(decoder_units, activation='relu')(decoder_inputs)

    # LSTM step
    if decoder_lstm_layer is not None:
        try:
            decoder_outputs, dec_h, dec_c = decoder_lstm_layer(
                decoder_embedding,
                initial_state=[decoder_state_input_h, decoder_state_input_c]
            )
        except Exception:
            # Try calling the underlying cell
            lstm_cell = getattr(decoder_lstm_layer, 'cell', None)
            if lstm_cell is not None:
                decoder_outputs, dec_h, dec_c = decoder_lstm_layer(decoder_embedding, initial_state=[decoder_state_input_h, decoder_state_input_c])
            else:
                # Fallback: pass embedding through a Dense
                decoder_outputs = Dense(decoder_units, activation='relu')(decoder_embedding)
                dec_h = Dense(decoder_units, activation='tanh')(decoder_outputs)
                dec_c = Dense(decoder_units, activation='tanh')(decoder_outputs)
    else:
        decoder_outputs = Dense(decoder_units, activation='relu')(decoder_embedding)
        dec_h = Dense(decoder_units, activation='tanh')(decoder_outputs)
        dec_c = Dense(decoder_units, activation='tanh')(decoder_outputs)

    # Attention
    if attention_layer is not None:
        try:
            attention_out = attention_layer(decoder_outputs, encoder_outputs_input)
        except Exception:
            # Try swapping args or using simple dot-product
            try:
                attention_out = attention_layer([decoder_outputs, encoder_outputs_input])
            except Exception:
                # Simple additive attention fallback
                attention_out = tf.keras.layers.Dot(axes=[2, 2])([decoder_outputs, encoder_outputs_input])
    else:
        attention_out = tf.keras.layers.Dot(axes=[2, 2])([decoder_outputs, encoder_outputs_input])

    # Combine
    try:
        attention_added = tf.keras.layers.Add()([decoder_outputs, attention_out])
        normalized = tf.keras.layers.LayerNormalization()(attention_added)
    except Exception:
        normalized = attention_out

    # Dense / Output
    if decoder_dense_layer is not None:
        try:
            dense_out = decoder_dense_layer(normalized)
        except Exception:
            dense_out = Dense(decoder_units, activation='relu')(normalized)
    else:
        dense_out = Dense(decoder_units, activation='relu')(normalized)

    # if the model exposes a final projection, try to use it
    final_output_layer = _get_layer_by_candidates(training_model, ['decoder_output', 'output', 'softmax', 'predictions', 'decoder_dense'])
    if final_output_layer is not None:
        # If the final_output_layer is the same as decoder_dense_layer, `dense_out` is already the output
        if final_output_layer is decoder_dense_layer or (hasattr(final_output_layer, 'name') and hasattr(decoder_dense_layer, 'name') and final_output_layer.name == decoder_dense_layer.name):
            decoder_outputs = dense_out
        else:
            try:
                decoder_outputs = final_output_layer(dense_out)
            except Exception:
                # attempt to project dense_out to the expected input dim of final_output_layer
                projected = None
                try:
                    # Try to infer expected input dim from the layer's kernel shape
                    if hasattr(final_output_layer, 'weights') and final_output_layer.weights:
                        kernel = final_output_layer.weights[0]
                        expected_in_dim = int(kernel.shape[0])
                        projected = Dense(expected_in_dim, activation='relu')(dense_out)
                        decoder_outputs = final_output_layer(projected)
                    else:
                        raise RuntimeError('No kernel information available')
                except Exception:
                    # as a final fallback, project straight to vocab_size if given, else to 512
                    out_dim = int(vocab_size) if vocab_size is not None else 512
                    decoder_outputs = Dense(out_dim, activation='softmax')(dense_out)
    else:
        out_dim = int(vocab_size) if vocab_size is not None else 512
        decoder_outputs = Dense(out_dim, activation='softmax')(dense_out)

    decoder_model = Model(
        inputs=[decoder_inputs, decoder_state_input_h, decoder_state_input_c, encoder_outputs_input],
        outputs=[decoder_outputs, dec_h, dec_c]
    )

    return encoder_model, decoder_model


def decode_sequence(encoder_input_data, encoder_model, decoder_model, tokenizer):
    # Encoder prediction
    encoder_outputs, state_h, state_c = encoder_model.predict(encoder_input_data, verbose=0)

    # Decoder setup
    start_index = tokenizer.word_index.get('<start>')
    if start_index is None:
        # use OOV token index if available, otherwise 1
        start_index = tokenizer.word_index.get('<unk>', 1)

    embedding_input_dim = None
    try:
        # try to finde embedding layer in encoder monel
        for layer in decoder_model.layers:
            # Name enthält üblicherweise 'embedding' bei einer Embedding-Schicht
            if 'embedding' in getattr(layer, 'name', '').lower():
                embedding_input_dim = getattr(layer, 'input_dim', None)
                break
    except Exception:
        embedding_input_dim = None

    # if input dim is known, validate start index
    if embedding_input_dim is not None:
        try:
            embedding_input_dim = int(embedding_input_dim)
            if start_index >= embedding_input_dim or start_index < 0:
                # try to use OOV index if available
                unk_index = tokenizer.word_index.get('<unk>', None)
                if unk_index is not None and 0 <= unk_index < embedding_input_dim:
                    logging.warning(f"Start-Index {start_index} außerhalb der Embedding-Größe ({embedding_input_dim}); Verwende OOV-Index {unk_index}.")
                    start_index = unk_index
                else:
                    # clamp to max valid index
                    new_index = max(0, embedding_input_dim - 1)
                    logging.warning(f"Start-Index {start_index} außerhalb der Embedding-Größe ({embedding_input_dim}); clamp auf {new_index}.")
                    start_index = new_index
        except Exception:
            # otherwise use unk index
            start_index = tokenizer.word_index.get('<unk>', 1)

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

def greedy_decode(encoder_input_data, encoder_model, decoder_model, tokenizer, max_len=15):

    # Encoder
    enc_outs = encoder_model.predict(encoder_input_data, verbose=0)
    # Erwartet: [encoder_outputs, state_h, state_c]
    if isinstance(enc_outs, (list, tuple)) and len(enc_outs) >= 3:
        encoder_outputs, state_h, state_c = enc_outs[0], enc_outs[1], enc_outs[2]
    else:
        # fallback: assume entire output is encoder_outputs, create zero states
        encoder_outputs = enc_outs
        enc_dim = int(encoder_outputs.shape[-1]) if encoder_outputs.shape[-1] is not None else 512
        state_h = np.zeros((encoder_outputs.shape[0], enc_dim), dtype=np.float32)
        state_c = np.zeros_like(state_h)

    # Sstart end token indices
    start_index = tokenizer.word_index.get('<start>', tokenizer.word_index.get('<unk>', 1))
    end_index = tokenizer.word_index.get('<end>', None)

    target_seq = np.array([[start_index]], dtype=np.int32)
    decoded_tokens = []
    last_probs = None
    per_step_max = []

    for _ in range(max_len):
        try:
            outputs = decoder_model.predict([target_seq, state_h, state_c, encoder_outputs], verbose=0)
        except Exception:
            try:
                outputs = decoder_model.predict([target_seq, state_h, state_c], verbose=0)
            except Exception:
                outputs = decoder_model.predict([target_seq], verbose=0)

        # parse outputs: (probs, new_h, new_c) or probs only
        if isinstance(outputs, (list, tuple)) and len(outputs) >= 3:
            probs = outputs[0]
            state_h, state_c = outputs[1], outputs[2]
        elif isinstance(outputs, (list, tuple)) and len(outputs) == 1:
            probs = outputs[0]
        else:
            probs = outputs

        # probs shape expected: (batch, seq_len, vocab)
        try:
            prob_vec = probs[0, -1, :]
        except Exception:
            # If shape unexpected, try flatten
            try:
                prob_vec = np.ravel(probs[0]) if hasattr(probs, '__iter__') else np.array(probs).ravel()
            except Exception:
                prob_vec = None

        if prob_vec is None:
            break

        # normalize safely and store per-step max
        norm = _safe_normalize_probs(prob_vec)
        if norm is not None:
            per_step_max.append(float(np.max(norm)))
            last_probs = norm
        else:
            # fallback: try to softmax logits
            try:
                logits = np.asarray(prob_vec, dtype=np.float64)
                logits = logits - np.max(logits)
                ex = np.exp(logits)
                sm = ex / np.sum(ex)
                sm = sm.astype(np.float32)
                per_step_max.append(float(np.max(sm)))
                last_probs = sm
            except Exception:
                per_step_max.append(0.0)
                last_probs = None

        next_id = int(np.argmax(last_probs)) if last_probs is not None else int(np.argmax(prob_vec))

        if end_index is not None and next_id == end_index:
            break

        decoded_tokens.append(tokenizer.index_word.get(next_id, '<unk>'))
        target_seq = np.array([[next_id]], dtype=np.int32)

    sentence = " ".join(decoded_tokens).strip()

    # return normalized last_probs vector and per-step-max list for confidence calculation
    return sentence, last_probs, per_step_max


# --- Helper utilities for probability handling and confidence ---

def _safe_normalize_probs(vec):
    """Safely normalize a 1-D vector to a probability distribution (sum==1).
    Returns None when normalization is not possible.
    """
    try:
        if vec is None:
            return None

        v = np.asarray(vec, dtype=np.float64).ravel()

        if v.size <= 1:
            return None

        s = float(np.sum(v))

        if np.isfinite(s) and s > 0:
            return (v / s).astype(np.float32)

        logits = v - np.max(v)
        ex = np.exp(logits)
        s2 = float(np.sum(ex))

        if not np.isfinite(s2) or s2 <= 0:
            return None

        return (ex / s2).astype(np.float32)

    except Exception:
        return None


def compute_sequence_confidence(per_step_max_probs):
    """Compute a sequence-level confidence in percent from per-step max-probabilities (0..1).
    Uses the mean of valid per-step max values and returns 0..100.
    """
    try:
        if not per_step_max_probs:
            return 0.0
        arr = np.asarray(per_step_max_probs, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return 0.0
        mean = float(np.clip(np.mean(arr), 0.0, 1.0))
        return mean * 100.0
    except Exception:
        return 0.0


def build_top_predictions_from_last_probs(last_probs, tokenizer, top_k=5):
    """Return top-k word/conf pairs from a normalized last_probs vector (1D, sums to 1).
    Returns list of dicts {'word':..., 'confidence': ...} with confidence in percent.
    """
    try:
        if last_probs is None:
            return []
        probs = _safe_normalize_probs(last_probs)
        if probs is None:
            return []
        vocab_len = int(probs.shape[-1])
        # get top indices
        top_idx = np.argsort(probs)[-top_k:][::-1]
        out = []
        idx_to_word = getattr(tokenizer, 'index_word', {}) or {}
        for i in top_idx:
            w = idx_to_word.get(int(i), '<unk>')
            conf = float(np.clip(probs[int(i)], 0.0, 1.0) * 100.0)
            out.append({'word': w, 'confidence': conf})
        return out
    except Exception:
        return []

def main_inference(model_path):
    try:
        start_time = time.time()

        # Load model and tokenizer
        model_load_start = time.time()
        training_model = load_model(model_path)

        # Use resource_path for tokenizer
        candidates = [
            resource_path(os.path.join("tokenizers", "gloss_tokenizer.json")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tokenizers", "gloss_tokenizer.json")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tokenizers", "gloss_tokenizer.json")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tokenizers", "gloss_tokenizer.json")),
        ]
        tokenizer = None
        tokenizer_path = None
        for c in candidates:
            try:
                if c and os.path.exists(c):
                    tokenizer_path = c
                    break
            except Exception:
                continue
        if tokenizer_path is not None:
            try:
                tokenizer = load_tokenizer(tokenizer_path)
            except Exception as e:
                logging.warning(f"Failed to load tokenizer from {tokenizer_path}: {e}")

        if tokenizer is None:
            # Create a minimal fallback tokenizer object with basic word_index
            logging.warning("Tokenizer file not found or failed to load — using fallback minimal tokenizer. Translation output will be limited.")
            class _FallbackToken:
                def __init__(self):
                    # minimal mapping: unk/start/end
                    self.word_index = {'<unk>': 1, '<start>': 1, '<end>': 2}
            tokenizer = _FallbackToken()

        model_load_time = time.time() - model_load_start

        # Build inference models
        vocab_size = None
        try:
            vocab_size = len(tokenizer.word_index) + 1
        except Exception:
            vocab_size = None

        build_start = time.time()
        encoder_model, decoder_model = build_inference_models(training_model, vocab_size=vocab_size)
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
                # predicted_word, probabilities = decode_sequence(
                #     encoder_input_data,
                #     encoder_model,
                #     decoder_model,
                #     tokenizer
                # )
                # Use greedy decode which returns (sentence, last_probs_vector, per_step_max_list)
                predicted_sentence, last_probs_vec, per_step_max_list = greedy_decode(encoder_input_data, encoder_model, decoder_model, tokenizer, max_len=15)
                predicted_word = predicted_sentence
                inference_time = time.time() - inference_start

                total_time = time.time() - start_time

                if predicted_word:
                    # Build top predictions from the last normalized probability vector
                    top_predictions = build_top_predictions_from_last_probs(last_probs_vec, tokenizer, top_k=5)
                    # Compute sequence confidence from per-step max probabilities
                    confidence = compute_sequence_confidence(per_step_max_list)

                    # Return detailed results
                    return {
                        # backward-compatible keys
                        'predicted_word': predicted_word,
                        # Friendly keys expected by the desktop app
                        'translation': predicted_word,
                        'model': os.path.basename(model_path) if model_path else None,
                        'confidence': round(confidence, 2),
                        'top_predictions': top_predictions,
                        'timing': {
                            'total_processing_time': round(total_time, 3),
                            'model_load_time': round(model_load_time, 3),
                            'build_time': round(build_time, 3),
                            'preprocessing_time': round(data_load_time, 3),
                            'data_load_time': round(data_load_time, 3),
                            'inference_time': round(inference_time, 3)
                        }
                    }
                else:
                    return {
                        'predicted_word': None,
                        'translation': '',
                        'model': os.path.basename(model_path) if model_path else None,
                        'confidence': 0.0,
                        'top_predictions': [],
                        'timing': {
                            'total_processing_time': round(total_time, 3),
                            'model_load_time': round(model_load_time, 3),
                            'build_time': round(build_time, 3),
                            'preprocessing_time': round(data_load_time, 3),
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
