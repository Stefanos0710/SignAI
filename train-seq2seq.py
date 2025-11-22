"""
# very goooood dataset: https://arxiv.org/pdf/2210.06791v1

others:
RWTH-PHOENIX-T
ASLG-PC12
CSL-Daily
VTT-SL

"""

import os
import csv
import json
import logging
import numpy as np
import tensorflow as tf
import pandas as pd
import io
from typing import List, Dict, Union, Tuple
import time
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Reshape
from tensorflow.keras.regularizers import l1
import concurrent.futures
import warnings
import pickle

# logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# expected number of features per frame
EXPECTED_FEATURES = 1086

# silence specific DeprecationWarning noise that originates from csv parsing of some files
warnings.filterwarnings("ignore", message="string or file could not be read to its end due to unmatched data")

def _parse_csv_text(file_name: str, text: str, used_encoding: str = 'utf-8') -> Tuple[Union[Dict, None], Union[Tuple[str, str], None]]:
    """
    Parse CSV-like text and return a sample dict or an error tuple.

    This functon tries to detect the CSV delimiter, parse each non-empty line into a numeric
    frame and collect frames that have at least EXPECTED_FEATURES values. The returned sample is a
    dict with keys: 'keypoints_sequence' (numpy.ndarray of shape [frames, EXPECTED_FEATURES]),
    'gloss' (string, usually inferred from filename or from a gloss column) and 'source_file'.

    Returns:
        (sample_dict, None) on success or (None, (file_name, error_code)) on failure.

    Notes:
        - Lines that cannot be parsed to numeric values are skipped.
        - A pandas fallback is attempted when simple line parsing yields no frames.
        - Small parsing errors result in an error code that helps debugging.
    """
    # detect delimiter using csv.Sniffer on a small sample
    detected_sep = None
    try:
        sample = '\n'.join(text.splitlines()[:20])
        dialect = csv.Sniffer().sniff(sample)
        detected_sep = dialect.delimiter
    except Exception:
        detected_sep = None

    # fallback: count common delimiters in first lines
    if detected_sep is None:
        try:
            first_lines = [l for l in text.splitlines()[:10] if l.strip()]
            sep_counts = {',': 0, ';': 0, ' ': 0, '\t': 0}
            for l in first_lines:
                for s in sep_counts:
                    sep_counts[s] += l.count(s)
            best = max(sep_counts.items(), key=lambda x: x[1])
            if best[1] > 0:
                detected_sep = best[0]
        except Exception:
            detected_sep = None

    rows = []
    skipped_lines = 0
    line_num = 0

    # try to parse each line
    for raw_line in text.splitlines():
        line_num += 1
        line = raw_line.strip()
        if not line:
            skipped_lines += 1
            continue

        vals = None
        tried_seps = []
        if detected_sep is not None:
            tried_seps.append(detected_sep)
        tried_seps.extend([',', ';', ' ', '\t'])
        for sep in tried_seps:
            try:
                # np.fromstring isvery fast for preprocessing
                vals = np.fromstring(line, sep=sep, dtype=np.float32)
                if vals.size > 0:
                    break
            except Exception:
                vals = None
        if vals is None or vals.size == 0:
            skipped_lines += 1
            continue

        if vals.size >= EXPECTED_FEATURES:
            frame = vals[:EXPECTED_FEATURES].astype(np.float32)
            if np.isnan(frame).any() or np.isinf(frame).any():
                frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
            rows.append(frame)
        else:
            skipped_lines += 1
            continue

    # if no rows found, try pandas fallback
    if len(rows) == 0:
        try:
            df = pd.read_csv(io.StringIO(text), sep=None, engine='python', encoding=used_encoding or 'utf-8', header=None, on_bad_lines='skip')
            df_numeric = df.apply(pd.to_numeric, errors='coerce')
            fallback_rows = []
            for i in range(df_numeric.shape[0]):
                vals = df_numeric.iloc[i].values
                if np.isnan(vals).all():
                    continue
                vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                if vals.size >= EXPECTED_FEATURES:
                    fallback_rows.append(vals[:EXPECTED_FEATURES])
            if len(fallback_rows) == 0:
                return None, (file_name, 'no_valid_frames_after_pandas')
            rows = fallback_rows
            logging.info(f"Pandas fallback succeeded for {file_name}: extracted {len(rows)} frames")
        except Exception as e:
            return None, (file_name, f'pandas_fallback_error:{str(e)}')

    try:
        arr = np.vstack(rows).astype(np.float32)
    except Exception as e:
        return None, (file_name, f'stack_error:{str(e)}')

    if arr.shape[1] > EXPECTED_FEATURES:
        arr = arr[:, :EXPECTED_FEATURES]
    if arr.shape[1] < EXPECTED_FEATURES:
        return None, (file_name, f'feature_dim_mismatch:{arr.shape}')

    # Try to find a glossin header gloss
    gloss_text = os.path.splitext(file_name)[0]
    if 'gloss' in text.lower():
        try:
            sample_df = pd.read_csv(io.StringIO(text), nrows=1)
            gloss_col = next((c for c in sample_df.columns if str(c).lower() == 'gloss'), None)
            if gloss_col is not None and pd.notna(sample_df.iloc[0][gloss_col]):
                gloss_text = str(sample_df.iloc[0][gloss_col]).strip()
        except Exception:
            pass

    sample = {
        'keypoints_sequence': arr,
        'gloss': gloss_text,
        'source_file': file_name
    }
    return sample, None


def load_data_from_folder(folder_path: str, max_files: int = None, parallel_workers: int = None, use_cache: bool = True, cache_file: str = None) -> List[Dict[str, Union[np.ndarray, str]]]:
    """
    Load and parse CSV files from a folder into a list of sample dicts.

    The function scans the given folder for .csv files, optionally limits the number of files,
    reads file contents using a list of common encodings and parses each file (in parallel)
    using the internal CSV text parser. Results are returned as a list of samples.

    Parameters:
        folder_path: path to folder with CSV files.
        max_files: optional limit how many files to load.
        parallel_workers: number of threads for parallel parsing. If None a sensible default is used.
        use_cache: if True, a cache file is used to store parsed results between runs.
        cache_file: explicit path for the cache file, otherwise a hidden file inside folder is used.

    Returns:
        A list of sample dicts. Raises FileNotFoundError if folder not exists, or ValueError when no valid samples.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # define cache file path
    if cache_file is None:
        cache_file = os.path.join(folder_path, '.parsed_cache.pkl')

    # if the catch is there, use and load it
    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
            logging.info(f"Loaded cached parsed data from {cache_file} ({len(cached)} samples).")
            return cached
        except Exception as e:
            logging.warning(f"Failed to load cache {cache_file}: {e}. Reparsing files.")

    # log which files are found
    csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]
    logging.info(f"Found {len(csv_files)} CSV files in {folder_path}")

    # limit number of files if requested
    if max_files is not None and max_files > 0:
        csv_files = csv_files[:max_files]

    # define number of parallel workers for faster processing
    if parallel_workers is None:
        parallel_workers = min(8, (os.cpu_count() or 4))

    all_samples = []
    start_time = time.time()

    failed_files = []

    # read all files into memory first (fast I/O) and parse in threads
    file_texts = []
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'iso-8859-1']
    for file_name in csv_files:
        csv_path = os.path.join(folder_path, file_name)
        text = None
        used_encoding = None
        for enc in encodings:
            try:
                with open(csv_path, 'r', encoding=enc, errors='replace') as f:
                    text = f.read()
                used_encoding = enc
                break
            except Exception:
                text = None
                continue
        if text is None:
            failed_files.append((file_name, 'encoding_failed'))
            continue
        file_texts.append((file_name, text, used_encoding))

    # parse in parallel threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        future_to_file = {executor.submit(_parse_csv_text, fn, tx, enc): fn for fn, tx, enc in file_texts}
        for future in concurrent.futures.as_completed(future_to_file):
            fn = future_to_file[future]
            try:
                sample, fail = future.result()
                if sample is not None:
                    all_samples.append(sample)
                else:
                    failed_files.append(fail)
            except Exception as e:
                failed_files.append((fn, f'exception:{str(e)}'))

    if not all_samples:
        raise ValueError(f"No valid samples found in any CSV file. Failed files (examples): {failed_files[:50]}")

    # save cache
    if use_cache:
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(all_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info(f"Saved parsed data cache to {cache_file}")
        except Exception as e:
            logging.warning(f"Failed to save cache to {cache_file}: {e}")

    logging.info(f'Total valid samples processed: {len(all_samples)}')
    if failed_files:
        logging.warning(f"Failed to parse {len(failed_files)} files. Examples: {failed_files[:10]}")
    logging.info(f"Data loading took {time.time() - start_time:.2f} seconds.")
    return all_samples


def build_encoder_input(all_samples, max_frames=None):
    """
    Build a 3D numpy array for encoder input from sample keypoint sequences.

    This function collects all 'keypoints_sequence' arrays from samples, decides a frame length
    (either the provided max_frames or the longest sequence in the data) and returns an array of shape
    (num_samples, used_max_frames, feature_dim) with zero-padding for shorter sequences.

    Returns:
        (encoder_input_array, used_max_frames)

    Raises:
        ValueError if no sequences are provided.
    """
    # extract all keapoints sequences
    sequences = [s["keypoints_sequence"] for s in all_samples]
    if len(sequences) == 0:
        raise ValueError("No sequences provided to build_encoder_input")

    # define max_frames to use (also important for training and padding afterwords)
    if max_frames is None:
        used_max = max(int(seq.shape[0]) for seq in sequences)
    else:
        used_max = int(max_frames)
        if used_max <= 0:
            # defensiv: falls jemand 0 übergibt, nehme längste Sequenz
            used_max = max(int(seq.shape[0]) for seq in sequences)

    # chack feature dimension
    feature_dim = int(sequences[0].shape[1])

    # create encoder input array
    encoder_input = np.zeros((len(sequences), used_max, feature_dim), dtype=np.float32)

    # copy the sequences into encoder input with padding
    for i, seq in enumerate(sequences):
        if seq is None or seq.size == 0:
            continue
        length = min(seq.shape[0], used_max)
        encoder_input[i, :length, :] = seq[:length, :]

    return encoder_input, used_max


def build_tokenizer(all_samples, extra_tokens=("<start>", "<end>")):
    """
    Create a Keras Tokenizer fitted on gloss texts and add special tokens.

    The tokenizer is created with an out-of-vocabulary token and is fit on the collected gloss
    strings from all_samples. Extra special tokens passed in `extra_tokens` are guaranteed to
    exist in the word index after fitting (they are appended if missing).

    Returns:
        The fitted Tokenizer instance.

    Raises:
        ValueError if no gloss texts are found in samples.
    """
    # set unknown token for out-of-vocabulary words
    tokenizer = Tokenizer(oov_token="<unk>")

    # extract gloss texts and clean
    gloss_texts = [s["gloss"].strip() if s.get("gloss") is not None else "" for s in all_samples]
    gloss_texts = [g for g in gloss_texts if g]

    # log data
    logging.info(f"Number of gloss texts for tokenizer: {len(gloss_texts)}")
    unique_glosses = len(set(gloss_texts))
    logging.info(f"Unique gloss samples: {unique_glosses}")

    if len(gloss_texts) == 0:
        raise ValueError("No gloss texts available to build tokenizer. Check your CSV parsing and 'gloss' extraction.")

    # # DEBUG: show sample texts
    # logging.info("Sample texts before tokenization:")
    # for text in gloss_texts[:5]:
    #     logging.info(f"  {text}")

    # train tokenizer on texts
    tokenizer.fit_on_texts(gloss_texts)

    # DEBUG: vocabulary size
    logging.info(f"Vocabulary size before special tokens: {len(tokenizer.word_index)}")

    # DEBUG: show most common words
    sorted_vocab = sorted(tokenizer.word_counts.items(), key=lambda x: x[1], reverse=True)
    logging.info("Most common words:")
    for word, count in sorted_vocab[:10]:
        logging.info(f"  {word}: {count}")

    # add special tokens if not already present
    for tkn in extra_tokens:
        if tkn not in tokenizer.word_index:
            tokenizer.word_index[tkn] = len(tokenizer.word_index) + 1

    # return tokenizer
    logging.info(f"Final vocabulary size: {len(tokenizer.word_index)}")
    return tokenizer


def build_decoder_data(all_samples, tokenizer):
    """
    Prepare decoder input and target integer sequences for training.

    The function wraps each gloss with start and end tokens, converts texts to integer
    sequences using the provided tokenizer and builds two arrays: decoder_input_data and
    decoder_target_data (both padded to the same length). These arrays are used as
    model inputs and training targets for sequence-to-sequence training.

    Returns:
        (decoder_input_data, decoder_target_data)

    Raises:
        ValueError when tokenizer produces no sequences or when glosstexts are missing.
    """
    # define start and end tokens
    start_token = "<start>"
    end_token = "<end>"

    # add start and end tokens to gloss texts
    gloss_texts = [s.get("gloss", "").strip() for s in all_samples]
    gloss_texts = [g for g in gloss_texts if g]
    if len(gloss_texts) == 0:
        raise ValueError("No gloss texts available to build decoder sequences.")

    # glosstexts with tokens
    gloss_texts_with_tokens = [f"{start_token} {g} {end_token}" for g in gloss_texts]

    # convert to token sequences
    sequences = tokenizer.texts_to_sequences(gloss_texts_with_tokens)
    if not sequences or all(len(s) == 0 for s in sequences):
        raise ValueError("Tokenizer produced empty sequences. Check tokenizer and gloss texts.")

    # create decoder input and target sequences
    decoder_input_sequences = []
    decoder_target_sequences = []

    for seq in sequences:
        if len(seq) < 2:
            # if the seq is to short ignore -> e.g. <start> + <end> = none
            continue
        decoder_input_sequences.append(seq[:-1])
        decoder_target_sequences.append(seq[1:])

    if len(decoder_input_sequences) == 0:
        raise ValueError("No valid decoder sequences after filtering short sequences.")

    # padding at same length
    max_len = max(len(s) for s in sequences) - 1
    if max_len <= 0:
        raise ValueError(f"Invalid decoder max length computed: {max_len}")

    # decoder input and target data
    decoder_input_data = pad_sequences(decoder_input_sequences, maxlen=max_len, padding='post')
    decoder_target_data = pad_sequences(decoder_target_sequences, maxlen=max_len, padding='post')

    # return both (input and target) from decoder
    logging.info(f"Decoder sequences padded to length: {max_len}")
    return decoder_input_data, decoder_target_data


def build_seq2seq_model(
        max_frames, num_features, vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        dropout_rate=0.2,
        l1_reg=0.0005
):
    """
    Build a sequence-to-sequence model that maps frame sequences to token sequences.

    The encoder accepts variable-length sequences of `num_features` per frame and uses a
    bidirectional LSTM to create context states. The decoder uses an Embedding layer
    followed by an LSTM and an additive attention mechanism to attend over encoder outputs.
    A final dense layer with softmax produces token probabilities over `vocab_size` classes.

    Parameters:
        max_frames: maximum frame length (not strictly required by this builder if masking is used).
        num_features: number of features per frame.
        vocab_size: size of the target vocabulary (including padding and special tokens).

    Returns:
        A Keras Model instance that takes [encoder_inputs, decoder_inputs] and outputs token logits.
    """
    # use tf.keras.layers. directly, that no additional top-level imports are needed
    encoder_inputs = tf.keras.Input(shape=(None, num_features), name="encoder_inputs")

    # mask paddings (0.0)
    masked = tf.keras.layers.Masking(mask_value=0.0, name="encoder_masking")(encoder_inputs)

    # bidirectional LSTM
    bi_lstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(hidden_dim, return_sequences=True, return_state=True),
        name="encoder_bidirectional_lstm"
    )

    # enc_outputs_and_states: (outputs, f_h, f_c, b_h, b_c)
    enc_outputs_and_states = bi_lstm(masked)
    encoder_outputs = enc_outputs_and_states[0]

    # states from h and c
    state_h = tf.keras.layers.Concatenate(name="encoder_state_h")([enc_outputs_and_states[1], enc_outputs_and_states[3]])
    state_c = tf.keras.layers.Concatenate(name="encoder_state_c")([enc_outputs_and_states[2], enc_outputs_and_states[4]])
    encoder_states = [state_h, state_c]

    # decoder
    decoder_inputs = tf.keras.Input(shape=(None,), name="decoder_inputs")

    # mask_zero=True makes that padding token 0 is masked
    decoder_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True, name="decoder_embedding")(decoder_inputs)
    decoder_lstm = tf.keras.layers.LSTM(hidden_dim * 2, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

    # attention between encoder and decoder
    attention = tf.keras.layers.AdditiveAttention(name="attention")
    context = attention([decoder_outputs, encoder_outputs])
    concat = tf.keras.layers.Concatenate(name="decoder_concat")([decoder_outputs, context])

    # finall dense layer
    decoder_dense = tf.keras.layers.Dense(vocab_size, activation="softmax", name="decoder_dense")
    final_outputs = decoder_dense(concat)

    # return the whole model
    model = tf.keras.Model([encoder_inputs, decoder_inputs], final_outputs, name="seq2seq_model_with_masking")
    return model


def train_main(
        train_data_folder,
        version_model=28,
        epochs=200,
        batch_size=64,
        validation_split=0.2,
        input_sequence_length=1,
        embedding_dim=512,
        hidden_dim=1024,
        dropout_rate=0.3,
        l1_reg=0.001
):
    try:

        """
        Main training entry point: load data, build tokenizer and model, then train and save.

        This function performs the full training workflow:
          1. load and parse samples from `train_data_folder`;
          2. build and fit a tokenizer on gloss texts;
          3. create encoder and decoder training arrays;
          4. build the seq2seq model and compile it with an optimizer and loss;
          5. run model.fit with common callbacks and finally save model and tokenizer to disk.

        Returns:
            The Keras History object from model.fit on success.

        Raises:
            Various exceptions when data is missing or model configuration is invalid.
        """
        # 1. Load data
        samples = load_data_from_folder(train_data_folder)

        # 2. create tokenizer
        tokenizer = build_tokenizer(samples)

        # 3. process encoder input data
        encoder_input_data, used_max_frames = build_encoder_input(samples, max_frames=input_sequence_length)
        input_feature_dim = encoder_input_data.shape[-1]

        # 4. process decoder input and target data
        decoder_input_data, decoder_target_data = build_decoder_data(samples, tokenizer)

        # Additionally indformation: show model shapes
        logging.info(f"Encoder input shape: {encoder_input_data.shape}")
        logging.info(f"Decoder input shape: {decoder_input_data.shape}")
        logging.info(f"Decoder target shape: {decoder_target_data.shape}")

        # 5. create model
        target_vocab_size = len(tokenizer.word_index) + 1
        model = build_seq2seq_model(
            max_frames=input_sequence_length,
            num_features=input_feature_dim,
            vocab_size=target_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            l1_reg=l1_reg
        )

        # Sanity-check: model-output-dimension equals target vocab size
        model_output_vocab_dim = int(model.output_shape[-1]) if model.output_shape and model.output_shape[-1] is not None else None
        logging.info(f"Model final output vocab dim: {model_output_vocab_dim}, expected: {target_vocab_size}")
        if model_output_vocab_dim is None or model_output_vocab_dim != target_vocab_size:
            raise ValueError(f"Model output vocab size ({model_output_vocab_dim}) does not match tokenizer size ({target_vocab_size}).\n" \
                             f"This often indicates an issue in tokenizer building or the 'vocab_size' passed to model builder.")

        # Learning Rate Schedule
        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate,
            first_decay_steps=1000,
            t_mul=2.0,
            m_mul=0.9,
            alpha=0.0001
        )

        # Optimizer
        initial_learning_rate = 0.001
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=initial_learning_rate,
            weight_decay=0.001,
            clipnorm=1.0,
            epsilon=1e-7,
            beta_1=0.9,
            beta_2=0.999
        )

        # compile model
        logging.info(f"Target vocabulary size (including padding/oov/special): {target_vocab_size}")
        if target_vocab_size <= 1:
            raise ValueError(f"Vocabulary size too small ({target_vocab_size}). Ensure tokenization produced at least 1 real token + 1 padding token.")

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_categorical_accuracy')]
        )

        # show model summary
        model.summary()

        # callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                min_delta=0.001
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'models/checkpoint_v{version_model}_' + 'epoch_{epoch:02d}.keras',
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=f'./logs/model_v{version_model}',
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        ]

        # training
        history = model.fit(
            [encoder_input_data, decoder_input_data],
            decoder_target_data,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            shuffle=True
        )

        # save model and tokenizer
        model_save_path = f"models/trained_model_v{version_model}.keras"
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        model.save(model_save_path)
        logging.info(f"Model saved to: {model_save_path}")

        tokenizer_path = "tokenizers/gloss_tokenizer.json"
        os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)

        try:
            with open(tokenizer_path, 'w', encoding='utf-8') as f:
                f.write(tokenizer.to_json())
        except Exception as e:
            logging.warning(f"Failed to write tokenizer json to {tokenizer_path}: {e}")
        logging.info(f"Tokenizer saved to: {tokenizer_path}")

        return history

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        # create directories if not exist
        os.makedirs("data/train_data", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("tokenizers", exist_ok=True)

        # configuration for model training
        config = {
            "train_data_folder": "data/train_data",
            "version_model": 28,
            "epochs": 500,
            "batch_size": 8,
            "validation_split": 0.2,
            "input_sequence_length": None, # None means use the max length found in dataset
            "embedding_dim": 128,
            "hidden_dim": 256,
            "dropout_rate": 0.2,
            "l1_reg": 0.0005
        }

        # starting training
        history = train_main(**config)

        # show training plots
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))

        # accuracy plot
        plt.subplot(1, 2, 1)

        met_name = None
        if 'sparse_categorical_accuracy' in history.history:
            met_name = 'sparse_categorical_accuracy'
        elif 'accuracy' in history.history:
            met_name = 'accuracy'
        elif 'acc' in history.history:
            met_name = 'acc'

        if met_name is not None:
            plt.plot(history.history[met_name], label='Training Accuracy')
            val_key = f'val_{met_name}'
            if val_key in history.history:
                plt.plot(history.history[val_key], label='Validation Accuracy')
        else:
            plt.text(0.5, 0.5, 'No accuracy metric available', horizontalalignment='center')

        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss')

        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # ending and saving model + tokenizer
        plt.tight_layout()
        plt.savefig(f'models/training_history_v{config["version_model"]}.png')
        plt.close()
        logging.info("Training completed successfully")

    except Exception as e:
        logging.error(f"Program terminated with error: {str(e)}")
        raise
