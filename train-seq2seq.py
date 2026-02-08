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
import re
from typing import List, Dict, Union, Tuple
import time
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Masking, Bidirectional, Concatenate, Embedding, AdditiveAttention, LayerNormalization, MultiHeadAttention, Embedding, DepthwiseConv1D, Lambda

import concurrent.futures
import warnings
import pickle
import random

# NOTE: Set random seeds for reproducibility
import keras
keras.utils.set_random_seed(42)

# logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# NOTE: I think the biggest problems lay here, in the fact that we don't have enough features for this difficult task -> 151 and every second frame is not enough
# expected number of features per frame
EXPECTED_FEATURES = 151
# minimal accepted numeric values in a parsed frame (flexible fallback)
MIN_ACCEPTED_FEATURES = 50

# silence specific DeprecationWarning noise that originates from csv parsing of some files
warnings.filterwarnings("ignore", message="string or file could not be read to its end due to unmatched data")
    
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
                # np.fromstring is very fast for purely-numeric lines but fails when lines start with strings
                # try simple numeric extraction using splitting and converting tokens to float
                if sep in [',',';','\t',' ']:
                    toks = re.split(r'[{},\t ]+'.format(re.escape(sep)), line) if sep != ' ' else re.split(r'\s+', line)
                else:
                    toks = re.split(re.escape(sep), line)
                num_vals = []
                for t in toks:
                    try:
                        if t is None:
                            continue
                        vv = float(t)
                        num_vals.append(vv)
                    except Exception:
                        # skip non-numeric tokens (e.g., Video_Name, Gloss)
                        continue
                if len(num_vals) > 0:
                    vals = np.array(num_vals, dtype=np.float32)
                    break
            except Exception:
                vals = None
        if vals is None or vals.size == 0:
            skipped_lines += 1
            continue

        # accept frames with at least a minimal number of numeric features
        if vals.size >= (EXPECTED_FEATURES if EXPECTED_FEATURES and EXPECTED_FEATURES > 0 else MIN_ACCEPTED_FEATURES):
            # if EXPECTED_FEATURES is set and larger than actual, we'll trim/pad later in build_encoder_input
            frame = vals[:EXPECTED_FEATURES].astype(np.float32) if (EXPECTED_FEATURES and vals.size >= EXPECTED_FEATURES) else vals.astype(np.float32)
            if np.isnan(frame).any() or np.isinf(frame).any():
                frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
            rows.append(frame)
        else:
            # if not enough numeric tokens, skip
            skipped_lines += 1
            continue

    # if no rows found, try pandas fallback
    if len(rows) == 0:
        try:
            df = pd.read_csv(io.StringIO(text), sep=None, engine='python', encoding=used_encoding or 'utf-8', header=0, on_bad_lines='skip')
            # attempt to keep only numeric columns: coerce and drop all-NaN columns
            df_numeric = df.apply(pd.to_numeric, errors='coerce')
            # drop columns that are completely NaN
            df_numeric = df_numeric.loc[:, df_numeric.notna().any(axis=0)]
            if df_numeric.shape[1] == 0:
                return None, (file_name, 'pandas_no_numeric_columns')

            fallback_rows = []
            for i in range(df_numeric.shape[0]):
                vals = df_numeric.iloc[i].values
                # keep only non-nan numeric tokens (this will remove text columns like Video_Name/Gloss)
                good = vals[~np.isnan(vals)]
                if good.size == 0:
                    continue
                good = np.nan_to_num(good, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                if good.size >= MIN_ACCEPTED_FEATURES:
                    fallback_rows.append(good)
                else:
                    # allow rows that match the majority width later by padding - still collect them
                    fallback_rows.append(good)

            if len(fallback_rows) == 0:
                return None, (file_name, 'no_valid_frames_after_pandas')

            # make all rows the same width by padding with zeros to the max width found in this file
            maxw = max(r.size for r in fallback_rows)
            padded = [np.pad(r, (0, maxw - r.size), 'constant', constant_values=0.0) for r in fallback_rows]
            rows = padded
            logging.info(f"Pandas fallback succeeded for {file_name}: extracted {len(rows)} frames with width {maxw}")
        except Exception as e:
            return None, (file_name, f'pandas_fallback_error:{str(e)}')

    try:
        arr = np.vstack(rows).astype(np.float32)
    except Exception as e:
        return None, (file_name, f'stack_error:{str(e)}')

    # if EXPECTED_FEATURES was set and arr has more columns, trim; if fewer, keep - upstream will handle differing dims
    if EXPECTED_FEATURES and arr.shape[1] > EXPECTED_FEATURES:
        arr = arr[:, :EXPECTED_FEATURES]
    if EXPECTED_FEATURES and arr.shape[1] < EXPECTED_FEATURES:
        # don't fail here: return actual dim and let later steps trim/pad across dataset
        logging.warning(f"File {file_name} has {arr.shape[1]} features < EXPECTED_FEATURES ({EXPECTED_FEATURES}). Using {arr.shape[1]} features for this sample.")

    # Try to find a gloss in header or first columns
    gloss_text = os.path.splitext(file_name)[0]
    if 'gloss' in text.lower() or 'Gloss' in text[:200]:
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

    # NOTE: This is necessary else the special tokens get filtered out which are added during decoding (+ negations)
    custom_filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'

    # set unknown token for out-of-vocabulary words
    tokenizer = Tokenizer(oov_token="<unk>", lower=True, filters=custom_filters)

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

    # Inject Special Tokens
    special_tokens = ["<start>", "<end>"]
    
    for token in special_tokens:
        if token not in tokenizer.word_index:
            # Add to the end of the vocab
            new_id = len(tokenizer.word_index) + 1
            tokenizer.word_index[token] = new_id
            tokenizer.index_word[new_id] = token
            logging.info(f"Manually added {token} at ID {new_id}")

    # DEBUG: show most common words
    sorted_vocab = sorted(tokenizer.word_counts.items(), key=lambda x: x[1], reverse=True)
    logging.info("Most common words:")
    for word, count in sorted_vocab[:10]:
        logging.info(f"  {word}: {count}")

    # Verify special tokens are present
    logging.info("="*60)
    logging.info("SPECIAL TOKEN VERIFICATION:")
    for tkn in ["<start>", "<end>", "<unk>"]:
        token_id = tokenizer.word_index.get(tkn, None)
        logging.info(f"  {tkn}: ID={token_id}")
        if token_id is None:
            raise ValueError(f"Special token {tkn} not found in tokenizer vocabulary.")
    logging.info("="*60)

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

    # DEBUGGING: Check for data leakage and token encoding
    logging.info(f"Decoder sequences padded to length: {max_len}")
    logging.info("="*60)
    logging.info(f"Sample decoder_input[0]: {decoder_input_data[0][:20]}")
    logging.info(f"Sample decoder_target[0]: {decoder_target_data[0][:20]}")
    logging.info(f"Are they identical? {np.array_equal(decoder_input_data[0], decoder_target_data[0])}")
    
    # Verify start/end tokens are properly encoded
    start_id = tokenizer.word_index.get(start_token, None)
    end_id = tokenizer.word_index.get(end_token, None)
    logging.info("")
    logging.info("TOKEN ENCODING VERIFICATION:")
    logging.info(f"  {start_token} token ID: {start_id}")
    logging.info(f"  {end_token} token ID: {end_id}")
    logging.info(f"  First token in decoder_input[0]: {decoder_input_data[0][0]} (should be {start_id})")
    logging.info(f"  Last non-zero in decoder_target[0]: {decoder_target_data[0][np.nonzero(decoder_target_data[0])[0][-1] if np.any(decoder_target_data[0]) else 0]} (should be {end_id})")
    
    return decoder_input_data, decoder_target_data

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
            num_heads=num_attention_heads,
            dff=encoder_units * 4,  # Typically 2-4x d_model
            dropout_rate=dropout_rate
        )
    elif architecture == "multi_attention":
        return build_seq2seq_model_multi_attention(
            max_frames, num_features, vocab_size,
            embedding_dim, encoder_units, decoder_units,
            dropout_rate, recurrent_dropout_rate,
            use_layer_norm, num_attention_heads
        )
    else:
        return build_seq2seq_model_baseline(
            max_frames, num_features, vocab_size,
            embedding_dim, encoder_units, decoder_units,
            dropout_rate, recurrent_dropout_rate
        )


def train_main(
        train_data_folder,
        version_model=31,
        # v31 is old architecture with fixed masking and attention, 
        # v32 is new architecture with multihead attention, 
        # v33 is transformer architecture
        epochs=10,
        batch_size=16,
        validation_split=0.2,
        input_sequence_length=1,
        embedding_dim=512,
        hidden_dim=1024,
        dropout_rate=0.3,
        l1_reg=0.001,
        augment=False,
        augment_factor=1,  # number of augmented samples to create per original (1 or 2)
        architecture="multi_attention"  # "baseline", "multi_attention", or "transformer"
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

        sequence_lengths = [len(s['gloss'].split()) for s in samples]

        logging.info(f"Total training samples: {len(samples)}")
        logging.info(f"Average tokens per sample: {np.mean(sequence_lengths):.1f}")
        logging.info(f"Min: {np.min(sequence_lengths)}, Max: {np.max(sequence_lengths)}")
        logging.info(f"Sequences with <3 tokens: {sum(1 for l in sequence_lengths if l < 3)}/{len(sequence_lengths)}")

        # optionally augment data (if augment=true)
        if augment:
            logging.info(f"Data augmentation enabled: generating {augment_factor} extra augmented variants per sample (original + extras)...")
            augmented = []
            sw_n = 2
            for s in samples:
                try:
                    # max_augments expects total variants (including original), so add +1
                    vars = augment_sample_variants(s, make_speed_warp_n=sw_n, max_augments=(augment_factor + 1))
                    augmented.extend(vars)
                except Exception as e:
                    logging.warning(f"Augmentation failed for {s.get('source_file')}: {e}")
            logging.info(f"Augmented samples: original={len(samples)} -> augmented_total={len(augmented)}")
            samples = augmented

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
        logging.info(f"Building model with architecture: {architecture}")
        model = build_seq2seq_model(
            max_frames=input_sequence_length,
            num_features=input_feature_dim,
            vocab_size=target_vocab_size,
            embedding_dim=embedding_dim,
            encoder_units=embedding_dim,
            decoder_units=hidden_dim,
            dropout_rate=dropout_rate,
            recurrent_dropout_rate=0.1,
            architecture=architecture,
            use_layer_norm=True,
            use_multi_head_attention=True,
            num_attention_heads=8
        )

        # Sanity-check: model-output-dimension equals target vocab size
        model_output_vocab_dim = int(model.output_shape[-1]) if model.output_shape and model.output_shape[-1] is not None else None
        logging.info(f"Model final output vocab dim: {model_output_vocab_dim}, expected: {target_vocab_size}")
        if model_output_vocab_dim is None or model_output_vocab_dim != target_vocab_size:
            raise ValueError(f"Model output vocab size ({model_output_vocab_dim}) does not match tokenizer size ({target_vocab_size}).\n" \
                             f"This often indicates an issue in tokenizer building or the 'vocab_size' passed to model builder.")

        # Learning Rate Schedule with Warmup for Transformer
        # Transformers need gradual warmup to stabilize training
        d_model = embedding_dim
        
        
        # Use transformer schedule if using transformer architecture, else cosine decay
        if architecture == "transformer":
            lr_schedule = TransformerSchedule(d_model)
            logging.info(f"Using Transformer LR schedule")
        else:
            initial_learning_rate = 0.001
            lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate,
                first_decay_steps=1000,
                t_mul=2.0,
                m_mul=0.9,
                alpha=0.0001
            )
            logging.info("Using CosineDecayRestarts LR schedule")

        # Optimizer
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=0.001,
            clipnorm=1.0,
            epsilon=1e-7,
            beta_1=0.9,
            beta_2=0.98  # Higher beta2 for more stable updates
        )

        # compile model
        logging.info(f"Target vocabulary size (including padding/oov/special): {target_vocab_size}")
        if target_vocab_size <= 1:
            raise ValueError(f"Vocabulary size too small ({target_vocab_size}). Ensure tokenization produced at least 1 real token + 1 padding token.")

        # NOTE: Use ignore_class to exclude padding from loss calculation
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            ignore_class=0  # Ignore padding token (0) in loss calculation
        )
        metrics = [
            # NOTE: BLEU-4 or ROUGE should be monitored here
            tf.keras.metrics.SparseCategoricalAccuracy(
                name='sparse_categorical_accuracy',
                dtype=tf.float32,
            ),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=5, 
                name='top_5_accuracy',
                dtype=tf.float32,
            )
        ]
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

        # show model summary
        model.summary()

        # callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,  # Increased patience for slower convergence
                restore_best_weights=True,
                min_delta=0.001
            ),
            # NOTE: ReduceLROnPlateau removed - conflicts with CosineDecayRestarts schedule
            # The lr_schedule already handles learning rate adjustments
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'models/checkpoint_v{version_model}_' + 'epoch_{epoch:02d}.keras',
                save_best_only=True,
                monitor='val_loss', #NOTE: You should use BLEU here also if possible to be comparable - it is also less strict
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


# data augmentation utilities
def _pairs_view(arr: np.ndarray):
    """Return view of arr as (T, N_pairs, 2) and number of pairs."""
    T, F = arr.shape
    n_pairs = F // 2
    paired = arr[:, :n_pairs*2].reshape((T, n_pairs, 2)).copy()
    return paired, n_pairs


def _pairs_to_flat(paired: np.ndarray, orig_F: int):
    """Convert paired (T, n_pairs, 2) back to flat (T, F) preserving original F by padding zeros if needed."""
    T, n_pairs, _ = paired.shape
    flat = paired.reshape((T, n_pairs*2))
    if n_pairs*2 < orig_F:
        padw = orig_F - n_pairs*2
        flat = np.pad(flat, ((0,0),(0,padw)), constant_values=0.0)
    return flat


def jitter_frames(arr: np.ndarray, pct: float = 0.01):
    """Apply jitter noise ±pct (relative) to coordinates.
    pct: maximum absolute relative change (e.g., 0.01 for ±1%)."""
    paired, n_pairs = _pairs_view(arr)
    # relative noise per coordinate
    noise = np.random.uniform(-pct, pct, size=paired.shape).astype(np.float32)
    paired = paired * (1.0 + noise)
    return _pairs_to_flat(paired, arr.shape[1])


def scale_and_shift(arr: np.ndarray, scale_range=(0.97,1.03), shift_x_range=(0.01,0.03), shift_y_range=(0.01,0.02)):
    """Apply scaling around center and random shifts in x/y (percent of range).
    scale_range: (min,max)
    shift ranges are fractions of data range."""
    paired, n_pairs = _pairs_view(arr)
    # compute center across all valid coords
    xs = paired[:,:,0]
    ys = paired[:,:,1]
    # ignore zeros (masked) when computing ranges
    x_valid = xs[np.abs(xs) > 0]
    y_valid = ys[np.abs(ys) > 0]
    if x_valid.size == 0 or y_valid.size == 0:
        center_x = np.mean(xs)
        center_y = np.mean(ys)
        x_range = 1.0
        y_range = 1.0
    else:
        center_x = np.mean(x_valid)
        center_y = np.mean(y_valid)
        x_range = x_valid.max() - x_valid.min() if x_valid.max() != x_valid.min() else 1.0
        y_range = y_valid.max() - y_valid.min() if y_valid.max() != y_valid.min() else 1.0

    s = np.random.uniform(scale_range[0], scale_range[1])

    # scale around center
    paired = (paired - np.array([center_x, center_y])) * s + np.array([center_x, center_y])

    # shift
    shift_x_pct = np.random.uniform(shift_x_range[0], shift_x_range[1])
    shift_y_pct = np.random.uniform(shift_y_range[0], shift_y_range[1])

    # choose direction randomly
    shift_x = shift_x_pct * x_range * (1 if random.random() < 0.5 else -1)
    shift_y = shift_y_pct * y_range * (1 if random.random() < 0.5 else -1)
    paired[:,:,0] += shift_x
    paired[:,:,1] += shift_y

    return _pairs_to_flat(paired, arr.shape[1])


def rotate_frames(arr: np.ndarray, deg_range=(-3.0, 3.0)):
    paired, n_pairs = _pairs_view(arr)
    theta = np.deg2rad(np.random.uniform(deg_range[0], deg_range[1]))
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # center
    xs = paired[:,:,0]
    ys = paired[:,:,1]
    valid_x = xs[np.abs(xs) > 0]
    valid_y = ys[np.abs(ys) > 0]
    if valid_x.size == 0 or valid_y.size == 0:
        cx = np.mean(xs)
        cy = np.mean(ys)
    else:
        cx = np.mean(valid_x)
        cy = np.mean(valid_y)

    # rotate each point
    x_rel = paired[:,:,0] - cx
    y_rel = paired[:,:,1] - cy
    x_rot = x_rel * cos_t - y_rel * sin_t
    y_rot = x_rel * sin_t + y_rel * cos_t
    paired[:,:,0] = x_rot + cx
    paired[:,:,1] = y_rot + cy

    return _pairs_to_flat(paired, arr.shape[1])


def mask_random_keypoints(arr: np.ndarray, min_mask=1, max_mask=2):
    paired, n_pairs = _pairs_view(arr)
    T = paired.shape[0]

    for t in range(T):
        k = random.randint(min_mask, max_mask)
        idx = np.random.choice(n_pairs, size=k, replace=False)
        paired[t, idx, :] = 0.0

    return _pairs_to_flat(paired, arr.shape[1])


# Temporal augmentations
def temporal_dropout(arr: np.ndarray, pct_range=(0.05, 0.10)):
    T = arr.shape[0]
    pct = np.random.uniform(pct_range[0], pct_range[1])
    n_drop = int(np.round(T * pct))

    if n_drop <= 0:
        return arr

    idx = np.arange(T)
    drop_idx = np.random.choice(idx, size=min(n_drop, T-1), replace=False)
    keep_mask = np.ones(T, dtype=bool)
    keep_mask[drop_idx] = False
    new = arr[keep_mask]

    if new.shape[0] == 0:
        return arr

    return new


def temporal_duplicate(arr: np.ndarray, pct_range=(0.03, 0.05)):
    T = arr.shape[0]
    pct = np.random.uniform(pct_range[0], pct_range[1])
    n_dup = int(np.round(T * pct))

    if n_dup <= 0:
        return arr

    idx = np.arange(T)
    dup_idx = np.random.choice(idx, size=min(n_dup, T), replace=False)
    new_list = []

    for i in range(T):
        new_list.append(arr[i])
        if i in dup_idx:
            new_list.append(arr[i].copy())

    return np.stack(new_list, axis=0)


def speed_warp(arr: np.ndarray, factor_range=(0.95, 1.05)):
    """Resample the sequence length by factor in factor_range using linear interpolation."""
    T, F = arr.shape
    factor = np.random.uniform(factor_range[0], factor_range[1])
    new_T = max(1, int(np.round(T * factor)))

    if new_T == T:
        return arr.copy()

    # original time positions
    orig_t = np.linspace(0, 1, T)
    new_t = np.linspace(0, 1, new_T)
    new = np.zeros((new_T, F), dtype=np.float32)

    for f in range(F):
        new[:, f] = np.interp(new_t, orig_t, arr[:, f])

    return new


def augment_sample_variants(sample: Dict, make_speed_warp_n: int = 2, max_augments: int = 2):
    """Generate augmented variants for a sample. Returns list including the original sample first."""
    arr = sample['keypoints_sequence']
    orig_F = arr.shape[1]
    variants = []

    # original
    variants.append({'keypoints_sequence': arr.copy(), 'gloss': sample.get('gloss', ''), 'source_file': sample.get('source_file')})

    # 1x jitter
    v_jitter = jitter_frames(arr, pct=0.01)
    v_jitter = mask_random_keypoints(v_jitter)
    variants.append({'keypoints_sequence': v_jitter, 'gloss': sample.get('gloss', ''), 'source_file': sample.get('source_file') + '.aug_jitter'})

    # 1x scale or shift + rotate
    if random.random() < 0.5:
        v_scale = scale_and_shift(arr)
    else:
        v_scale = scale_and_shift(arr)

    v_scale = rotate_frames(v_scale, deg_range=(-3,3))
    v_scale = mask_random_keypoints(v_scale)
    variants.append({'keypoints_sequence': v_scale, 'gloss': sample.get('gloss', ''), 'source_file': sample.get('source_file') + '.aug_scale'})

    # optional temporal augmentation (dropout + duplicate)
    v_temp = arr.copy()
    if random.random() < 0.9:
        v_temp = temporal_dropout(v_temp)
    if random.random() < 0.7:
        v_temp = temporal_duplicate(v_temp)

    # ensure dtype and shape
    v_temp = v_temp.astype(np.float32)
    variants.append({'keypoints_sequence': v_temp, 'gloss': sample.get('gloss', ''), 'source_file': sample.get('source_file') + '.aug_temp'})

    # speed_warp 2..3 times
    n_sw = make_speed_warp_n if make_speed_warp_n >= 2 else 2
    for i in range(n_sw):
        v_sw = speed_warp(arr)
        # apply small jitter after warping
        v_sw = jitter_frames(v_sw, pct=0.005)
        variants.append({'keypoints_sequence': v_sw, 'gloss': sample.get('gloss', ''), 'source_file': sample.get('source_file') + f'.aug_speed{i+1}'})

    # limit number of variants if requested
    if max_augments > 0 and len(variants) > max_augments:
        variants = variants[:max_augments]

    return variants

if __name__ == "__main__":
    try:
        # create directories if not exist
        os.makedirs("data/train_data", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("tokenizers", exist_ok=True)

        config = {
            "train_data_folder": "data/train_data",
            "version_model": 29,
            "epochs": 100,
            "batch_size": 8,
            "validation_split": 0.2,
            "input_sequence_length": None,
            "embedding_dim": 256,
            "hidden_dim": 512,  
            "dropout_rate": 0.2,
            "l1_reg": 0.0005,
            "augment": False, # NOTE: Currently, this does not respect the validation split properly and leads the model to overfit by seeing augmented versions of validation samples during training
            "augment_factor": 2,
            "architecture": "multi_attention" # Either baseline or multi_attention or transformer
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
