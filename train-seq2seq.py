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
import inspect
import numpy as np
import tensorflow as tf
import pandas as pd
import io
import re
from typing import List, Dict, Union, Tuple, Optional
import time
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Masking, Bidirectional, Concatenate, Embedding, AdditiveAttention, LayerNormalization, MultiHeadAttention, Embedding, DepthwiseConv1D, Lambda, Add

import concurrent.futures
import warnings
import pickle
import jiwer
from sacrebleu.metrics import BLEU
from augemantations import Augmentation

# NOTE: Set random seeds for reproducibility
import keras
keras.utils.set_random_seed(42)

# enable mixed precision for faster training on compatible GPUs
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# expected number of features per frame from preprocessing_train_data.py
# pose(7*3=21) + left_hand(21*3=63) + right_hand(21*3=63) + face(93*3=279) = 426
EXPECTED_FEATURES = 426
# minimal accepted numeric values in a parsed frame (rows below this are likely malformed)
MIN_ACCEPTED_FEATURES = 32

# silence specific DeprecationWarning noise that originates from csv parsing of some files
warnings.filterwarnings("ignore", message="string or file could not be read to its end due to unmatched data")


def normalize_text(s):
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


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
    rows = []

    try:
        rdr = csv.reader(io.StringIO(text))
        first = True
        for row in rdr:
            if not row:
                continue

            # expected header: name,GLOSS,Frame,...
            if first:
                first = False
                if len(row) >= 3 and row[2].strip().lower() == "frame":
                    continue

            # numeric features usually start after 3 metadata columns (name, gloss, frame)
            if len(row) - 3 >= EXPECTED_FEATURES:
                start_idx = 3
            elif len(row) - 2 >= EXPECTED_FEATURES:
                start_idx = 2
            else:
                # fallback: find first numeric token
                start_idx = 0
                while start_idx < len(row):
                    try:
                        float(row[start_idx])
                        break
                    except Exception:
                        start_idx += 1

            vals_list = []
            for tok in row[start_idx:]:
                try:
                    vals_list.append(float(tok))
                except Exception:
                    continue

            if len(vals_list) < MIN_ACCEPTED_FEATURES:
                continue

            vals = np.array(vals_list, dtype=np.float32)
            if vals.size >= EXPECTED_FEATURES:
                frame = vals[:EXPECTED_FEATURES]
            else:
                frame = np.pad(vals, (0, EXPECTED_FEATURES - vals.size), mode='constant', constant_values=0.0)

            if np.isnan(frame).any() or np.isinf(frame).any():
                frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
            rows.append(frame.astype(np.float32))

    except Exception:
        # if csv parse fails, keep pandas fallback below
        rows = []

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
    gloss_test = normalize_text(gloss_text)
    
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
    gloss_texts = [
        normalize_text(s["gloss"]) 
        for s in all_samples 
        if s.get("gloss") is not None and s["gloss"].strip()
    ]
    gloss_texts = [g for g in gloss_texts if g]

    # log data
    logging.info(f"Number of gloss texts for tokenizer: {len(gloss_texts)}")
    unique_glosses = len(set(gloss_texts))
    logging.info(f"Unique gloss samples: {unique_glosses}")

    if len(gloss_texts) == 0:
        raise ValueError("No gloss texts available to build tokenizer. Check your CSV parsing and 'gloss' extraction.")

    # DEBUG: show sample texts
    logging.info("Sample texts before tokenization:")
    for text in gloss_texts[:5]:
        logging.info(f"  {text}")

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
    gloss_texts = [
        normalize_text(s.get("gloss", "")) 
        for s in all_samples 
        if s.get("gloss")
    ]
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

    # Teacher-forcing sanity check: validate shift only on real token positions (ignore right-side padding).
    if decoder_input_data.size > 0 and decoder_target_data.size > 0:
        for row_idx in range(decoder_input_data.shape[0]):
            real_len = int(np.count_nonzero(decoder_input_data[row_idx]))
            if real_len <= 1:
                continue
            inp_shift = decoder_input_data[row_idx, 1:real_len]
            tgt_shift = decoder_target_data[row_idx, :real_len - 1]
            if not np.array_equal(inp_shift, tgt_shift):
                raise ValueError(
                    "Teacher forcing alignment check failed at row "
                    f"{row_idx}: decoder_input/target are not correctly shifted."
                )

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


def greedy_decode(model, encoder_input, tokenizer, max_len):
    """Greedy token-by-token decoding until <end> or max_len is reached."""
    start_id = tokenizer.word_index.get("<start>")
    end_id = tokenizer.word_index.get("<end>")
    if start_id is None or end_id is None:
        raise ValueError("Tokenizer must contain <start> and <end> tokens.")

    if encoder_input.ndim == 2:
        encoder_input_batch = np.expand_dims(encoder_input, axis=0)
    else:
        encoder_input_batch = encoder_input

    decoded_ids = [start_id]
    for _ in range(max_len):
        decoder_input = np.array([decoded_ids], dtype=np.int32)
        logits = model.predict([encoder_input_batch, decoder_input], verbose=0)
        next_id = int(np.argmax(logits[0, len(decoded_ids) - 1]))
        decoded_ids.append(next_id)
        if next_id == end_id:
            break

    words = []
    for token_id in decoded_ids[1:]:
        if token_id == end_id:
            break
        if token_id == 0 or token_id == start_id:
            continue
        words.append(tokenizer.index_word.get(token_id, "<unk>"))
    return " ".join(words).strip()


def _safe_tokens(text: str) -> List[str]:
    return [t for t in str(text).strip().split() if t]


def _ngram_counter(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
    counts: Dict[Tuple[str, ...], int] = {}
    if len(tokens) < n or n <= 0:
        return counts
    for i in range(len(tokens) - n + 1):
        ng = tuple(tokens[i:i + n])
        counts[ng] = counts.get(ng, 0) + 1
    return counts


def _rouge_n_pair(ref_tokens: List[str], hyp_tokens: List[str], n: int) -> Tuple[float, float, float]:
    ref_counts = _ngram_counter(ref_tokens, n)
    hyp_counts = _ngram_counter(hyp_tokens, n)

    ref_total = sum(ref_counts.values())
    hyp_total = sum(hyp_counts.values())

    if ref_total == 0 and hyp_total == 0:
        return 1.0, 1.0, 1.0
    if ref_total == 0 or hyp_total == 0:
        return 0.0, 0.0, 0.0

    overlap = 0
    for ng, c in hyp_counts.items():
        overlap += min(c, ref_counts.get(ng, 0))

    precision = overlap / hyp_total if hyp_total > 0 else 0.0
    recall = overlap / ref_total if ref_total > 0 else 0.0
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def _lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def _rouge_l_pair(ref_tokens: List[str], hyp_tokens: List[str]) -> Tuple[float, float, float]:
    if len(ref_tokens) == 0 and len(hyp_tokens) == 0:
        return 1.0, 1.0, 1.0
    if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
        return 0.0, 0.0, 0.0

    lcs = _lcs_length(ref_tokens, hyp_tokens)
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def compute_corpus_rouge(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    r1_p, r1_r, r1_f = [], [], []
    r2_p, r2_r, r2_f = [], [], []
    rl_p, rl_r, rl_f = [], [], []

    for ref, hyp in zip(references, hypotheses):
        ref_t = _safe_tokens(ref)
        hyp_t = _safe_tokens(hyp)

        p, r, f = _rouge_n_pair(ref_t, hyp_t, n=1)
        r1_p.append(p)
        r1_r.append(r)
        r1_f.append(f)

        p, r, f = _rouge_n_pair(ref_t, hyp_t, n=2)
        r2_p.append(p)
        r2_r.append(r)
        r2_f.append(f)

        p, r, f = _rouge_l_pair(ref_t, hyp_t)
        rl_p.append(p)
        rl_r.append(r)
        rl_f.append(f)

    return {
        "rouge1_p": float(np.mean(r1_p)) if r1_p else 0.0,
        "rouge1_r": float(np.mean(r1_r)) if r1_r else 0.0,
        "rouge1_f": float(np.mean(r1_f)) if r1_f else 0.0,
        "rouge2_p": float(np.mean(r2_p)) if r2_p else 0.0,
        "rouge2_r": float(np.mean(r2_r)) if r2_r else 0.0,
        "rouge2_f": float(np.mean(r2_f)) if r2_f else 0.0,
        "rougel_p": float(np.mean(rl_p)) if rl_p else 0.0,
        "rougel_r": float(np.mean(rl_r)) if rl_r else 0.0,
        "rougel_f": float(np.mean(rl_f)) if rl_f else 0.0,
    }


class SignLanguageEvaluationCallback(tf.keras.callbacks.Callback):
    """Evaluate text-generation quality on validation set after each epoch."""

    def __init__(
            self,
            val_encoder: np.ndarray,
            val_references: List[str],
            tokenizer,
            max_len: int,
            sample_size: Optional[int] = 256,
            seed: int = 42,
            log_file_path: Optional[str] = None,
    ):
        super().__init__()
        self.val_encoder = val_encoder
        self.val_references = val_references
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.log_file_path = log_file_path

        n = len(self.val_references)
        if n == 0:
            self.eval_indices = np.array([], dtype=np.int64)
        elif sample_size is None or sample_size <= 0 or sample_size >= n:
            self.eval_indices = np.arange(n, dtype=np.int64)
        else:
            rng = np.random.default_rng(seed)
            self.eval_indices = rng.choice(np.arange(n), size=int(sample_size), replace=False)

        # Pick 5 fixed indices for visually tracing translations across epochs
        if len(self.eval_indices) >= 5:
            rng_fixed = np.random.default_rng(seed)
            self.fixed_5_indices = rng_fixed.choice(self.eval_indices, size=5, replace=False)
        else:
            self.fixed_5_indices = self.eval_indices

        # Initialize the log file with a header if provided
        if self.log_file_path:
            import os
            os.makedirs(os.path.dirname(self.log_file_path) or ".", exist_ok=True)
            with open(self.log_file_path, "w", encoding="utf-8") as f:
                f.write("=== Training Logs and Progress ===\n\n")

        self.bleu = BLEU(effective_order=True)
        self.bleu1 = BLEU(max_ngram_order=1, effective_order=True)
        self.bleu2 = BLEU(max_ngram_order=2, effective_order=True)
        self.bleu3 = BLEU(max_ngram_order=3, effective_order=True)
        self.bleu4 = BLEU(max_ngram_order=4, effective_order=True)

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if self.eval_indices.size == 0:
            return

        refs = []
        hyps = []

        for i in self.eval_indices:
            pred = greedy_decode(self.model, self.val_encoder[i], self.tokenizer, self.max_len)
            ref = self.val_references[i]
            refs.append(normalize_text(ref) if ref else "")
            hyps.append(normalize_text(pred) if pred else "")

        val_wer = float(jiwer.wer(refs, hyps))
        val_bleu = float(self.bleu.corpus_score(hyps, [refs]).score)
        val_bleu1 = float(self.bleu1.corpus_score(hyps, [refs]).score)
        val_bleu2 = float(self.bleu2.corpus_score(hyps, [refs]).score)
        val_bleu3 = float(self.bleu3.corpus_score(hyps, [refs]).score)
        val_bleu4 = float(self.bleu4.corpus_score(hyps, [refs]).score)
        rouge_scores = compute_corpus_rouge(refs, hyps)

        logs["val_wer"] = val_wer
        logs["val_bleu"] = val_bleu
        logs["val_bleu1"] = val_bleu1
        logs["val_bleu2"] = val_bleu2
        logs["val_bleu3"] = val_bleu3
        logs["val_bleu4"] = val_bleu4

        logs["val_rouge1_p"] = rouge_scores["rouge1_p"]
        logs["val_rouge1_r"] = rouge_scores["rouge1_r"]
        logs["val_rouge1_f"] = rouge_scores["rouge1_f"]
        logs["val_rouge2_p"] = rouge_scores["rouge2_p"]
        logs["val_rouge2_r"] = rouge_scores["rouge2_r"]
        logs["val_rouge2_f"] = rouge_scores["rouge2_f"]
        logs["val_rougel_p"] = rouge_scores["rougel_p"]
        logs["val_rougel_r"] = rouge_scores["rougel_r"]
        logs["val_rougel_f"] = rouge_scores["rougel_f"]

        log_str = (
            f"Epoch {epoch + 1} text-metrics | "
            f"WER={val_wer:.4f} BLEU={val_bleu:.2f} B1={val_bleu1:.2f} B2={val_bleu2:.2f} "
            f"B3={val_bleu3:.2f} B4={val_bleu4:.2f} "
            f"R1-F={rouge_scores['rouge1_f']:.4f} R2-F={rouge_scores['rouge2_f']:.4f} RL-F={rouge_scores['rougel_f']:.4f}"
        )
        
        logging.info(log_str)

        # Build nice string layout for the 5 fixed samples
        samples_out_lines = []
        samples_out_lines.append(f"\n--- Epoch {epoch + 1}: 5 Test Translation Samples ---")
        for idx_sample in self.fixed_5_indices:
            fixed_pred = greedy_decode(self.model, self.val_encoder[idx_sample], self.tokenizer, self.max_len)
            fixed_ref = self.val_references[idx_sample]
            samples_out_lines.append(f"  True Translation : {fixed_ref}")
            samples_out_lines.append(f"  Model Prediction : {fixed_pred}")
            samples_out_lines.append("  " + "-"*40)
        
        samples_str = "\n".join(samples_out_lines)

        # Print to console for visual feedback
        print(samples_str)

        # Append to log file if path exists
        if self.log_file_path:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(f"\n================ EPOCH {epoch + 1} ================\n")
                
                # Also log loss/acc if available in standard `logs`
                general_metrics = " | ".join([f"{k}={v:.4f}" for k, v in logs.items() if not k.startswith("val_") and not "rouge" in k and not "bleu" in k])
                if general_metrics:
                    f.write(f"Train Metrics: {general_metrics}\n")
                f.write(f"Eval  Metrics: WER={val_wer:.4f} BLEU={val_bleu:.2f} B1={val_bleu1:.2f} B2={val_bleu2:.2f} B3={val_bleu3:.2f} B4={val_bleu4:.2f}\n")
                f.write(f"ROUGE Metrics: R1-F={rouge_scores['rouge1_f']:.4f} R2-F={rouge_scores['rouge2_f']:.4f} RL-F={rouge_scores['rougel_f']:.4f}\n")
                
                f.write(samples_str + "\n\n")

class Seq2SeqBatchSequence(tf.keras.utils.Sequence):
    """Batch-wise loader to avoid materializing the full training tensors on GPU."""

    def __init__(
            self,
            encoder_data: np.ndarray,
            decoder_data: np.ndarray,
            target_data: np.ndarray,
            indices: np.ndarray,
            batch_size: int,
            shuffle: bool = True,
            seed: Optional[int] = 42,
            augmenter: Optional[Augmentation] = None,
            augment_each_epoch: bool = False,
                augment_factor: Optional[int] = None,
                keep_original: Optional[bool] = None,
    ):
        self.base_indices = np.asarray(indices, dtype=np.int64)

        # Keep an immutable snapshot of the ORIGINAL train split as augmentation base.
        # This guarantees each epoch starts from normal data, not previous augmented data.
        self.base_encoder = np.asarray(encoder_data[self.base_indices]).copy()
        self.base_decoder = np.asarray(decoder_data[self.base_indices]).copy()
        self.base_target = np.asarray(target_data[self.base_indices]).copy()

        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng(seed)
        self.augmenter = augmenter
        self.augment_each_epoch = bool(augment_each_epoch)
        self.augment_factor = None if augment_factor is None else max(0, int(augment_factor))
        self.keep_original = None if keep_original is None else bool(keep_original)
        self.epoch_encoder = None
        self.epoch_decoder = None
        self.epoch_target = None
        self.indices = np.array([], dtype=np.int64)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.indices))
        batch_idx = self.indices[start:end]
        x_enc = self.epoch_encoder[batch_idx]
        x_dec = self.epoch_decoder[batch_idx]
        y = self.epoch_target[batch_idx]
        return (x_enc, x_dec), y

    def _rebuild_epoch_data(self):
        base_enc = self.base_encoder
        base_dec = self.base_decoder
        base_tgt = self.base_target

        if self.augment_each_epoch and self.augmenter is not None:
            effective_factor = (
                self.augment_factor
                if self.augment_factor is not None
                else max(0, int(getattr(self.augmenter, "augment_factor", 0)))
            )
            if effective_factor > 0:
                epoch_enc, epoch_dec, epoch_tgt = self.augmenter.augment_training_split(
                    base_enc,
                    base_dec,
                    base_tgt,
                    augment_factor=self.augment_factor,
                    keep_original=self.keep_original,
                )
            else:
                epoch_enc, epoch_dec, epoch_tgt = base_enc, base_dec, base_tgt
        else:
            epoch_enc, epoch_dec, epoch_tgt = base_enc, base_dec, base_tgt

        self.epoch_encoder = np.asarray(epoch_enc, dtype=np.float32)
        self.epoch_decoder = np.asarray(epoch_dec)
        self.epoch_target = np.asarray(epoch_tgt)
        if self.epoch_encoder.shape[0] != self.epoch_decoder.shape[0] or self.epoch_encoder.shape[0] != self.epoch_target.shape[0]:
            raise ValueError(
                "Augmented split size mismatch: "
                f"enc={self.epoch_encoder.shape[0]}, "
                f"dec={self.epoch_decoder.shape[0]}, "
                f"tgt={self.epoch_target.shape[0]}"
            )
        self.indices = np.arange(self.epoch_encoder.shape[0], dtype=np.int64)

    def on_epoch_end(self):
        self._rebuild_epoch_data()
        if self.shuffle:
            self.rng.shuffle(self.indices)


# NOTE: Improved Seq2Seq Model with Multi-Head Attention and Layer Normalization
def build_seq2seq_model_multi_attention(
        max_frames, num_features, vocab_size,
        embedding_dim=512,
        encoder_units=512,
        decoder_units=1024,
    dropout_rate=0.4,
    recurrent_dropout_rate=0.0,
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
    # Keep argument for backwards compatibility while forcing cuDNN-safe recurrent dropout.
    recurrent_dropout_rate = 0.0

    # ===== ENCODER =====
    encoder_inputs = Input(shape=(None, num_features), name="encoder_inputs")
    encoder_masked_inputs = Masking(mask_value=0.0, name="encoder_masking")(encoder_inputs)

    # define masking
    lstm_mask = Lambda(
        lambda t: tf.reduce_sum(tf.abs(t), axis=-1) > 1e-6,
        name="encoder_mask"
    )(encoder_masked_inputs)

    # Spatial projection: helps model focus on important keypoint relationships
    x = Dense(encoder_units * 2, activation="relu", name="encoder_projection")(encoder_masked_inputs)
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
            activation="tanh",
            recurrent_activation="sigmoid",
            dropout=dropout_rate,
            recurrent_dropout=0.0,
        ),
        name="encoder_bidirectional"
    )

    encoder_outputs_and_states = encoder_lstm(
        x,
        mask=lstm_mask   # adding mask to LSTM for better handling of variable-length sequences!
    )
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
        activation="tanh",
        recurrent_activation="sigmoid",
        dropout=dropout_rate,
        recurrent_dropout=0.0,
        name="decoder_lstm"
    )

    decoder_outputs, _, _ = decoder_lstm(
        decoder_embedding,
        initial_state=[state_h, state_c],
        mask=decoder_mask,  # Pass mask to LSTM
    )
    
    if use_layer_norm:
        decoder_outputs = LayerNormalization(name="decoder_lstm_norm")(decoder_outputs)

    def create_cross_mask(inputs):
        dec_mask, enc_mask = inputs
        
        dec_mask = tf.cast(dec_mask[:, :, tf.newaxis], tf.bool)   # (B, T_dec, 1)
        enc_mask = tf.cast(enc_mask[:, tf.newaxis, :], tf.bool)   # (B, 1, T_enc)
        
        # Keep attention mask boolean to avoid float/bool casting edge cases.
        return tf.cast(tf.logical_and(dec_mask, enc_mask), tf.bool)  # (B, T_dec, T_enc)

    cross_mask = Lambda(create_cross_mask, name="cross_mask")([decoder_mask, lstm_mask])

    # ===== ATTENTION =====
    # Multi-head attention: learns different alignment patterns simultaneously
    if (encoder_units * 2) % num_attention_heads != 0:
        raise ValueError(
            f"encoder_units*2 ({encoder_units * 2}) must be divisible by num_attention_heads ({num_attention_heads})."
        )

    cross_attention_layer = MultiHeadAttention(
        num_heads=num_attention_heads,
        key_dim=(encoder_units * 2) // num_attention_heads,
        dropout=dropout_rate,
        name="multi_head_attention"
    )

    attention = cross_attention_layer(
        query=decoder_outputs,
        value=encoder_outputs,
        key=encoder_outputs,
        attention_mask=cross_mask
    )

    attention = Add()([decoder_outputs, attention])  # Residual connection for better gradient flow 

    if use_layer_norm:
        attention = LayerNormalization(name="attention_norm")(attention)

    decoder_combined = Concatenate(axis=-1, name="decoder_concat")([attention, decoder_outputs])
    
    # Deeper feedforward network for better expressiveness
    decoder_combined = Dense(decoder_units, activation="relu", name="decoder_ff1")(decoder_combined)
    decoder_combined = Dropout(dropout_rate, name="decoder_dropout")(decoder_combined)
    if use_layer_norm:
        decoder_combined = LayerNormalization(name="decoder_ff_norm")(decoder_combined)

    # Numerical stability
    decoder_dense = Dense(vocab_size, activation="softmax", dtype="float32", name="decoder_dense")
    final_outputs = decoder_dense(decoder_combined)

    model = tf.keras.Model([encoder_inputs, decoder_inputs], final_outputs, name="seq2seq_improved")
    return model


# NOTE: Factory function to build different seq2seq architectures
def build_seq2seq_model(
        max_frames, num_features, vocab_size,
        embedding_dim=128,
        encoder_units=256,
        decoder_units=512,
    dropout_rate=0.4,
    recurrent_dropout_rate=0.0,
        architecture="multi_attention",
        use_layer_norm=True,
        use_multi_head_attention=True,
        num_attention_heads=8,
        num_encoder_layers=4,
        num_decoder_layers=4
):
    """
    Factory function for the slimmed-down model setup.
    
    Args:
        architecture: kept for backward compatibility and ignored.
    """
    return build_seq2seq_model_multi_attention(
        max_frames=max_frames,
        num_features=num_features,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        encoder_units=encoder_units,
        decoder_units=decoder_units,
        dropout_rate=dropout_rate,
        recurrent_dropout_rate=recurrent_dropout_rate,
        use_layer_norm=use_layer_norm,
        num_attention_heads=num_attention_heads,
    )


def train_main(
        train_data_folder,
        version_model=31,
        epochs=10,
        batch_size=64,
        validation_split=0.1,
        input_sequence_length=None,
        embedding_dim=512,
        hidden_dim=1024,
        dropout_rate=0.4,
        l1_reg=0.001,
        augment_train_each_epoch=True,
        augment_factor=1,
        architecture="multi_attention",  # "baseline", "multi_attention", or "transformer"
        metrics_sample_size=256,
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

        # 4.1 Build explicit 10% unseen validation split (instead of Keras validation_split)
        n_samples = encoder_input_data.shape[0]
        if n_samples != decoder_input_data.shape[0] or n_samples != decoder_target_data.shape[0]:
            raise ValueError("Encoder/decoder sample count mismatch after preprocessing.")

        gloss_refs = [s["gloss"] for s in samples]

        if n_samples < 2:
            raise ValueError("Need at least 2 samples to create a train/validation split.")

        val_fraction = 0.1
        val_size = max(1, int(round(n_samples * val_fraction)))
        val_size = min(val_size, n_samples - 1)

        rng = np.random.default_rng(42)
        indices = np.arange(n_samples)
        rng.shuffle(indices)

        val_idx = indices[:val_size]
        train_idx = indices[val_size:]

        val_gloss_refs = [gloss_refs[i] for i in val_idx]

        logging.info(
            "Split data into train/val with fixed 10%% validation: train=%d, val=%d",
            len(train_idx),
            len(val_idx),
        )

        encoder_gib = encoder_input_data.nbytes / (1024 ** 3)
        logging.info("Full encoder tensor size in RAM: %.2f GiB", encoder_gib)

        train_sequence = Seq2SeqBatchSequence(
            encoder_data=encoder_input_data,
            decoder_data=decoder_input_data,
            target_data=decoder_target_data,
            indices=train_idx,
            batch_size=batch_size,
            shuffle=True,
            seed=42,
            augmenter=Augmentation(seed=42, augment_factor=augment_factor, keep_original=True),
            augment_each_epoch=augment_train_each_epoch,
            augment_factor=None,
            keep_original=None,
        )
        val_sequence = Seq2SeqBatchSequence(
            encoder_data=encoder_input_data,
            decoder_data=decoder_input_data,
            target_data=decoder_target_data,
            indices=val_idx,
            batch_size=min(batch_size, 8),
            shuffle=False,
            seed=42,
            augmenter=None,
            augment_each_epoch=False,
            augment_factor=0,
        )
        logging.info(
            "Epoch-wise augmentation: train_only=%s, factor=%d",
            bool(augment_train_each_epoch),
            int(augment_factor),
        )

        # 5. create model
        target_vocab_size = len(tokenizer.word_index) + 1
        logging.info(f"Building model with architecture: {architecture}")
        model = build_seq2seq_model(
            max_frames=used_max_frames,
            num_features=input_feature_dim,
            vocab_size=target_vocab_size,
            embedding_dim=embedding_dim,
            encoder_units=embedding_dim,
            decoder_units=hidden_dim,
            dropout_rate=dropout_rate,
            recurrent_dropout_rate=0.0,
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

        initial_learning_rate = 0.0001 # adjusted learning rate for more stable training; old was 0.001
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
        loss_kwargs = {
            "from_logits": False,
            "ignore_class": 0,  # Ignore padding token (0) in loss calculation
        }
        if "label_smoothing" in inspect.signature(tf.keras.losses.SparseCategoricalCrossentropy).parameters:
            loss_kwargs["label_smoothing"] = 0.1
            logging.info("Enabled label_smoothing=0.1 for SparseCategoricalCrossentropy")
        else:
            logging.info("SparseCategoricalCrossentropy has no label_smoothing in this TensorFlow version; skipping it")

        loss = tf.keras.losses.SparseCategoricalCrossentropy(**loss_kwargs)
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
                patience=30,  # Increased patience for more stable training
                restore_best_weights=True,
                min_delta=0.0005
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
            ),
            SignLanguageEvaluationCallback(
                val_encoder=encoder_input_data[val_idx],
                val_references=val_gloss_refs,
                tokenizer=tokenizer,
                max_len=decoder_target_data.shape[1],
                sample_size=metrics_sample_size,
                seed=42,
                log_file_path=f'./logs/model_v{version_model}/training_logs.txt'
            ),
        ]
        # training
        history = model.fit(
            train_sequence,
            epochs=epochs,
            validation_data=val_sequence,
            callbacks=callbacks,
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

        config = {
            "train_data_folder": "data/train_data",
            "version_model": 38_1,
            "epochs": 200,
            "batch_size": 64,
            "validation_split": 0.1,
            "input_sequence_length": 300,
            "embedding_dim": 256,
            "hidden_dim": 512,  
            "dropout_rate": 0.4,
            "l1_reg": 0.0001,
            "augment_train_each_epoch": True,
            "augment_factor": 2,
            "architecture": "multi_attention", # Either baseline or multi_attention or transformer
            "metrics_sample_size": 128,
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

        # Plot text generation metrics (WER, all BLEU forms, all ROUGE forms)
        epochs_x = np.arange(1, len(history.history.get('loss', [])) + 1)

        plt.figure(figsize=(14, 10))

        plt.subplot(2, 2, 1)
        if 'val_wer' in history.history:
            plt.plot(epochs_x, history.history['val_wer'], label='WER', color='red')
        plt.title('Validation WER')
        plt.xlabel('Epoch')
        plt.ylabel('WER')
        plt.legend()

        plt.subplot(2, 2, 2)
        bleu_keys = ['val_bleu', 'val_bleu1', 'val_bleu2', 'val_bleu3', 'val_bleu4']
        for k in bleu_keys:
            if k in history.history:
                plt.plot(epochs_x, history.history[k], label=k)
        plt.title('Validation BLEU (All Forms)')
        plt.xlabel('Epoch')
        plt.ylabel('BLEU Score (0-100)')
        plt.legend()

        plt.subplot(2, 2, 3)
        rouge_p_keys = ['val_rouge1_p', 'val_rouge2_p', 'val_rougel_p']
        rouge_r_keys = ['val_rouge1_r', 'val_rouge2_r', 'val_rougel_r']
        rouge_f_keys = ['val_rouge1_f', 'val_rouge2_f', 'val_rougel_f']
        for k in rouge_p_keys + rouge_r_keys + rouge_f_keys:
            if k in history.history:
                plt.plot(epochs_x, history.history[k], label=k)
        plt.title('Validation ROUGE (P/R/F All Forms)')
        plt.xlabel('Epoch')
        plt.ylabel('ROUGE')
        plt.legend(loc='best', fontsize=8)

        plt.subplot(2, 2, 4)
        if 'val_bleu4' in history.history:
            plt.plot(epochs_x, history.history['val_bleu4'], label='BLEU-4', linewidth=2)
        if 'val_rougel_f' in history.history:
            plt.plot(epochs_x, np.array(history.history['val_rougel_f']) * 100.0, label='ROUGE-L F1 * 100', linewidth=2)
        if 'val_wer' in history.history:
            wer_as_accuracy = (1.0 - np.array(history.history['val_wer'])) * 100.0
            plt.plot(epochs_x, wer_as_accuracy, label='1-WER (%)', linewidth=2)
        plt.title('High-Level Text Metric Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Comparable Scale (percent)')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'models/text_metrics_history_v{config["version_model"]}.png')
        plt.close()

        # One combined chart with all key text metrics in comparison
        plt.figure(figsize=(14, 6))
        comparison_series = [
            ('val_bleu', 1.0, 'BLEU'),
            ('val_bleu1', 1.0, 'BLEU-1'),
            ('val_bleu2', 1.0, 'BLEU-2'),
            ('val_bleu3', 1.0, 'BLEU-3'),
            ('val_bleu4', 1.0, 'BLEU-4'),
            ('val_rouge1_f', 100.0, 'ROUGE-1 F1 * 100'),
            ('val_rouge2_f', 100.0, 'ROUGE-2 F1 * 100'),
            ('val_rougel_f', 100.0, 'ROUGE-L F1 * 100'),
            ('val_wer', -100.0, '1-WER (%)'),
        ]

        for key, factor, label in comparison_series:
            if key not in history.history:
                continue
            vals = np.array(history.history[key], dtype=np.float32)
            if key == 'val_wer':
                vals = (1.0 - vals) * 100.0
            else:
                vals = vals * factor
            plt.plot(epochs_x, vals, label=label)

        plt.title('All Text Metrics Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Score (percent-like scale)')
        plt.legend(loc='best', fontsize=9)
        plt.tight_layout()
        plt.savefig(f'models/text_metrics_comparison_v{config["version_model"]}.png')
        plt.close()

        logging.info("Training completed successfully")

    except Exception as e:
        logging.error(f"Program terminated with error: {str(e)}")
        raise
