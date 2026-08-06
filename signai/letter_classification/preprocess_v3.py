import os
# os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np
import mediapipe as mp
import logging
import time
import shutil

"""
v3 TODOS:
    - Dataset Balance: Ensure all classes have the same number of samples (augment underrepresented classes if needed).
    - Z-Flip: Mirror the Z-coordinates for all samples to increase left/right hand diversity.
    - Feature Expansion: Calculate extra features from the keypoints:
        - Distances between Tip for each finger (except the thumb and pinky)
        - Distances between the Tip of the thumb and the Tip and Dip of every other finger.

Pipeline:
    1) Extract keypoints + calculate new features.
    2) Center keypoints (wrist = origin).
    3) Normalize keypoints (e.g., scale by middle finger length).
    4) Split into train / val / test.
    5) Apply mirror + augmentation only on the training split.
    6) Save split datasets as .npz.

"""

# ----------------------------
# Logging Setup
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

# ----------------------------
# Initialization of MediaPipe Hands
# ----------------------------
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5, # with 0.7 we get 3.12% invalid samples, with 0.5 we get 1.67% invalid samples, 
    model_complexity=1
)

# ----------------------------
# Folder of the alphabet-dataset
# ----------------------------
dataset_folder = "signai/letter_classification/data/SignAlphaSet/SignAlphaSet"

# ----------------------------
# Lists to store keypoints and labels
# ----------------------------
all_keypoints = []
all_labels = []
failed_files = []
failed_by_class = {}
failed_by_reason = {
    "read_failed": 0,
    "no_hand": 0,
    "other": 0
}

NUM_LANDMARKS = 21
COORD_DIMS = 3
BASE_KEYPOINT_FEATURES = NUM_LANDMARKS * COORD_DIMS

# ----------------------------
# Extract keypoints from image
# ----------------------------
def extract_keypoints(image_path):
    image = cv2.imread(image_path)
    if image is None:
        logging.warning(f"Could not read image: {image_path}")
        return None, "read_failed"

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        logging.warning(f"No hand detected in image: {image_path}")
        return None, "no_hand"

    hand = results.multi_hand_landmarks[0]
    keypoints = np.array([(lm.x, lm.y, lm.z) for lm in hand.landmark], dtype=np.float32)
    return keypoints, None

def calculate_extra_features(keypoints):
    # Thumb: 4 (tip)
    # Index: 8 (Tip), 7 (DIP), 6 (PIP)
    # Middle: 12 (Tip), 11 (DIP), 10 (PIP)
    # Ring: 16 (Tip), 15 (DIP), 14 (PIP)
    # Pinky: 20 (TIP), 19 (DIP), 18 (PIP)

    # Get the relevant keypoints
    thumb_tip = keypoints[4]
    index_tip = keypoints[8]
    index_dip = keypoints[7]
    middle_tip = keypoints[12]
    middle_dip = keypoints[11]
    ring_tip = keypoints[16]
    ring_dip = keypoints[15]
    ring_pip = keypoints[14]
    pinky_tip = keypoints[20]
    pinky_dip = keypoints[19]
    pinky_pip = keypoints[18]
    middle_pip = keypoints[10]
    index_pip = keypoints[6]

    # create list of extra features to be appended to the original keypoints
    features = []

    # Distances between Tip for each finger (except the thumb and pinky)
    features.append(np.linalg.norm(index_tip - middle_tip))
    features.append(np.linalg.norm(middle_tip - ring_tip))

    # Distances between the Tip of the thumb and the Dip and Pip of every other finger.
    features.append(np.linalg.norm(thumb_tip - index_dip))
    features.append(np.linalg.norm(thumb_tip - index_pip))
    features.append(np.linalg.norm(thumb_tip - middle_dip))
    features.append(np.linalg.norm(thumb_tip - middle_pip))
    features.append(np.linalg.norm(thumb_tip - ring_dip))
    features.append(np.linalg.norm(thumb_tip - ring_pip))
    features.append(np.linalg.norm(thumb_tip - pinky_dip))
    features.append(np.linalg.norm(thumb_tip - pinky_pip))

    # scale to normalize distances by the length of the middle finger (wrist to middle tip)
    wrist = keypoints[0]
    middle_finger_tip = keypoints[12]
    scale = np.linalg.norm(middle_finger_tip - wrist)

    # Normalize each of the features by this scale
    normalized_features = [f / scale for f in features]

    return normalized_features

# ----------------------------
# Center the keypoints
# ----------------------------
def center_keypoints(keypoints):
    # get wrist keypoint (landmark 0) => (0,0,0)
    wrist_keypoint = keypoints[0]
    centered_keypoints = keypoints - wrist_keypoint
    return centered_keypoints

# ----------------------------
# Normalize the keypoints
# ----------------------------
def normalize_keypoints(keypoints):
    wrist_keypoint = keypoints[0]
    middle_finger_tip = keypoints[12]
    scale = np.linalg.norm(middle_finger_tip - wrist_keypoint)
    if scale < 1e-8:
        logging.warning("Distance too small, skipping normalization.")
        return keypoints
    normalized_keypoints = keypoints / scale
    return normalized_keypoints


def sample_to_keypoints(sample):
    sample = np.asarray(sample, dtype=np.float32)
    if sample.ndim == 2 and sample.shape == (NUM_LANDMARKS, COORD_DIMS):
        return sample
    if sample.ndim == 1 and sample.shape[0] >= BASE_KEYPOINT_FEATURES:
        return sample[:BASE_KEYPOINT_FEATURES].reshape(NUM_LANDMARKS, COORD_DIMS)
    raise ValueError(f"Unexpected sample shape {sample.shape}. Expected (21,3) or flat vector >=63.")


def build_sample_features(keypoints):
    keypoints = np.asarray(keypoints, dtype=np.float32)
    keypoints_flat = keypoints.reshape(-1)
    extra_features = np.asarray(calculate_extra_features(keypoints), dtype=np.float32)
    sample_features = np.concatenate([keypoints_flat, extra_features], axis=0)
    expected_length = BASE_KEYPOINT_FEATURES + extra_features.shape[0]
    if sample_features.shape[0] != expected_length:
        raise ValueError(
            f"Unexpected feature length {sample_features.shape[0]}; expected {expected_length}."
        )
    return sample_features


def mirror_keypoints(keypoints, mirror_x=True, mirror_z=False):
    mirrored = keypoints.copy()
    if mirror_x:
        mirrored[:, 0] *= -1.0
    if mirror_z:
        mirrored[:, 2] *= -1.0
    mirrored[0] = 0.0
    return mirrored


def mirror_sample_features(sample, mirror_x=True, mirror_z=False):
    keypoints = sample_to_keypoints(sample)
    mirrored_keypoints = mirror_keypoints(keypoints, mirror_x=mirror_x, mirror_z=mirror_z)
    return build_sample_features(mirrored_keypoints)


def augment_keypoints_diverse(keypoints, rng):
    augmented = keypoints.copy()

    # optional random mirror for extra diversity
    if rng.random() < 0.5:
        augmented[:, 0] *= -1.0

    non_wrist = augmented[1:].copy()

    # choose one of several light augmentation styles
    style = int(rng.integers(0, 4))

    if style == 0:
        # geometric combo (rotation + scale + translation)
        angle = np.deg2rad(rng.uniform(-5.0, 5.0))
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

        scale = rng.uniform(0.90, 1.10)
        tx = rng.choice([-1.0, 1.0]) * rng.uniform(0.01, 0.04)
        ty = rng.choice([-1.0, 1.0]) * rng.uniform(0.01, 0.04)

        xy = non_wrist[:, :2]
        xy = (xy @ rotation_matrix.T) * scale
        xy += np.array([tx, ty], dtype=np.float32)
        non_wrist[:, :2] = xy
        non_wrist[:, 2] *= scale

    elif style == 1:
        # anisotropic scale + tiny global drift
        sx = rng.uniform(0.92, 1.08)
        sy = rng.uniform(0.92, 1.08)
        sz = rng.uniform(0.92, 1.08)
        non_wrist[:, 0] *= sx
        non_wrist[:, 1] *= sy
        non_wrist[:, 2] *= sz
        non_wrist[:, 0] += rng.uniform(-0.03, 0.03)
        non_wrist[:, 1] += rng.uniform(-0.03, 0.03)

    elif style == 2:
        # finger-local jitter (keeps wrist fixed)
        finger_jitter = rng.normal(0.0, 0.012, size=non_wrist.shape).astype(np.float32)
        non_wrist += finger_jitter

    else:
        # mixed light transform
        angle = np.deg2rad(rng.uniform(-5.0, 5.0))
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
        xy = non_wrist[:, :2] @ rotation_matrix.T
        xy += rng.normal(0.0, 0.008, size=xy.shape).astype(np.float32)
        xy += np.array([
            rng.choice([-1.0, 1.0]) * rng.uniform(0.01, 0.03),
            rng.choice([-1.0, 1.0]) * rng.uniform(0.01, 0.03)
        ], dtype=np.float32)
        non_wrist[:, :2] = xy
        non_wrist[:, 2] += rng.normal(0.0, 0.008, size=non_wrist[:, 2].shape).astype(np.float32)

    # always add small noise (excluding wrist)
    noise = rng.normal(0.0, 0.01, size=non_wrist.shape).astype(np.float32)
    non_wrist += noise

    augmented[1:] = non_wrist
    augmented[0] = 0.0
    return augmented


def augment_sample_features_diverse(sample, rng):
    keypoints = sample_to_keypoints(sample)
    augmented_keypoints = augment_keypoints_diverse(keypoints, rng)
    return build_sample_features(augmented_keypoints)


def augment_train_dataset(X_train, y_train, random_seed=42):
    rng = np.random.default_rng(random_seed)

    # 1) Mirror all train samples for X, Z and X+Z
    X_mirror_x = np.array(
        [mirror_sample_features(sample, mirror_x=True, mirror_z=False) for sample in X_train],
        dtype=np.float32
    )
    X_mirror_z = np.array(
        [mirror_sample_features(sample, mirror_x=False, mirror_z=True) for sample in X_train],
        dtype=np.float32
    )
    X_mirror_xz = np.array(
        [mirror_sample_features(sample, mirror_x=True, mirror_z=True) for sample in X_train],
        dtype=np.float32
    )

    X_train_base = np.concatenate([X_train, X_mirror_x, X_mirror_z, X_mirror_xz], axis=0)
    y_train_base = np.concatenate([y_train, y_train, y_train, y_train], axis=0)

    # 2) Balance classes by augmenting only underrepresented classes
    classes, counts = np.unique(y_train_base, return_counts=True)
    target_per_class = int(np.max(counts)) if len(counts) > 0 else 0

    X_extra = []
    y_extra = []
    class_balance_log = {}

    for cls, count in zip(classes, counts):
        deficit = target_per_class - int(count)
        class_balance_log[int(cls)] = {
            "before": int(count),
            "added": max(deficit, 0),
            "after": int(count + max(deficit, 0))
        }

        if deficit <= 0:
            continue

        cls_indices = np.where(y_train_base == cls)[0]
        for _ in range(deficit):
            source_idx = int(rng.choice(cls_indices))
            X_extra.append(augment_sample_features_diverse(X_train_base[source_idx], rng))
            y_extra.append(cls)

    if X_extra:
        X_extra = np.array(X_extra, dtype=np.float32)
        y_extra = np.array(y_extra, dtype=np.int64)
        X_train_final = np.concatenate([X_train_base, X_extra], axis=0)
        y_train_final = np.concatenate([y_train_base, y_extra], axis=0)
    else:
        X_train_final = X_train_base
        y_train_final = y_train_base

    shuffle_idx = np.arange(len(X_train_final))
    rng.shuffle(shuffle_idx)
    X_train_final = X_train_final[shuffle_idx]
    y_train_final = y_train_final[shuffle_idx]

    return X_train_final, y_train_final, class_balance_log

# ----------------------------
# Split dataset into train, val, test and save as npz files
# ----------------------------
def create_dataset(
    X,
    y,
    output_folder="signai/letter_classification/data/processed_dataset",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    augmentation_seed=42
):
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {ratio_sum:.6f} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})."
        )

    if min(train_ratio, val_ratio, test_ratio) < 0.0:
        raise ValueError("Split ratios must be non-negative.")

    # ensure fresh output folder for full reprocessing
    if os.path.exists(output_folder):
        logging.info(f"Existing processed dataset found. Removing folder: {output_folder}")
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # split first on original (non-augmented) samples
    rng = np.random.default_rng(augmentation_seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    X = X[idx]
    y = y[idx]

    N = len(X)
    train_end = int(train_ratio * N)
    val_end = int((train_ratio + val_ratio) * N)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    pre_aug_train_size = len(X_train)
    pre_aug_val_size = len(X_val)
    pre_aug_test_size = len(X_test)

    # Augmentation only for training split:
    # - mirror all training samples (X, Z and X+Z)
    # - class-balance by adding augmented train samples to minority classes
    X_train, y_train, class_balance_log = augment_train_dataset(
        X_train,
        y_train,
        random_seed=augmentation_seed
    )

    if len(X_val) != pre_aug_val_size or len(X_test) != pre_aug_test_size:
        raise RuntimeError("Validation/Test size changed during augmentation. This should never happen.")

    # save as npz in output folder
    np.savez_compressed(os.path.join(output_folder, "train_data.npz"), X=X_train, y=y_train)
    np.savez_compressed(os.path.join(output_folder, "val_data.npz"), X=X_val, y=y_val)
    np.savez_compressed(os.path.join(output_folder, "test_data.npz"), X=X_test, y=y_test)

    logging.info(f"Datasets saved in folder: {output_folder}")
    logging.info(
        f"Train samples: {len(X_train)} (before augmentation: {pre_aug_train_size}; "
        f"mirrored X/Z/X+Z + class-balanced augmentation), "
        f"Val samples: {len(X_val)}, Test samples: {len(X_test)}"
    )
    logging.info(
        f"Original split sizes (before train augmentation): "
        f"train={pre_aug_train_size}, val={pre_aug_val_size}, test={pre_aug_test_size}"
    )

    if class_balance_log:
        logging.info("Train class balancing summary (after mirroring)")
        for cls in sorted(class_balance_log.keys()):
            stats = class_balance_log[cls]
            logging.info(
                f"- class {cls}: before={stats['before']}, added={stats['added']}, after={stats['after']}"
            )


# ----------------------------
# Main processing
# ----------------------------
if __name__ == "__main__":
    alphabet = sorted(os.listdir(dataset_folder))  # ["A","B",...,"Z"]
    class_to_idx = {c: i for i, c in enumerate(alphabet)}

    total = 0
    total_files = 0

    total_start = time.perf_counter()
    extract_time = 0.0
    center_time = 0.0
    normalize_time = 0.0
    save_time = 0.0

    progress_every = 100

    for class_name in alphabet:
        class_path = os.path.join(dataset_folder, class_name)
        if not os.path.isdir(class_path):
            continue

        files = os.listdir(class_path)
        total_files += len(files)
        logging.info(f"Class '{class_name}' has {len(files)} files.")

        class_start = time.perf_counter()
        class_processed = 0
        class_failed = 0

        for file_name in files:
            file_path = os.path.join(class_path, file_name)

            # extract keypoints
            start = time.perf_counter()
            tmp_keypoints, err_reason = extract_keypoints(file_path)
            extract_time += time.perf_counter() - start

            # check if keypoints are fine
            if tmp_keypoints is None:
                logging.warning(
                    f"Skipping file '{file_name}' in class '{class_name}' due to keypoint extraction failure."
                )
                failed_files.append(file_path)
                failed_by_class.setdefault(class_name, []).append(file_path)
                if err_reason in failed_by_reason:
                    failed_by_reason[err_reason] += 1
                else:
                    failed_by_reason["other"] += 1
                class_failed += 1
                continue

            # center and normalize 
            start = time.perf_counter()
            tmp_keypoints = center_keypoints(tmp_keypoints)
            center_time += time.perf_counter() - start

            start = time.perf_counter()
            tmp_keypoints = normalize_keypoints(tmp_keypoints)
            normalize_time += time.perf_counter() - start

            sample_features = build_sample_features(tmp_keypoints)

            # append to dataset
            all_keypoints.append(sample_features)
            all_labels.append(class_to_idx[class_name])
            total += 1
            class_processed += 1

            # processed count logging/progress
            if total % progress_every == 0:
                elapsed = time.perf_counter() - total_start
                pct = (total / total_files * 100.0) if total_files > 0 else 0.0
                rate = total / elapsed if elapsed > 0 else 0.0
                eta = (total_files - total) / rate if rate > 0 else 0.0
                logging.info(
                    f"Progress: {total}/{total_files} ({pct:.2f}%) | "
                    f"rate {rate:.2f} files/s | ETA {eta:.1f}s"
                )

        class_elapsed = time.perf_counter() - class_start
        class_total = len(files)
        class_pct = (class_processed / class_total * 100.0) if class_total > 0 else 0.0
        logging.info(
            f"Class '{class_name}' done: {class_processed}/{class_total} "
            f"({class_pct:.2f}%) | failed {class_failed} | {class_elapsed:.2f}s"
        )

    # =================================================
    # Convert to arrays
    # =================================================
    X = np.array(all_keypoints, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)

    if X.ndim != 2 or X.shape[1] <= BASE_KEYPOINT_FEATURES:
        raise ValueError(
            f"Saved samples must contain keypoints + extra features. Got shape {X.shape}."
        )

    logging.info(f"Valid samples: {len(X)}")
    logging.info(f"Features per sample: {X.shape[1]} (63 keypoints + {X.shape[1] - BASE_KEYPOINT_FEATURES} extra)")
    logging.info(f"Failed samples: {len(failed_files)}")

    # =================================================
    # Split and save train/val/test datasets
    # =================================================
    start = time.perf_counter()
    create_dataset(X, y)
    save_time += time.perf_counter() - start

    total_time = time.perf_counter() - total_start

    # =================================================
    # Summary
    # =================================================
    failed_count = len(failed_files)
    processed_count = len(X)
    total_count = total_files
    failed_pct = (failed_count / total_count * 100.0) if total_count > 0 else 0.0
    processed_pct = (processed_count / total_count * 100.0) if total_count > 0 else 0.0
    avg_extract = extract_time / total_count if total_count > 0 else 0.0
    avg_center = center_time / processed_count if processed_count > 0 else 0.0
    avg_normalize = normalize_time / processed_count if processed_count > 0 else 0.0

    logging.info("=" * 60)
    logging.info("Summary")
    logging.info(f"Total files: {total_count}")
    logging.info(f"Processed (valid): {processed_count} ({processed_pct:.2f}%)")
    logging.info(f"Failed: {failed_count} ({failed_pct:.2f}%)")
    logging.info("Timing (seconds)")
    logging.info(f"- Total: {total_time:.2f}")
    logging.info(f"- Extraction total: {extract_time:.2f} (avg {avg_extract:.4f}/file)")
    logging.info(f"- Center total: {center_time:.2f} (avg {avg_center:.4f}/valid)")
    logging.info(f"- Normalize total: {normalize_time:.2f} (avg {avg_normalize:.4f}/valid)")
    logging.info(f"- Save dataset: {save_time:.2f}")

    if failed_by_reason:
        logging.info("Failed by reason")
        for reason, count in failed_by_reason.items():
            if count == 0:
                continue
            pct = (count / total_count * 100.0) if total_count > 0 else 0.0
            logging.info(f"- {reason}: {count} ({pct:.2f}%)")

    if failed_by_class:
        logging.info("Failed by class")
        for class_name in sorted(failed_by_class.keys()):
            files = failed_by_class[class_name]
            pct = (len(files) / total_count * 100.0) if total_count > 0 else 0.0
            logging.info(f"- {class_name}: {len(files)} ({pct:.2f}%)")
            for file_path in files:
                logging.info(f"  - {file_path}")
    else:
        logging.info("No failed files.")

    logging.info("=" * 60)
