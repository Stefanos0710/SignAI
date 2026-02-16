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
    4) Mirror Z (flip along the Z-axis).
    5) Mirror X (flip along X-axis, optional, usually included in data augmentation).
    6) Optional light augmentation (noise, rotation, scaling) for training data only.
    7) Shuffle and save the final dataset as .npz.

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
    min_detection_confidence=0.5, 
    model_complexity=1
)

# ----------------------------
# Folder of the alphabet-dataset
# ----------------------------
dataset_folder = "SignAlphaSet/data/SignAlphaSet/SignAlphaSet"

# ----------------------------
# Lists to store keypoints and labels
# ----------------------------
all_samples = [] # Stores flat vectors (coords + features)
all_labels = []
failed_files = []
failed_by_class = {}
failed_by_reason = {
    "read_failed": 0,
    "no_hand": 0,
    "other": 0
}

# ----------------------------
# FEATURE CALCULATION
# ----------------------------
def calculate_extra_features(keypoints):
    """
    Berechnet zusätzliche Distanz-Features basierend auf den (21, 3) Keypoints.
    Gibt ein Array mit den Features zurück.
    """
    # Keypoint Indices:
    # 0: Wrist
    # 4: Thumb Tip
    # 8: Index Tip, 7: Index Dip
    # 12: Middle Tip, 11: Middle Dip
    # 16: Ring Tip, 15: Ring Dip
    # 20: Pinky Tip, 19: Pinky Dip

    features = []
    
    thumb_tip = keypoints[4]
    index_tip = keypoints[8]
    index_dip = keypoints[7]
    index_pip = keypoints[6]
    
    middle_tip = keypoints[12]
    middle_dip = keypoints[11]
    middle_pip = keypoints[10]
    
    ring_tip = keypoints[16]
    ring_dip = keypoints[15]
    ring_pip = keypoints[14]
    
    pinky_tip = keypoints[20]
    pinky_dip = keypoints[19]
    pinky_pip = keypoints[18]


    # 1. Distances between Tip for each finger (except thumb and pinky)
    features.append(np.linalg.norm(index_tip - middle_tip))
    features.append(np.linalg.norm(middle_tip - ring_tip))

    # 2. Distances between Thumb Tip and (Tip and Dip) of every other finger
    # Fingers: Index (8,7,6), Middle (12,11,10), Ring (16,15,14), Pinky (20,19,18)
    
    # Index
    features.append(np.linalg.norm(thumb_tip - index_dip))
    features.append(np.linalg.norm(thumb_tip - index_pip))
    
    # Middle
    features.append(np.linalg.norm(thumb_tip - middle_dip))
    features.append(np.linalg.norm(thumb_tip - middle_pip))
    
    # Ring
    features.append(np.linalg.norm(thumb_tip - ring_dip))
    features.append(np.linalg.norm(thumb_tip - ring_pip))
    
    # Pinky
    features.append(np.linalg.norm(thumb_tip - pinky_dip))
    features.append(np.linalg.norm(thumb_tip - pinky_pip))

    return np.array(features, dtype=np.float32)

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

# ----------------------------
# Center the keypoints
# ----------------------------
def center_keypoints(keypoints):
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

# ----------------------------
# Mirroring Functions (working on flat vectors)
# Vector layout: [x1, y1, z1, ..., x21, y21, z21, feat1, feat2, ...]
# Coordinates are first 63 elements.
# ----------------------------
def mirror_vector(vector, flip_x=False, flip_z=False):
    new_vec = vector.copy()
    coords = new_vec[:63].reshape(21, 3)
    
    if flip_x:
        coords[:, 0] *= -1.0 # Flip X
    
    if flip_z:
        coords[:, 2] *= -1.0 # Flip Z
        
    new_vec[:63] = coords.flatten()
    # Features (distances) are invariant to mirroring, so they stay the same!
    return new_vec

def augment_vector_light(vector, rng):
    """
    Leichte Augmentierung für Balancing.
    """
    coords = vector[:63].reshape(21, 3).copy()
    features = vector[63:].copy()
    
    non_wrist = coords[1:].copy()

    # Rotation
    angle = np.deg2rad(rng.uniform(-5.0, 5.0))
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    scale = rng.uniform(0.95, 1.05)
    
    xy = non_wrist[:, :2]
    xy = (xy @ rotation_matrix.T) * scale
    
    # Translation
    tx = rng.uniform(-0.02, 0.02)
    ty = rng.uniform(-0.02, 0.02)
    xy += np.array([tx, ty], dtype=np.float32)
    
    non_wrist[:, :2] = xy
    non_wrist[:, 2] *= scale
    
    # Noise
    noise = rng.normal(0.0, 0.005, size=non_wrist.shape).astype(np.float32)
    non_wrist += noise
    
    coords[1:] = non_wrist
    coords[0] = 0.0
    # Features skalieren (da Distanzen linear mit Scale skalieren)
    features *= scale
    
    return np.concatenate([coords.flatten(), features])


def balance_classes(X, y, rng):
    """
    Bringt alle Klassen auf die Anzahl der größten Klasse.
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    max_count = np.max(counts)
    logging.info(f"Balancing all classes to {max_count} samples...")
    
    X_balanced = []
    y_balanced = []
    
    for cls in unique_classes:
        indices = np.where(y == cls)[0]
        samples = X[indices]
        current_count = len(samples)
        
        # Originale hinzufügen
        X_balanced.append(samples)
        y_balanced.append(np.full(current_count, cls, dtype=y.dtype))
        
        diff = max_count - current_count
        if diff > 0:
            # Zufällige Auswahl zum Auffüllen
            extra_indices = rng.choice(current_count, size=diff, replace=True)
            extra_samples = []
            for idx in extra_indices:
                # Augmentieren, damit keine exakten Dubletten entstehen
                aug = augment_vector_light(samples[idx], rng)
                extra_samples.append(aug)
            
            X_balanced.append(np.array(extra_samples))
            y_balanced.append(np.full(diff, cls, dtype=y.dtype))
            
    return np.concatenate(X_balanced, axis=0), np.concatenate(y_balanced, axis=0)


# ----------------------------
# Split dataset into train, val, test and save as npz files
# ----------------------------
def create_dataset_v3(
    X,
    y,
    output_folder="SignAlphaSet/data/processed_dataset",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    augmentation_seed=42
):
    if os.path.exists(output_folder):
        logging.info(f"Existing processed dataset found. Removing folder: {output_folder}")
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    rng = np.random.default_rng(augmentation_seed)

    # 1. Expand Dataset with Mirrors (Original, Z-Flip, X-Flip, XZ-Flip) for ALL samples
    logging.info("Generating mirrors (Original, Z-Flip, X-Flip, XZ-Flip)...")
    X_aug = []
    y_aug = []
    
    for i in range(len(X)):
        sample = X[i]
        label = y[i]
        
        # 1. Original
        X_aug.append(sample)
        y_aug.append(label)
        
        # 2. Z-Flip
        X_aug.append(mirror_vector(sample, flip_x=False, flip_z=True))
        y_aug.append(label)
        
        # 3. X-Flip
        X_aug.append(mirror_vector(sample, flip_x=True, flip_z=False))
        y_aug.append(label)
        
        # 4. XZ-Flip
        X_aug.append(mirror_vector(sample, flip_x=True, flip_z=True))
        y_aug.append(label)
        
    X_full = np.array(X_aug, dtype=np.float32)
    y_full = np.array(y_aug, dtype=np.int64)
    
    logging.info(f"Dataset size after mirroring: {len(X_full)}")

    # 2. Split
    # Shuffle first
    idx = np.arange(len(X_full))
    rng.shuffle(idx)
    X_full = X_full[idx]
    y_full = y_full[idx]

    N = len(X_full)
    train_end = int(train_ratio * N)
    val_end = int((train_ratio + val_ratio) * N)

    X_train, y_train = X_full[:train_end], y_full[:train_end]
    X_val, y_val = X_full[train_end:val_end], y_full[train_end:val_end]
    X_test, y_test = X_full[val_end:], y_full[val_end:]

    # 3. Balance (only Train set to avoid leaking augmented/synthetic data into val/test)
    logging.info("Balancing Training Set...")
    X_train, y_train = balance_classes(X_train, y_train, rng)
    
    # Final Shuffle Train
    idx_train = np.arange(len(X_train))
    rng.shuffle(idx_train)
    X_train = X_train[idx_train]
    y_train = y_train[idx_train]

    # save as npz in output folder
    np.savez_compressed(os.path.join(output_folder, "train_data.npz"), X=X_train, y=y_train)
    np.savez_compressed(os.path.join(output_folder, "val_data.npz"), X=X_val, y=y_val)
    np.savez_compressed(os.path.join(output_folder, "test_data.npz"), X=X_test, y=y_test)

    logging.info(f"Datasets saved in folder: {output_folder}")
    logging.info(
        f"Train samples: {len(X_train)}, "
        f"Val samples: {len(X_val)}, Test samples: {len(X_test)}"
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

            # Process: Center -> Normalize -> Features -> Flatten
            tmp_keypoints = center_keypoints(tmp_keypoints)
            tmp_keypoints = normalize_keypoints(tmp_keypoints)
            
            # Calculate extra features
            extra_features = calculate_extra_features(tmp_keypoints)
            
            # Flatten coordinates
            flat_coords = tmp_keypoints.flatten()
            
            # Concatenate
            full_vector = np.concatenate([flat_coords, extra_features], axis=0)

            all_samples.append(full_vector)
            all_labels.append(class_to_idx[class_name])
            total += 1
            class_processed += 1

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
    X = np.array(all_samples, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)

    logging.info(f"Valid samples (Base): {len(X)}")
    logging.info(f"Failed samples: {len(failed_files)}")

    # =================================================
    # Split, Mirror, Balance, Save
    # =================================================
    start = time.perf_counter()
    create_dataset_v3(X, y)
    save_time = time.perf_counter() - start

    total_time = time.perf_counter() - total_start

    logging.info("=" * 60)
    logging.info("Summary")
    logging.info(f"Total time: {total_time:.2f}s")
    logging.info(f"Save/Process dataset time: {save_time:.2f}s")
    logging.info("=" * 60)
