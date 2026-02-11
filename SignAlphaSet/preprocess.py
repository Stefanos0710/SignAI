import os
# os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np
import mediapipe as mp
import logging
import time

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
dataset_folder = "SignAlphaSet/data/SignAlphaSet/SignAlphaSet"

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

# ----------------------------
# Split dataset into train, val, test and save as npz files
# ----------------------------
def create_dataset(X, y, output_folder="SignAlphaSet/data/processed_dataset", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # create output folder if not exists
    os.makedirs(output_folder, exist_ok=True)

    # shuffle first
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    N = len(X)
    train_end = int(train_ratio * N)
    val_end = int((train_ratio + val_ratio) * N)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    # save as npz in output folder
    np.savez_compressed(os.path.join(output_folder, "train_data.npz"), X=X_train, y=y_train)
    np.savez_compressed(os.path.join(output_folder, "val_data.npz"), X=X_val, y=y_val)
    np.savez_compressed(os.path.join(output_folder, "test_data.npz"), X=X_test, y=y_test)

    logging.info(f"Datasets saved in folder: {output_folder}")
    logging.info(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")


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

            # append to dataset
            all_keypoints.append(tmp_keypoints)
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
            f"({class_pct:.2f}%) | failed {class_failed} | {class_elapsed:.2f}s\nd"
        )

    # =================================================
    # Convert to arrays
    # =================================================
    X = np.array(all_keypoints, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)

    logging.info(f"Valid samples: {len(X)}")
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
