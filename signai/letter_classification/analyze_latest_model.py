import os
import glob
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report

NUM_LANDMARKS = 21
COORD_DIMS = 3
BASE_KEYPOINT_FEATURES = NUM_LANDMARKS * COORD_DIMS

# =========================
# CONFIG
# =========================
MODEL_DIR = "signai/letter_classification/models"
TEST_DATA_PATH = "signai/letter_classification/data/processed_dataset/test_data.npz"


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze a specific SignAlphaSet model version.")
    parser.add_argument(
        "--model-version",
        type=int,
        required=True,
        help="Model version number to analyze (e.g. 3 for signalphaset_v3.keras).",
    )
    return parser.parse_args()


def resolve_model_path(model_dir: str, model_version: int) -> str:
    candidates = [os.path.join(model_dir, f"signalphaset_v{model_version}.keras")]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    available = sorted(glob.glob(os.path.join(model_dir, "*v*.keras")))
    available_names = [os.path.basename(path) for path in available]
    raise FileNotFoundError(
        f"Could not find model for version v{model_version} in {model_dir}. "
        f"Available: {available_names}"
    )

# =========================
# SELECT MODEL VERSION
# =========================
args = parse_args()
selected_model_path = resolve_model_path(MODEL_DIR, args.model_version)
print(f"Using specified model version v{args.model_version}: {selected_model_path}")

# =========================
# CREATE ANALYSIS FOLDER
# =========================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
analysis_dir = os.path.join(MODEL_DIR, f"analysis_{timestamp}")
os.makedirs(analysis_dir, exist_ok=True)

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model(selected_model_path)


def adapt_features_for_model(x_data, keras_model):
    x_data = np.asarray(x_data, dtype=np.float32)
    model_input_shape = keras_model.input_shape

    if isinstance(model_input_shape, list):
        model_input_shape = model_input_shape[0]

    if not isinstance(model_input_shape, tuple):
        raise ValueError(f"Unsupported model input shape format: {model_input_shape}")

    if len(model_input_shape) == 3:
        expected_steps = model_input_shape[1]
        expected_dims = model_input_shape[2]

        if expected_steps != NUM_LANDMARKS or expected_dims != COORD_DIMS:
            raise ValueError(
                f"Unsupported 3D model input shape: {model_input_shape}. "
                f"Expected (None, {NUM_LANDMARKS}, {COORD_DIMS})."
            )

        if x_data.ndim == 3 and x_data.shape[1:] == (NUM_LANDMARKS, COORD_DIMS):
            print(f"Using 3D test data as-is: {x_data.shape}")
            return x_data

        if x_data.ndim == 2 and x_data.shape[1] >= BASE_KEYPOINT_FEATURES:
            extra_feature_count = x_data.shape[1] - BASE_KEYPOINT_FEATURES
            adapted = x_data[:, :BASE_KEYPOINT_FEATURES].reshape(-1, NUM_LANDMARKS, COORD_DIMS)
            print(
                f"Detected flattened test data with {x_data.shape[1]} features "
                f"({BASE_KEYPOINT_FEATURES} keypoint + {extra_feature_count} extra). "
                f"Using keypoint part for model input: {adapted.shape}"
            )
            return adapted

        raise ValueError(
            f"Cannot adapt test data with shape {x_data.shape} for model input {model_input_shape}."
        )

    if len(model_input_shape) == 2:
        expected_features = model_input_shape[1]
        if expected_features is None:
            raise ValueError("Model expects dynamic feature size, cannot adapt safely.")

        if x_data.ndim == 3 and x_data.shape[1:] == (NUM_LANDMARKS, COORD_DIMS):
            x_data = x_data.reshape(x_data.shape[0], -1)
            print(f"Flattened 3D test data to 2D features: {x_data.shape}")
        elif x_data.ndim != 2:
            raise ValueError(
                f"Cannot adapt test data with shape {x_data.shape} for model input {model_input_shape}."
            )

        current_features = x_data.shape[1]
        if current_features == expected_features:
            print(f"Using 2D test data as-is: {x_data.shape}")
            return x_data

        if current_features > expected_features:
            adapted = x_data[:, :expected_features]
            print(
                f"Trimmed test features from {current_features} to {expected_features}: {adapted.shape}"
            )
            return adapted

        pad_width = expected_features - current_features
        adapted = np.pad(x_data, ((0, 0), (0, pad_width)), mode="constant")
        print(
            f"Padded test features from {current_features} to {expected_features}: {adapted.shape}"
        )
        return adapted

    raise ValueError(f"Unsupported model input rank for shape: {model_input_shape}")

# =========================
# LOAD TEST DATA
# =========================
data = np.load(TEST_DATA_PATH)
X_test = data["X"]
y_test = data["y"]
X_test = adapt_features_for_model(X_test, model)

# =========================
# PREDICT
# =========================
y_probs = model.predict(X_test, verbose=1)
y_pred = np.argmax(y_probs, axis=1)

# =========================
# BASIC METRICS
# =========================
loss, acc, top5 = model.evaluate(X_test, y_test, verbose=0)

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(analysis_dir, "confusion_matrix.png"), dpi=200)
plt.close()

# =========================
# CONFIDENCE HISTOGRAM
# =========================
max_conf = np.max(y_probs, axis=1)

plt.figure(figsize=(8, 5))
plt.hist(max_conf, bins=30)
plt.title("Prediction Confidence Distribution")
plt.xlabel("Confidence")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(analysis_dir, "confidence_distribution.png"), dpi=200)
plt.close()

# =========================
# CLASS REPORT
# =========================
report = classification_report(y_test, y_pred, output_dict=True)

per_class_accuracy = {}
for cls in report:
    if cls.isdigit():
        per_class_accuracy[int(cls)] = report[cls]["recall"]

sorted_classes = sorted(per_class_accuracy.items(), key=lambda x: x[1])

worst_5 = sorted_classes[:5]
best_5 = sorted_classes[-5:]

perfect_classes = [
    cls for cls, val in per_class_accuracy.items()
    if np.isclose(val, 1.0)
]

# =========================
# MISCLASSIFICATIONS
# =========================
misclassified = y_test != y_pred
mis_pairs = list(zip(y_test[misclassified], y_pred[misclassified]))

confusion_counts = {}

for true_label, pred_label in mis_pairs:
    key = (int(true_label), int(pred_label))
    confusion_counts[key] = confusion_counts.get(key, 0) + 1

sorted_confusions = sorted(
    confusion_counts.items(),
    key=lambda x: x[1],
    reverse=True
)

# =========================
# SAVE SUMMARY TXT
# =========================
summary_path = os.path.join(analysis_dir, "model_summary.txt")

with open(summary_path, "w") as f:
    f.write("MODEL ANALYSIS SUMMARY\n")
    f.write("="*40 + "\n\n")

    f.write(f"Model used: {selected_model_path}\n")
    f.write(f"Test samples: {len(X_test)}\n\n")

    f.write("BASIC METRICS\n")
    f.write(f"Loss: {loss:.4f}\n")
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"Top-5 Accuracy: {top5:.4f}\n\n")

    f.write("WORST 5 CLASSES (Recall)\n")
    for cls, val in worst_5:
        f.write(f"Class {cls}: {val:.4f}\n")
    f.write("\n")

    f.write("BEST 5 CLASSES (Recall)\n")
    for cls, val in best_5:
        f.write(f"Class {cls}: {val:.4f}\n")
    f.write("\n")

    f.write("PERFECT CLASSES (100%)\n")
    f.write(str(perfect_classes) + "\n\n")

    f.write("MOST COMMON CONFUSIONS\n")
    for (true_label, pred_label), count in sorted_confusions[:10]:
        f.write(f"True {true_label} → Pred {pred_label} | {count} times\n")
    f.write("\n")

    f.write("MODEL ARCHITECTURE\n")
    model.summary(print_fn=lambda x: f.write(x + "\n"))

print(f"\nAnalysis saved to: {analysis_dir}")
