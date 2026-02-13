import os
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report

# =========================
# CONFIG
# =========================
MODEL_DIR = "SignAlphaSet/models"
TEST_DATA_PATH = "SignAlphaSet/data/processed_dataset/test_data.npz"

# =========================
# FIND LATEST MODEL
# =========================
model_files = glob.glob(os.path.join(MODEL_DIR, "*.keras"))

if not model_files:
    raise FileNotFoundError("No .keras model found in models folder.")

latest_model_path = max(model_files, key=os.path.getmtime)
print(f"Using latest model: {latest_model_path}")

# =========================
# CREATE ANALYSIS FOLDER
# =========================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
analysis_dir = os.path.join(MODEL_DIR, f"analysis_{timestamp}")
os.makedirs(analysis_dir, exist_ok=True)

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model(latest_model_path)

# =========================
# LOAD TEST DATA
# =========================
data = np.load(TEST_DATA_PATH)
X_test = data["X"]
y_test = data["y"]

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

    f.write(f"Model used: {latest_model_path}\n")
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
