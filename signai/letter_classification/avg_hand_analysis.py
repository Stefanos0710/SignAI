import argparse
from pathlib import Path
from typing import Iterable

import matplotlib
from sklearn.metrics import confusion_matrix
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorflow import keras


DEFAULT_SPLITS = ("train", "val", "test")
NUM_LANDMARKS = 21
COORD_DIMS = 3
BASE_KEYPOINT_FEATURES = NUM_LANDMARKS * COORD_DIMS


def parse_args():
	parser = argparse.ArgumentParser(description="Average hand analysis for a specific SignAlphaSet model version.")
	parser.add_argument(
		"--model-version",
		type=int,
		required=True,
		help="Model version number to analyze (e.g. 3 for signalphaset_v3.keras).",
	)
	return parser.parse_args()


def resolve_model_path(models_dir: Path, model_version: int) -> Path:
	candidates = [models_dir / f"signalphaset_v{model_version}.keras"]
	for candidate in candidates:
		if candidate.exists():
			return candidate

	available = sorted([item.name for item in models_dir.glob("*v*.keras")])
	raise FileNotFoundError(
		f"Could not find model for version v{model_version} in {models_dir}. "
		f"Available: {available}"
	)


def adapt_features_for_keypoint_pose(x: np.ndarray) -> np.ndarray:
	"""Ensure samples are in `(N,21,3)` using keypoint features only.

	Supports:
	- `(N,21,3)` directly
	- `(N,F)` with `F>=63` (e.g. 63 or 73), where only first 63 are keypoints
	"""
	x = np.asarray(x, dtype=np.float32)

	if x.ndim == 3 and x.shape[1:] == (NUM_LANDMARKS, COORD_DIMS):
		return x

	if x.ndim == 2 and x.shape[1] >= BASE_KEYPOINT_FEATURES:
		return x[:, :BASE_KEYPOINT_FEATURES].reshape(-1, NUM_LANDMARKS, COORD_DIMS)

	raise ValueError(
		f"Unexpected input shape {x.shape}. Expected (N,21,3) or (N,F) with F>=63."
	)


def adapt_features_for_model(x: np.ndarray, model) -> np.ndarray:
	"""Adapt feature tensor to the loaded model input shape.

	Handles both 3D models `(None,21,3)` and 2D models `(None,F)`.
	"""
	x = np.asarray(x, dtype=np.float32)
	model_input_shape = model.input_shape

	if isinstance(model_input_shape, list):
		model_input_shape = model_input_shape[0]

	if len(model_input_shape) == 3:
		expected_steps = model_input_shape[1]
		expected_dims = model_input_shape[2]
		if expected_steps != NUM_LANDMARKS or expected_dims != COORD_DIMS:
			raise ValueError(
				f"Unsupported 3D model input shape: {model_input_shape}. "
				f"Expected (None, {NUM_LANDMARKS}, {COORD_DIMS})."
			)
		return adapt_features_for_keypoint_pose(x)

	if len(model_input_shape) == 2:
		expected_features = model_input_shape[1]
		if expected_features is None:
			raise ValueError("Model expects dynamic feature size, cannot adapt safely.")

		if x.ndim == 3 and x.shape[1:] == (NUM_LANDMARKS, COORD_DIMS):
			x = x.reshape(x.shape[0], -1)
		elif x.ndim != 2:
			raise ValueError(
				f"Cannot adapt test data with shape {x.shape} for model input {model_input_shape}."
			)

		current_features = x.shape[1]
		if current_features == expected_features:
			return x
		if current_features > expected_features:
			return x[:, :expected_features]

		pad_width = expected_features - current_features
		return np.pad(x, ((0, 0), (0, pad_width)), mode="constant")

	raise ValueError(f"Unsupported model input rank for shape: {model_input_shape}")


def load_processed_dataset(processed_dir: Path, splits: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
	"""Load and merge preprocessed dataset splits from disk.

	The function reads each `<split>_data.npz` file from `processed_dir`, checks that
	the required keys (`X` and `y`) exist, and concatenates all split arrays into one
	feature tensor and one label vector.

	Returns:
		tuple[np.ndarray, np.ndarray]:
			- X: shape `(num_samples, 21, 3)` with hand landmarks as float32.
			- y: shape `(num_samples,)` with class indices as int64.
	"""
	# collect data from all configured splits
	x_parts = []
	y_parts = []

	for split in splits:
		split_path = processed_dir / f"{split}_data.npz"
		if not split_path.exists():
			raise FileNotFoundError(f"Split file not found: {split_path}")

		data = np.load(split_path, allow_pickle=True)
		if "X" not in data or "y" not in data:
			raise KeyError(f"File {split_path} must contain keys 'X' and 'y'.")

		x_parts.append(data["X"].astype(np.float32, copy=False))
		y_parts.append(data["y"].astype(np.int64, copy=False))

	x = np.concatenate(x_parts, axis=0)
	y = np.concatenate(y_parts, axis=0)
	# return merged landmarks and labels
	return x, y


def discover_class_labels(alpha_dataset_dir: Path, num_classes: int) -> list[str]:
	"""Discover class labels from dataset folders.

	If `alpha_dataset_dir` exists, folder names are sorted and used as class labels.
	If not enough folder names are available, the function falls back to numeric
	string labels (`"0"`, `"1"`, ...).

	Returns:
		list[str]: ordered list of labels with length `num_classes`.
	"""
	# use folder names when available
	if alpha_dataset_dir.exists() and alpha_dataset_dir.is_dir():
		candidates = sorted([item.name for item in alpha_dataset_dir.iterdir() if item.is_dir()])
		if len(candidates) >= num_classes:
			return candidates[:num_classes]
	# fallback to index-based labels
	return [str(index) for index in range(num_classes)]


def compute_average_poses(x: np.ndarray, y: np.ndarray, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
	"""Compute one average 3D hand pose per class.

	For each class index, samples are selected from `x` using the labels in `y`, then
	the mean landmark coordinates are computed across all samples of that class.

	Returns:
		tuple[np.ndarray, np.ndarray]:
			- avg_poses: shape `(num_classes, 21, 3)` with class-wise mean landmarks.
			- class_counts: shape `(num_classes,)` with sample count per class.
	"""
	# convert flattened feature vectors (63/73/...) to pure keypoint pose tensors
	x = adapt_features_for_keypoint_pose(x)

	# allocate output arrays for means and counts
	avg_poses = np.zeros((num_classes, 21, 3), dtype=np.float32)
	class_counts = np.zeros(num_classes, dtype=np.int64)

	for class_index in range(num_classes):
		class_samples = x[y == class_index]
		class_counts[class_index] = len(class_samples)
		if len(class_samples) > 0:
			avg_poses[class_index] = class_samples.mean(axis=0)

	# return class means and number of samples
	return avg_poses, class_counts


def compute_distance_matrix(avg_poses: np.ndarray) -> np.ndarray:
	"""Build a pairwise Euclidean distance matrix between class-average poses.

	Each class pose is flattened to a 1D vector, then distances are computed for all
	class pairs. Smaller values indicate more similar average hand configurations.

	Returns:
		np.ndarray: shape `(num_classes, num_classes)` as float32.
	"""
	# flatten each pose and compare all class pairs
	flattened = avg_poses.reshape(avg_poses.shape[0], -1)
	diffs = flattened[:, np.newaxis, :] - flattened[np.newaxis, :, :]
	matrix = np.sqrt(np.sum(diffs * diffs, axis=2))
	return matrix.astype(np.float32)


def plot_distance_heatmap(distance_matrix: np.ndarray, labels: list[str], output_path: Path) -> None:
	"""Render and save a heatmap of the class distance matrix.

	The function creates a labeled matrix plot where rows and columns correspond to
	class labels. Color intensity encodes Euclidean distance between class-average poses.

	Returns:
		None: writes the image file to `output_path`.
	"""
	# create the heatmap figure
	fig, ax = plt.subplots(figsize=(11, 9), dpi=140)
	image = ax.imshow(distance_matrix, cmap="viridis")
	ax.set_title("Similarity Matrix (Euclidean Distance between Average Poses)")
	ax.set_xticks(np.arange(len(labels)))
	ax.set_yticks(np.arange(len(labels)))
	ax.set_xticklabels(labels, rotation=90)
	ax.set_yticklabels(labels)
	fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Distance")
	fig.tight_layout()
	fig.savefig(output_path)
	plt.close(fig)


def write_average_positions_3d(avg_poses: np.ndarray, labels: list[str], output_path: Path) -> None:
	"""Write per-letter average 3D landmark values to a text file.

	For each class label, the function writes a small CSV-like block with one row per
	landmark index (`0..20`) and columns `x,y,z`.

	Returns:
		None: writes the formatted text output to `output_path`.
	"""
	# build output lines for the text file
	lines = ["Average hand positions in 3D per letter", "=" * 40, ""]
	for class_index, label in enumerate(labels):
		lines.append(f"[{label}]")
		lines.append("landmark,x,y,z")
		for landmark_index, landmark in enumerate(avg_poses[class_index]):
			lines.append(
				f"{landmark_index},{float(landmark[0]):.6f},{float(landmark[1]):.6f},{float(landmark[2]):.6f}"
			)
		lines.append("")

	output_path.write_text("\n".join(lines), encoding="utf-8")

def calc_error_rate_pairwise(model, x_test, y_test, class_labels):
	"""Calculate pairwise error rates between classes using the provided model and test data.

	Returns a matrix where entry (i, j) is the fraction of samples of class i predicted as class j.
	"""

	# adapt features to model input and get predictions
	x_test = adapt_features_for_model(x_test, model)
	y_pred = model.predict(x_test).argmax(axis=1)

	# confusion matrix
	conf_matrix = confusion_matrix(y_test, y_pred, labels=np.arange(len(class_labels)))

	# convert to pairwise error rates (row-normalized)
	error_rates = np.zeros_like(conf_matrix, dtype=np.float32)
	for i in range(len(class_labels)):
		row_sum = conf_matrix[i].sum()
		if row_sum > 0:
			error_rates[i] = conf_matrix[i] / row_sum  # normalized probabilities

	return error_rates

def plot_error_rate_heatmap(error_rates_matrix, class_labels, output_path):
	"""Plot and save a heatmap of pairwise error rates between classes."""
	fig, ax = plt.subplots(figsize=(11, 9), dpi=140)
	image = ax.imshow(error_rates_matrix, cmap="Reds")
	ax.set_title("Pairwise Error Rates (Confusion Matrix Normalized)")
	ax.set_xticks(np.arange(len(class_labels)))
	ax.set_yticks(np.arange(len(class_labels)))
	ax.set_xticklabels(class_labels, rotation=90)
	ax.set_yticklabels(class_labels)
	fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Error Rate")
	fig.tight_layout()
	fig.savefig(output_path)
	plt.close(fig)

def run_analysis(model_version: int, zero_error_diagonal: bool = True) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
	"""Run the analysis pipeline with distance/error comparison and correlation metrics.

	The function uses train+val data to compute average class poses and class distances,
	then evaluates the trained model on test data to build a normalized error-rate matrix.
	It normalizes distances to [0, 1], optionally zeroes the error-rate diagonal,
	computes Pearson correlation between both matrices (without diagonal), and prints
	the top 5 confusion pairs.

	Returns:
		tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
			- avg_poses: class-wise average landmarks from train+val.
			- class_counts: number of train+val samples per class.
			- class_labels: ordered labels used in outputs.
			- distance_matrix_norm: pairwise normalized class distances in [0, 1].
	"""

	# define input and output paths
	base_dir = Path(__file__).resolve().parent
	processed_dir = base_dir / "data" / "processed_dataset"
	alpha_dataset_dir = base_dir / "data" / "SignAlphaSet" / "SignAlphaSet"
	output_dir = base_dir / "logs" / "avg_hand_analysis"

	# create output folder immediately after definition
	output_dir.mkdir(parents=True, exist_ok=True)

	# 1) load train/val data only for average pose statistics
	x_all, y_all = load_processed_dataset(processed_dir=processed_dir, splits=["train", "val"])
	num_classes = int(np.max(y_all)) + 1
	class_labels = discover_class_labels(alpha_dataset_dir=alpha_dataset_dir, num_classes=num_classes)
	avg_poses, class_counts = compute_average_poses(x=x_all, y=y_all, num_classes=num_classes)

	# 2) compute raw distance matrix and normalize it to [0, 1]
	distance_matrix = compute_distance_matrix(avg_poses=avg_poses)
	dist_min = float(np.min(distance_matrix))
	dist_max = float(np.max(distance_matrix))
	if dist_max > dist_min:
		distance_matrix_norm = ((distance_matrix - dist_min) / (dist_max - dist_min)).astype(np.float32)
	else:
		distance_matrix_norm = np.zeros_like(distance_matrix, dtype=np.float32)

	# 3) load test data separately for confusion/error analysis
	x_test, y_test = load_processed_dataset(processed_dir=processed_dir, splits=["test"])
	model_path = resolve_model_path(base_dir / "models", model_version)
	print(f"Using specified model version v{model_version}: {model_path}")
	model = keras.models.load_model(model_path)

	# 4) calculate normalized error-rate matrix from test predictions
	error_rates_matrix = calc_error_rate_pairwise(model, x_test, y_test, class_labels)

	# optionally set diagonal to zero (ignore correct predictions in pairwise error view)
	if zero_error_diagonal:
		np.fill_diagonal(error_rates_matrix, 0.0)

	# 5) pearson correlation between normalized distance and error-rate matrices (without diagonal)
	off_diag_mask = ~np.eye(num_classes, dtype=bool)
	distance_vector = distance_matrix_norm[off_diag_mask]
	error_vector = error_rates_matrix[off_diag_mask]
	if np.std(distance_vector) > 0 and np.std(error_vector) > 0:
		pearson_corr = float(np.corrcoef(distance_vector, error_vector)[0, 1])
	else:
		pearson_corr = float("nan")

	# 6) extract top 5 confusion pairs from off-diagonal error rates
	confusion_pairs = []
	for i in range(num_classes):
		for j in range(num_classes):
			if i == j:
				continue
			confusion_pairs.append((i, j, float(error_rates_matrix[i, j])))
	confusion_pairs.sort(key=lambda item: item[2], reverse=True)
	top_confusions = confusion_pairs[:5]

	# 7) save visual and numeric outputs
	error_rates_path = output_dir / "error_rates_heatmap.png"
	plot_error_rate_heatmap(error_rates_matrix, class_labels, error_rates_path)

	heatmap_path = output_dir / "similarity_heatmap.png"
	plot_distance_heatmap(distance_matrix_norm, class_labels, heatmap_path)

	avg_positions_path = output_dir / "average_hand_positions_3d.txt"
	write_average_positions_3d(avg_poses, class_labels, avg_positions_path)

	np.save(output_dir / "average_poses.npy", avg_poses)
	np.save(output_dir / "distance_matrix.npy", distance_matrix)
	np.save(output_dir / "distance_matrix_norm.npy", distance_matrix_norm)
	np.save(output_dir / "error_rates_matrix.npy", error_rates_matrix)

	# write compact text report for correlation and top confusions
	report_lines = [
		"Distance vs Error Analysis",
		"=" * 30,
		f"pearson_correlation_off_diagonal={pearson_corr:.6f}",
		"",
		"Top 5 confusion pairs (true -> predicted):",
	]
	for i, j, rate in top_confusions:
		report_lines.append(
			f"- {class_labels[i]} -> {class_labels[j]}: error_rate={rate:.6f}, distance_norm={float(distance_matrix_norm[i, j]):.6f}"
		)
	(output_dir / "distance_error_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

	print("Analysis completed.")
	print(f"- Similarity heatmap (normalized distance): {heatmap_path}")
	print(f"- Error-rate heatmap: {error_rates_path}")
	print(f"- 3D average positions: {avg_positions_path}")
	print(f"- Pearson correlation (off-diagonal): {pearson_corr:.6f}")
	print("- Top 5 confusion pairs:")
	for i, j, rate in top_confusions:
		print(
			f"  {class_labels[i]} -> {class_labels[j]} | error_rate={rate:.6f} | distance_norm={float(distance_matrix_norm[i, j]):.6f}"
		)

	# return all computed core artifacts
	return avg_poses, class_counts, class_labels, distance_matrix_norm


if __name__ == "__main__":
	args = parse_args()
	run_analysis(model_version=args.model_version)
	