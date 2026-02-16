from pathlib import Path
from typing import Iterable

import matplotlib
from sklearn.metrics import confusion_matrix
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_SPLITS = ("train", "val", "test")


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

    # get model predictions
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

def run_analysis() -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
	"""Run the full average-hand analysis pipeline.

	This function loads data, computes class-wise average poses, computes the distance
	matrix, saves a similarity heatmap, writes average 3D landmark values per letter,
	and stores raw numpy outputs.

	Returns:
		tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
			- avg_poses: class-wise average landmarks.
			- class_counts: number of samples per class.
			- class_labels: ordered labels used in outputs.
			- distance_matrix: pairwise class distances.
	"""
	# define input and output paths
	base_dir = Path(__file__).resolve().parent
	processed_dir = base_dir / "data" / "processed_dataset"
	alpha_dataset_dir = base_dir / "data" / "SignAlphaSet" / "SignAlphaSet"
	output_dir = base_dir / "logs" / "avg_hand_analysis"

	# load data and compute statistics
	x, y = load_processed_dataset(processed_dir=processed_dir, splits=DEFAULT_SPLITS)
	num_classes = int(np.max(y)) + 1
	class_labels = discover_class_labels(alpha_dataset_dir=alpha_dataset_dir, num_classes=num_classes)
	avg_poses, class_counts = compute_average_poses(x=x, y=y, num_classes=num_classes)
	distance_matrix = compute_distance_matrix(avg_poses=avg_poses)

	# load test data and model
	x, y = load_processed_dataset(processed_dir=processed_dir, splits=["test"])
	model = keras.models.load_model(base_dir / "models" / "signalphaset_v2.keras") # SignAlphaSet/models/signalphaset_v2.keras

	# here comes the compare error rate between every sign

	# then all output will be saved

	# gen and save heatmap 

	# create compared heatmap of similarity between average poses


	# ensure output directory exists
	output_dir.mkdir(parents=True, exist_ok=True)

	# save heatmap image
	heatmap_path = output_dir / "similarity_heatmap.png"
	plot_distance_heatmap(distance_matrix, class_labels, heatmap_path)

	# save average 3d positions per class
	avg_positions_path = output_dir / "average_hand_positions_3d.txt"
	write_average_positions_3d(avg_poses, class_labels, avg_positions_path)

	# save raw numpy outputs for later use
	np.save(output_dir / "average_poses.npy", avg_poses)
	np.save(output_dir / "distance_matrix.npy", distance_matrix)

	print("Analysis completed.")
	print(f"- Heatmap: {heatmap_path}")
	print(f"- 3D average positions: {avg_positions_path}")

	# return all computed core artifacts
	return avg_poses, class_counts, class_labels, distance_matrix


if __name__ == "__main__":
	run_analysis()
	