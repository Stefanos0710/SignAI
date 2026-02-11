import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging
import os

# set random seeds for reproducibility 
import keras
keras.utils.set_random_seed(42)

# logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# path for the preprocessed dataset
train_data_path = "SignAlphaSet/data/processed_dataset/train_data.npz"
val_data_path = "SignAlphaSet/data/processed_dataset/val_data.npz"
test_data_path = "SignAlphaSet/data/processed_dataset/test_data.npz"

def load_data(path_to_data):
    # get the data from the preprocessed .npz files
    data = np.load(path_to_data, allow_pickle=True)

    # get the X (keypoints) and y (labels) from the data
    if 'X' in data and 'y' in data:
        keypoints = data['X']
        labels = data['y']
        print("yeah, loaded data with keys 'X' and 'y'")
    else:
        logging.error(f"Data file {path_to_data} does not contain 'X' and 'y' keys.")
        raise KeyError(f"Data file {path_to_data} does not contain 'X' and 'y' keys.")

    # log the number of samples loaded
    logging.info(f"Loaded data from {path_to_data} with {len(keypoints)} samples.")
    return keypoints, labels

def preprocess_data():
    pass    

def build_model():
    # THESIS: With only 63 input values, a GNN is too complex; a small CNN + LSTM + softmax is enough for fast real-time classification.

    # Single LSTM + softmax: sequence in → CNN/GNN → LSTM (forward only) → take last hidden state (or a pooling over time) → dense layer → softmax for class probabilities (n=26)
    pass

def train_model():
    pass

if __name__ == "__main__":
    # Load data
    train_keypoints, train_labels = load_data(train_data_path)
    val_keypoints, val_labels = load_data(val_data_path)
    test_keypoints, test_labels = load_data(test_data_path)

    # Build and train model
    build_model()
    train_model()
