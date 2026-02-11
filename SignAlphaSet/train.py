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

# paths to the preprocessed dataset
train_data_path = "SignAlphaSet/data/processed_dataset/train_data.npz"
val_data_path = "SignAlphaSet/data/processed_dataset/val_data.npz"
test_data_path = "SignAlphaSet/data/processed_dataset/test_data.npz"

def load_data(path_to_data):
    # load the dataset file
    data = np.load(path_to_data, allow_pickle=True)

    # read keypoints and labels
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

def build_model(
    input_shape,
    num_classes=26,
    cnn_filters=64,
    lstm_units=128,
    dense_units=64,
    dropout_rate=0.3
):
    # THESIS: With only 63 input values, a GNN is too complex; a small CNN + LSTM + softmax is enough for fast real-time classification.
    inputs = keras.Input(shape=input_shape, name="keypoints")
    x = keras.layers.Masking(mask_value=0.0, name="masking")(inputs)

    # cnn for local temporal patterns
    x = keras.layers.Conv1D(
        filters=cnn_filters,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="cnn_1"
    )(x)
    x = keras.layers.Dropout(dropout_rate, name="cnn_dropout")(x)

    # lstm reads the sequence and keeps the last state
    x = keras.layers.LSTM(
        lstm_units,
        return_sequences=False,
        dropout=dropout_rate,
        recurrent_dropout=0.1,
        name="lstm"
    )(x)

    # dense layers map to class probabilities
    x = keras.layers.Dense(dense_units, activation="relu", name="dense")(x)
    x = keras.layers.Dropout(dropout_rate, name="dense_dropout")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="class_probs")(x)

    model = keras.Model(inputs, outputs, name="cnn_lstm_classifier")
    return model

def train_model(
    model,
    train_keypoints,
    train_labels,
    val_keypoints,
    val_labels,
    test_keypoints=None,
    test_labels=None,
    batch_size=64,
    epochs=30,
    learning_rate=1e-3,
    version_model=1
):
    # optimizer and loss for multi-class classification
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy", dtype=tf.float32),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_accuracy", dtype=tf.float32)
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # callbacks for early stop, checkpoints, and logs
    os.makedirs("models", exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            min_delta=0.001
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"models/signalphaset_checkpoint_v{version_model}_epoch_{{epoch:02d}}.keras",
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=f"./logs/signalphaset_v{version_model}",
            histogram_freq=1,
            write_graph=True,
            update_freq="epoch"
        )
    ]

    history = model.fit(
        train_keypoints,
        train_labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(val_keypoints, val_labels),
        callbacks=callbacks,
        shuffle=True
    )

    model_save_path = f"models/signalphaset_v{version_model}.keras"
    model.save(model_save_path)
    logging.info(f"Model saved to: {model_save_path}")

    if test_keypoints is not None and test_labels is not None:
        results = model.evaluate(test_keypoints, test_labels, batch_size=batch_size, verbose=1)
        logging.info(f"Test results: {results}")

    return history

if __name__ == "__main__":
    # load data
    train_keypoints, train_labels = load_data(train_data_path)
    val_keypoints, val_labels = load_data(val_data_path)
    test_keypoints, test_labels = load_data(test_data_path)

    input_shape = train_keypoints.shape[1:]
    num_classes = int(np.max(train_labels)) + 1

    # Build and train model
    model = build_model(input_shape=input_shape, num_classes=num_classes)
    train_model(
        model,
        train_keypoints,
        train_labels,
        val_keypoints,
        val_labels,
        test_keypoints,
        test_labels
    )
