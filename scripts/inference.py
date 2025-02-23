import os
import logging
import numpy as np
import json
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.text import tokenizer_from_json
from keras.models import load_model
from seq2seq_model import build_seq2seq_model  # Stelle sicher, dass die Modellarchitektur importiert wird

# Logging-Konfiguration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_json = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_json)
    return tokenizer


def load_preprocessed_keypoints(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    keypoints_x = data['keypoints_x']
    keypoints_y = data['keypoints_y']
    return keypoints_x, keypoints_y


def process_keypoints(keypoints_x, keypoints_y):
    """Skaliert und kombiniert Keypoints (X und Y)."""
    # Berechnung der Min- und Max-Werte für die Skalierung
    min_x = np.min(keypoints_x)
    max_x = np.max(keypoints_x)
    min_y = np.min(keypoints_y)
    max_y = np.max(keypoints_y)

    # Verschiebung und Skalierung der Keypoints (normalisiert auf den Bereich [0, 1])
    keypoints_x = (np.array(keypoints_x) - min_x) / (max_x - min_x)
    keypoints_y = (np.array(keypoints_y) - min_y) / (max_y - min_y)

    # Kombinieren der normalisierten x- und y-Koordinaten
    keypoints = np.concatenate([keypoints_x, keypoints_y])
    return keypoints.reshape(1, -1)  # Reshape, damit das Modell einen 2D-Array (Batch, Features) erhält


def predict_gloss(model, tokenizer, keypoints):
    """Gibt die Vorhersage für das Gloss basierend auf den Keypoints zurück."""
    # Modellvorhersage
    prediction = model.predict(keypoints)

    # Konvertierung der Vorhersage (Wahrscheinlichkeiten) in numerische Sequenzen
    predicted_sequence = np.argmax(prediction, axis=-1)

    # Umwandlung der Sequenz in Text (Gloss)
    predicted_gloss = tokenizer.sequences_to_texts(predicted_sequence)
    return predicted_gloss[0]


def main():
    model_path = r"C:\Users\stefa\PycharmProjects\SignAI\models\trained_model_v2.h5"
    tokenizer_path = r"C:\Users\stefa\PycharmProjects\SignAI\tokenizers\gloss_tokenizer.json"
    keypoints_file = r"C:\Users\stefa\PycharmProjects\SignAI\data\live_data\live_keypoints.csv"  # Beispielpfad zu deinen vorverarbeiteten Keypoints

    # Prüfe, ob alle Dateien vorhanden sind
    if not os.path.exists(model_path):
        logging.error(f"Modelldatei nicht gefunden: {model_path}")
        return

    if not os.path.exists(tokenizer_path):
        logging.error(f"Tokenizer-Datei nicht gefunden: {tokenizer_path}")
        return

    if not os.path.exists(keypoints_file):
        logging.error(f"Keypoints-Datei nicht gefunden: {keypoints_file}")
        return

    # Modell laden
    model = load_model(model_path)
    logging.info("Modell erfolgreich geladen.")

    # Tokenizer laden
    tokenizer = load_tokenizer(tokenizer_path)
    logging.info("Tokenizer erfolgreich geladen.")

    # Vorverarbeitete Keypoints laden
    keypoints_x, keypoints_y = load_preprocessed_keypoints(keypoints_file)
    logging.info("Vorverarbeitete Keypoints erfolgreich geladen.")

    # Keypoints für die Vorhersage vorbereiten (Skalierung und Kombination)
    processed_keypoints = process_keypoints(keypoints_x, keypoints_y)

    # Vorhersage treffen
    gloss_prediction = predict_gloss(model, tokenizer, processed_keypoints)
    logging.info(f"Vorhergesagtes Gloss: {gloss_prediction}")


if __name__ == "__main__":
    main()
