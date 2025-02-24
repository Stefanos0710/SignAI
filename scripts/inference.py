import os
import logging
import numpy as np
import tensorflow as tf
import pandas as pd
import json
from keras.models import load_model
from keras._tf_keras.keras.preprocessing.text import tokenizer_from_json  # Korrigiert

# Logging-Konfiguration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_json = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_json)
    return tokenizer

def load_preprocessed_keypoints(file_path):
    df = pd.read_csv(file_path)

    # Nur Spalten mit "_x" und "_y" auswählen
    keypoints_x = df.filter(like="_x").values
    keypoints_y = df.filter(like="_y").values

    return keypoints_x, keypoints_y

def process_keypoints(keypoints_x, keypoints_y):
    """Skaliert und kombiniert Keypoints (X und Y)."""
    min_x = np.min(keypoints_x)
    max_x = np.max(keypoints_x)
    min_y = np.min(keypoints_y)
    max_y = np.max(keypoints_y)

    keypoints_x = (np.array(keypoints_x) - min_x) / (max_x - min_x)
    keypoints_y = (np.array(keypoints_y) - min_y) / (max_y - min_y)

    keypoints = np.concatenate([keypoints_x, keypoints_y])
    return keypoints.reshape(1, -1)

def predict_gloss(model, tokenizer, keypoints):
    """Gibt die Vorhersage für das Gloss basierend auf den Keypoints zurück."""
    # Beispiel für die Vorbereitung der Eingaben:
    encoder_input_data = keypoints  # Dein vorbereiteter Encoder-Input

    # Setze den Startwert für den Decoder. Dies könnte ein Startsymbol sein, z.B. [<start>].
    start_token = np.array([[1]])  # Beispiel: ID für das Startsymbol könnte 1 sein
    decoder_input_data = start_token

    # Vorhersage
    prediction = model.predict([encoder_input_data, decoder_input_data])
    predicted_sequence = np.argmax(prediction, axis=-1)
    predicted_gloss = tokenizer.sequences_to_texts(predicted_sequence)
    return predicted_gloss[0]

def main():
    model_path = r"C:\Users\stefa\PycharmProjects\SignAI\models\trained_model_v2.h5"
    tokenizer_path = r"C:\Users\stefa\PycharmProjects\SignAI\tokenizers\gloss_tokenizer.json"
    keypoints_file = r"C:\Users\stefa\PycharmProjects\SignAI\data\live_data\live_keypoints.csv"

    if not os.path.exists(model_path):
        logging.error(f"Modelldatei nicht gefunden: {model_path}")
        return
    if not os.path.exists(tokenizer_path):
        logging.error(f"Tokenizer-Datei nicht gefunden: {tokenizer_path}")
        return
    if not os.path.exists(keypoints_file):
        logging.error(f"Keypoints-Datei nicht gefunden: {keypoints_file}")
        return

    model = load_model(model_path)
    logging.info("Modell erfolgreich geladen.")

    tokenizer = load_tokenizer(tokenizer_path)
    logging.info("Tokenizer erfolgreich geladen.")

    keypoints_x, keypoints_y = load_preprocessed_keypoints(keypoints_file)
    logging.info("Vorverarbeitete Keypoints erfolgreich geladen.")

    processed_keypoints = process_keypoints(keypoints_x, keypoints_y)
    gloss_prediction = predict_gloss(model, tokenizer, processed_keypoints)
    logging.info(f"Vorhergesagtes Gloss: {gloss_prediction}")

if __name__ == "__main__":
    main()
