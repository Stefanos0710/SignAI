import csv
import os
import logging
from collections import defaultdict
import numpy as np
import tensorflow as tf
from keras import preprocessing
from keras._tf_keras.keras.preprocessing.text import Tokenizer
#from tensorflow.keras.preprocesing.text import Tokenizer
from seq2seq_model import build_seq2seq_model


# Logging-Konfiguration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_csv_file(file_path):
    """
    Verarbeitet eine CSV-Datei mit Spalten für Video_Name, Gloss und Keypoints.
    """
    processed_data = []
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)  # Verwendet Spaltennamen aus der ersten Zeile
            for row_number, row in enumerate(reader, start=1):
                try:
                    video_name = row["Video_Name"]
                    gloss = row["Gloss"]
                    keypoints_x = [float(row[f"Keypoint_{i}_x"]) for i in range(21)]
                    keypoints_y = [float(row[f"Keypoint_{i}_y"]) for i in range(21)]

                    processed_data.append({
                        "video_name": video_name,
                        "gloss": gloss,
                        "keypoints_x": keypoints_x,
                        "keypoints_y": keypoints_y,
                    })
                except KeyError as e:
                    logging.error(f"Spalte fehlt in Datei {file_path}, Zeile {row_number}: {e}.")
                except ValueError as e:
                    logging.warning(f"Ungültige Werte in Datei {file_path}, Zeile {row_number}: {e}.")
    except FileNotFoundError:
        logging.error(f"Datei nicht gefunden: {file_path}")
    return processed_data


def prepare_data_for_training(processed_data, gloss_tokenizer):
    """
    Bereitet die extrahierten Daten für das Training vor, skaliert die Keypoints.
    """
    input_data = []
    target_data = []

    for entry in processed_data:
        gloss = entry["gloss"]
        keypoints_x = entry["keypoints_x"]
        keypoints_y = entry["keypoints_y"]

        # Berechnung der Min- und Max-Werte für die Skalierung
        min_x = np.min(keypoints_x)
        max_x = np.max(keypoints_x)
        min_y = np.min(keypoints_y)
        max_y = np.max(keypoints_y)

        # Verschiebung und Skalierung der Keypoints
        keypoints_x = (keypoints_x - min_x) / (max_x - min_x)  # Skaliert x-Werte auf [0, 1]
        keypoints_y = (keypoints_y - min_y) / (max_y - min_y)  # Skaliert y-Werte auf [0, 1]

        keypoints = keypoints_x + keypoints_y  # Kombinierte Keypoints (X und Y)

        input_data.append(keypoints)  # Eingabedaten (Keypoints)
        target_data.append(gloss)  # Ziel-Daten (Gloss)

    # Umwandlung von Ziel-Daten in numerische Werte
    target_data = gloss_tokenizer.texts_to_sequences(target_data)

    # Padding der Ziel-Daten auf gleiche Länge (falls erforderlich)
    max_target_length = max(len(seq) for seq in target_data)
    target_data = tf.keras.preprocessing.sequence.pad_sequences(target_data, maxlen=max_target_length, padding='post')

    # Optional: Padding der Eingabedaten
    max_input_length = max(len(seq) for seq in input_data)
    input_data = tf.keras.preprocessing.sequence.pad_sequences(input_data, maxlen=max_input_length, padding='post')

    return np.array(input_data), np.array(target_data)

def main():
    train_data_folder = r"C:\Users\stefa\PycharmProjects\SignAI\data\train_data"

    if not os.path.exists(train_data_folder):
        logging.error(f"Der Ordner {train_data_folder} existiert nicht.")
        return

    csv_files = [os.path.join(train_data_folder, f) for f in os.listdir(train_data_folder) if f.endswith('.csv')]

    if not csv_files:
        logging.warning(f"Keine CSV-Dateien im Ordner {train_data_folder} gefunden.")
        return

    all_data = []
    for file in csv_files:
        logging.info(f"Verarbeite Datei: {file}")
        processed_data = process_csv_file(file)
        if processed_data:
            logging.info(f"Datei {file} erfolgreich verarbeitet. Anzahl gültiger Zeilen: {len(processed_data)}")
            all_data.extend(processed_data)
        else:
            logging.warning(f"Keine gültigen Daten in Datei: {file}")

    # Tokenizer für Gloss erstellen
    gloss_tokenizer = Tokenizer()
    gloss_tokenizer.fit_on_texts([entry["gloss"] for entry in all_data])

    # Daten für das Modell vorbereiten
    logging.info(f"Bereite Trainingsdaten vor...")
    input_data, target_data = prepare_data_for_training(all_data, gloss_tokenizer)

    logging.info(f"Trainiere Modell...")
    input_vocab_size = len(gloss_tokenizer.word_index) + 1  # Die +1 ist für das Padding-Token
    target_vocab_size = len(gloss_tokenizer.word_index) + 1
    embedding_dim = 256
    hidden_dim = 512

    model = build_seq2seq_model(input_vocab_size, target_vocab_size, embedding_dim, hidden_dim)

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.fit([input_data, target_data], target_data, epochs=5 )

    logging.info(f"Training abgeschlossen!")

    # Speichern des Modells
    model_save_path = r"C:\Users\stefa\PycharmProjects\SignAI\models\trained_model_v2.h5"
    model.save(model_save_path)  # Modell speichern
    logging.info(f"Modell erfolgreich gespeichert unter: {model_save_path}")

if __name__ == "__main__":
    main()
