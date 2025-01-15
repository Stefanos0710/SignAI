import numpy as np
import os
import csv


def load_csv_data(folder_path):
    input_sequences = []
    target_sequences = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                reader = csv.reader(file)
                next(reader)  # Überspringe den Header

                for row in reader:
                    try:
                        # Erste Spalte: Eingabedaten (Keypoints)
                        input_sequence = [float(value) for value in row[2:44]]  # x, y-Koordinaten für 21 Keypoints
                        # Zweite Spalte: Ziel (Gloss-ID oder Ähnliches)
                        target_value = int(row[1])  # Beispiel für Gloss-ID

                        input_sequences.append(input_sequence)
                        target_sequences.append(target_value)
                    except ValueError as e:
                        print(f"Fehler beim Verarbeiten einer Zeile in {file_name}: {e}")

    return np.array(input_sequences, dtype=np.float32), np.array(target_sequences, dtype=np.int32)


# Ordnerpfad zu den CSV-Dateien
csv_folder = "C:/Users/stefa/PycharmProjects/SignAI/data/train_data"

# Daten laden
try:
    input_data, target_data = load_csv_data(csv_folder)
    print(f"Daten erfolgreich geladen: {input_data.shape}, {target_data.shape}")
except Exception as e:
    print(f"Fehler beim Laden der Daten: {e}")
