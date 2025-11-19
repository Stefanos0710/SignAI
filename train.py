import os
import csv
import json
import logging
import numpy as np
import tensorflow as tf
import pandas as pd
from typing import List, Dict, Union, Tuple
import time
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Reshape
from tensorflow.keras.regularizers import l1
# Logging-Konfiguration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# very goooood dataset: https://arxiv.org/pdf/2210.06791v1

########################################################################
# 1. Data Loading and Preprocessing
########################################################################

def load_data_from_folder(folder_path: str) -> List[Dict[str, Union[np.ndarray, str]]]:
    """
    Lädt alle CSV-Dateien aus einem Ordner und extrahiert Keypoints und Gloss-Texte.
    Verarbeitet jetzt alle Frames pro Geste als eine Einheit.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    all_samples = []
    start_time = time.time()

    # Finde alle CSV-Dateien
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    logging.info(f"Found {len(csv_files)} CSV files in {folder_path}")

    for file_num, file_name in enumerate(csv_files, 1):
        csv_file_path = os.path.join(folder_path, file_name)
        try:
            # Versuche verschiedene Encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'iso-8859-1']
            df = None

            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_file_path, encoding=encoding)
                    logging.info(f"Successfully loaded {file_name} with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue

            if df is None:
                logging.error(f"Could not read {file_name} with any encoding")
                continue

            # Debug-Information
            logging.info(f"Processing file {file_num}/{len(csv_files)}: {file_name}")
            logging.info(f"Columns found: {df.columns.tolist()}")
            logging.info(f"Number of rows: {len(df)}")

            # Validiere Spaltenstruktur
            required_prefixes = ['pose_', 'hand_', 'face_']
            if not all(any(col.startswith(prefix) for col in df.columns) for prefix in required_prefixes):
                logging.error(f"Missing required coordinate prefixes in {file_name}")
                continue

            # Sammle alle Frames für diese Geste
            all_frames = []
            gloss_text = ""

            for idx, row in df.iterrows():
                try:
                    # Keypoints extrahieren
                    keypoints = []

                    # Pose Keypoints (33 Punkte)
                    for i in range(33):
                        keypoints.extend([float(row[f"pose_{i}_x"]), float(row[f"pose_{i}_y"])])

                    # Hand Keypoints (42 Punkte)
                    for i in range(42):
                        keypoints.extend([float(row[f"hand_{i}_x"]), float(row[f"hand_{i}_y"])])

                    # Face Keypoints (468 Punkte)
                    for i in range(468):
                        keypoints.extend([float(row[f"face_{i}_x"]), float(row[f"face_{i}_y"])])

                    # Überprüfe auf ungültige Werte
                    keypoints = np.array(keypoints, dtype=np.float32)
                    if np.any(np.isnan(keypoints)) or np.any(np.isinf(keypoints)):
                        logging.warning(f"Invalid values detected in {file_name}, row {idx + 1}")
                        continue

                    all_frames.append(keypoints)

                    # Gloss-Text speichern (sollte für alle Frames einer Geste gleich sein)
                    if idx == 0:  # Nur beim ersten Frame
                        if 'Gloss' in row:
                            gloss_text = str(row['Gloss']).strip()
                        elif 'gloss' in row:
                            gloss_text = str(row['gloss']).strip()

                except Exception as e:
                    logging.error(f"Error processing row {idx + 1} in {file_name}: {str(e)}")
                    continue

            if all_frames and gloss_text:
                # Berechne Mittelwert über alle Frames
                average_keypoints = np.mean(all_frames, axis=0)

                # Normalisierung
                min_val = average_keypoints.min()
                max_val = average_keypoints.max()
                range_val = max_val - min_val if max_val > min_val else 1.0
                normalized_keypoints = (average_keypoints - min_val) / range_val

                # Füge Sample hinzu
                all_samples.append({
                    "keypoints": normalized_keypoints,
                    "gloss": gloss_text,
                    "source_file": file_name
                })

        except Exception as e:
            logging.error(f"Error reading file {csv_file_path}: {str(e)}")
            continue

    if not all_samples:
        raise ValueError("No valid samples found in any CSV file")

    logging.info(f"Total valid samples processed: {len(all_samples)}")
    return all_samples

def build_encoder_input(all_samples):
    """
    Konvertiert die Keypoints in ein numpy array für den Encoder Input
    """
    all_keypoints = [s["keypoints"] for s in all_samples]
    all_keypoints = np.stack(all_keypoints, axis=0)  # (num_samples, feature_dim)
    all_keypoints = np.expand_dims(all_keypoints, axis=1)  # (num_samples, 1, feature_dim)
    return all_keypoints


def build_tokenizer(all_samples, extra_tokens=("<start>", "<end>")):
    """
    Erstellt und trainiert den Tokenizer auf den Gloss-Texten
    """
    tokenizer = Tokenizer(oov_token="<unk>")

    # Extrahiere Gloss-Texte
    gloss_texts = [s["gloss"] for s in all_samples]

    # Debug: Beispiele vor Tokenisierung
    logging.info("Sample texts before tokenization:")
    for text in gloss_texts[:5]:
        logging.info(f"  {text}")

    # Tokenizer auf Texten trainieren
    tokenizer.fit_on_texts(gloss_texts)

    # Debug: Vokabular-Statistiken
    logging.info(f"Vocabulary size before special tokens: {len(tokenizer.word_index)}")

    # Häufigste Wörter anzeigen
    sorted_vocab = sorted(tokenizer.word_counts.items(), key=lambda x: x[1], reverse=True)
    logging.info("Most common words:")
    for word, count in sorted_vocab[:10]:
        logging.info(f"  {word}: {count}")

    # Spezielle Tokens hinzufügen
    for tkn in extra_tokens:
        if tkn not in tokenizer.word_index:
            tokenizer.word_index[tkn] = len(tokenizer.word_index) + 1

    logging.info(f"Final vocabulary size: {len(tokenizer.word_index)}")
    return tokenizer


def build_decoder_data(all_samples, tokenizer):
    """
    Erstellt Decoder Input und Target Sequenzen aus den Gloss-Texten
    """
    start_token = "<start>"
    end_token = "<end>"

    # Füge Start/End Tokens hinzu
    gloss_texts = [s["gloss"] for s in all_samples]
    gloss_texts_with_tokens = [f"{start_token} {g} {end_token}" for g in gloss_texts]

    # Konvertiere zu Token-Sequenzen
    sequences = tokenizer.texts_to_sequences(gloss_texts_with_tokens)

    # Erstelle Input/Target Paare
    decoder_input_sequences = []
    decoder_target_sequences = []

    for seq in sequences:
        decoder_input_sequences.append(seq[:-1])  # alles außer letztem Token
        decoder_target_sequences.append(seq[1:])  # alles außer erstem Token

    # Padding auf gleiche Länge
    max_len = max(len(s) for s in sequences) - 1
    decoder_input_data = pad_sequences(decoder_input_sequences, maxlen=max_len, padding='post')
    decoder_target_data = pad_sequences(decoder_target_sequences, maxlen=max_len, padding='post')

    logging.info(f"Decoder sequences padded to length: {max_len}")
    return decoder_input_data, decoder_target_data


def build_seq2seq_model(
        input_sequence_length,
        input_feature_dim,
        target_vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        dropout_rate=0.2,
        l1_reg=0.0005
):
    # Encoder
    encoder_inputs = Input(shape=(input_sequence_length, input_feature_dim), name="encoder_inputs")

    # Normalization
    normalized_inputs = tf.keras.layers.LayerNormalization(name="encoder_norm")(encoder_inputs)

    # Reduzierte Anzahl von Hidden Layers
    hidden_layers = normalized_inputs
    for i in range(2):  # Reduziert von 3 auf 2 Layer
        hidden_layers = Dense(
            256,  # Reduziert von 1024
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=0.0005),
            name=f"encoder_dense_{i + 1}"
        )(hidden_layers)
        hidden_layers = tf.keras.layers.BatchNormalization(name=f"batch_norm_{i + 1}")(hidden_layers)
        hidden_layers = Dropout(dropout_rate, name=f"dropout_{i + 1}")(hidden_layers)

    # Einzelner Bidirektionaler Encoder statt zwei
    encoder_lstm = tf.keras.layers.Bidirectional(
        LSTM(
            hidden_dim,
            return_sequences=True,
            return_state=True,
            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=0.0005),
            recurrent_regularizer=tf.keras.regularizers.l2(0.0005),
            name="encoder_lstm"
        )
    )(hidden_layers)

    # States für den Decoder
    encoder_outputs = encoder_lstm[0]
    state_h = tf.keras.layers.Concatenate()([encoder_lstm[1], encoder_lstm[3]])
    state_c = tf.keras.layers.Concatenate()([encoder_lstm[2], encoder_lstm[4]])
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(None,), name="decoder_inputs")

    decoder_embedding = tf.keras.layers.Embedding(
        target_vocab_size,
        embedding_dim,
        embeddings_regularizer=tf.keras.regularizers.l2(0.0005),
        name="decoder_embedding"
    )(decoder_inputs)

    decoder_lstm = LSTM(
        hidden_dim * 2,
        return_sequences=True,
        return_state=True,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=0.0005),
        recurrent_regularizer=tf.keras.regularizers.l2(0.0005),
        name="decoder_lstm"
    )

    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

    # Vereinfachte Attention
    attention = tf.keras.layers.MultiHeadAttention(
        num_heads=4,  # Reduziert von 8
        key_dim=hidden_dim,
        name="multi_head_attention"
    )(decoder_outputs, encoder_outputs)

    attention_output = tf.keras.layers.LayerNormalization()(decoder_outputs + attention)

    # Reduzierte Dense Layers
    dense = Dense(
        hidden_dim,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=0.0005),
        name="decoder_dense"
    )(attention_output)

    outputs = Dense(
        target_vocab_size,
        activation="softmax",
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_reg, l2=0.0005),
        name="decoder_output"
    )(dense)

    model = Model([encoder_inputs, decoder_inputs], outputs, name="seq2seq_model")
    return model

# Die train_main Funktion muss auch angepasst werden:
def train_main(
        train_data_folder,
        version_model=1,
        epochs=200,
        batch_size=16,
        validation_split=0.2,
        input_sequence_length=1,
        embedding_dim=512,
        hidden_dim=1024,
        dropout_rate=0.3,
        l1_reg=0.001
):
    try:
        # 1. Daten laden
        samples = load_data_from_folder(train_data_folder)

        # 2. Tokenizer erstellen
        tokenizer = build_tokenizer(samples)

        # 3. Encoder Input vorbereiten
        encoder_input_data = build_encoder_input(samples)
        input_feature_dim = encoder_input_data.shape[-1]

        # 4. Decoder Daten vorbereiten
        decoder_input_data, decoder_target_data = build_decoder_data(samples, tokenizer)

        # Log Daten-Shapes
        logging.info(f"Encoder input shape: {encoder_input_data.shape}")
        logging.info(f"Decoder input shape: {decoder_input_data.shape}")
        logging.info(f"Decoder target shape: {decoder_target_data.shape}")

        # 5. Modell erstellen
        target_vocab_size = len(tokenizer.word_index) + 1
        model = build_seq2seq_model(
            input_sequence_length=input_sequence_length,
            input_feature_dim=input_feature_dim,
            target_vocab_size=target_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            l1_reg=l1_reg
        )

        # Learning Rate Schedule
        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate,
            first_decay_steps=1000,
            t_mul=2.0,
            m_mul=0.9,
            alpha=0.0001
        )

        # Optimizer
        # Ersetzen Sie den bisherigen Optimizer-Code mit diesem:
        initial_learning_rate = 0.001
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=initial_learning_rate,  # Fester Wert statt Schedule
            weight_decay=0.001,
            clipnorm=1.0,
            epsilon=1e-7,
            beta_1=0.9,
            beta_2=0.999
        )
        # Modell kompilieren
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy', tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
        )

        # Modell-Summary ausgeben
        model.summary()

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                min_delta=0.001
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'models/checkpoint_v{version_model}_' + 'epoch_{epoch:02d}.keras',
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=f'./logs/model_v{version_model}',
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        ]

        # Training
        history = model.fit(
            [encoder_input_data, decoder_input_data],
            decoder_target_data,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            shuffle=True
        )

        # Modell und Tokenizer speichern
        model_save_path = f"models/trained_model_v{version_model}.keras"
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        model.save(model_save_path)
        logging.info(f"Model saved to: {model_save_path}")

        tokenizer_path = "tokenizers/gloss_tokenizer.json"
        os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
        with open(tokenizer_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer.to_json(), f, ensure_ascii=False)
        logging.info(f"Tokenizer saved to: {tokenizer_path}")

        return history

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        # Verzeichnisstruktur erstellen
        os.makedirs("data/train_data", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("tokenizers", exist_ok=True)

        # Konfiguration
        config = {
            "train_data_folder": "data/train_data",
            "version_model": 21,
            "epochs": 500,
            "batch_size": 8,
            "validation_split": 0.2,
            "input_sequence_length": 1,
            "embedding_dim": 128,
            "hidden_dim": 256,
            "dropout_rate": 0.2,
            "l1_reg": 0.0005
        }

        # Training starten
        history = train_main(**config)

        # Training-Verlauf visualisieren
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))

        # Accuracy Plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Loss Plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'models/training_history_v{config["version_model"]}.png')
        plt.close()

        logging.info("Training completed successfully")

    except Exception as e:
        logging.error(f"Program terminated with error: {str(e)}")
        raise
