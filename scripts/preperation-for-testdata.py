import cv2
import mediapipe as mp
import csv
import os

# Initialisierung der Mediapipe-Module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.8,
                       min_tracking_confidence=0.8)

mp_draw = mp.solutions.drawing_utils  # Zum Zeichnen der Keypoints

# Ordner für Eingabe- und Ausgabe-Videos
input_videos_folder = "C:/Users/stefa/PycharmProjects/SignAI/data/raw_data"
output_videos_folder = "C:/Users/stefa/PycharmProjects/SignAI/data/train_data"

# Sicherstellen, dass der Ausgabeordner existiert
os.makedirs(output_videos_folder, exist_ok=True)

# Durch alle Ordner im Eingabeordner iterieren
for subfolder_name in os.listdir(input_videos_folder):
    subfolder_path = os.path.join(input_videos_folder, subfolder_name)

    # Überprüfen, ob es sich um einen Ordner handelt
    if not os.path.isdir(subfolder_path):
        print(f"Überspringe Datei: {subfolder_name} (kein Ordner)")
        continue

    # Suche die Videodatei und die Textdatei im Unterordner
    video_file = None
    gloss_file = None
    for file_name in os.listdir(subfolder_path):
        if file_name.lower().endswith('.mp4'):
            video_file = os.path.join(subfolder_path, file_name)
        elif file_name.lower().endswith('.txt'):
            gloss_file = os.path.join(subfolder_path, file_name)

    # Überprüfen, ob sowohl Video- als auch Textdatei vorhanden sind
    if not video_file or not gloss_file:
        print(f"Fehlende Dateien im Ordner: {subfolder_name}")
        continue

    # Gloss aus der Textdatei lesen
    with open(gloss_file, 'r', encoding='utf-8') as gloss_file_reader:
        gloss_text = gloss_file_reader.read().strip()

    # Erstelle eine CSV-Datei für das Video
    csv_file_path = os.path.join(output_videos_folder, f"{subfolder_name}_traindata.csv")
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Header: Video-Name, Gloss + Keypoints
        writer.writerow(["Video_Name", "Gloss"] + [f"Keypoint_{i}_x" for i in range(21)] + [f"Keypoint_{i}_y" for i in range(21)])

        # Video öffnen
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Video konnte nicht geöffnet werden: {video_file}")
            continue

        print(f"Verarbeite Video: {subfolder_name}")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print(f"Ende des Videos erreicht: {subfolder_name}")
                break

            # Frame von BGR zu RGB konvertieren
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Hand-Keypoints erkennen
            result = hands.process(frame_rgb)

            # Überprüfen, ob Hände erkannt wurden
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Keypoints speichern
                    h, w, _ = frame.shape
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)

                    keypoints_data = []
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        relative_x = x - wrist_x
                        relative_y = y - wrist_y
                        keypoints_data.append(relative_x)
                        keypoints_data.append(relative_y)

                    # Schreibe Videonamen, Gloss und Keypoints in die CSV
                    writer.writerow([subfolder_name, gloss_text] + keypoints_data)

            # Frame anzeigen (optional)
            cv2.imshow("Videoanalyse", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Ressourcen freigeben
        cap.release()

cv2.destroyAllWindows()
print("Analyse abgeschlossen. Ergebnisse in den jeweiligen CSV-Dateien gespeichert.")
