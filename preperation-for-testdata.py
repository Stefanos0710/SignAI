import cv2
import mediapipe as mp
import csv
import os

# Initialisierung der Mediapipe-Module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.8,
                       min_tracking_confidence=0.8)

mp_draw = mp.solutions.drawing_utils  # Zum Zeichnen der Keypoints

# Ordner für Eingabe- und Ausgabe-Videos
input_videos_folder = "training-videos"
output_videos_folder = "training-csv-data"

# Sicherstellen, dass der Ausgabeordner existiert
os.makedirs(output_videos_folder, exist_ok=True)

# CSV-Datei zum Speichern der Ergebnisse öffnen
csv_file_path = os.path.join(output_videos_folder, "video_keypoints.csv")
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Header: Video-Name + Keypoints
    writer.writerow(["Video_Name"] + [f"Keypoint_{i}_x" for i in range(21)] + [f"Keypoint_{i}_y" for i in range(21)])

    # Durch alle Videos im Ordner iterieren
    for video_name in os.listdir(input_videos_folder):
        video_path = os.path.join(input_videos_folder, video_name)

        # Sicherstellen, dass es sich um eine Videodatei handelt
        if not video_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print(f"Überspringe Datei: {video_name} (kein Video)")
            continue

        # Video öffnen
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Video konnte nicht geöffnet werden: {video_name}")
            continue

        print(f"Verarbeite Video: {video_name}")

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print(f"Ende des Videos erreicht: {video_name}")
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

                    # Schreibe Videonamen und Keypoints in die CSV
                    writer.writerow([video_name] + keypoints_data)

            # Frame anzeigen (optional)
            cv2.imshow("Videoanalyse", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Ressourcen freigeben
        cap.release()

cv2.destroyAllWindows()
print(f"Analyse abgeschlossen. Ergebnisse gespeichert in: {csv_file_path}")
