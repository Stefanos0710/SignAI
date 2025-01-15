import cv2
import mediapipe as mp
import csv
import os

# Mediapipe-Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,  # Maximal 2 Hände erkennen
                       min_detection_confidence=0.8,
                       min_tracking_confidence=0.8)

mp_draw = mp.solutions.drawing_utils  # Zum Zeichnen der Keypoints

# Sicherstellen, dass der Ausgabeordner existiert
output_folder = "\data\live_data"
os.makedirs(output_folder, exist_ok=True)

# CSV-Datei erstellen oder überschreiben
csv_file_path = os.path.join(output_folder, "live_keypoints.csv")
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Header erstellen
    header = ["Frame"]
    for hand in ["Hand_1", "Hand_2"]:
        header += [f"{hand}_Keypoint_{i}_x" for i in range(21)] + [f"{hand}_Keypoint_{i}_y" for i in range(21)]
    writer.writerow(header)

    # Live-Kamera starten
    cap = cv2.VideoCapture(0)  # Kamera öffnen (0 = Standardkamera)

    if not cap.isOpened():
        print("Kamera konnte nicht geöffnet werden!")
        exit()

    frame_counter = 0  # Zähler für Frames

    print("Drücke 'q', um die Live-Erfassung zu beenden.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Fehler beim Lesen des Kamerabildes!")
            break

        # Frame von BGR zu RGB konvertieren
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Hand-Keypoints erkennen
        result = hands.process(frame_rgb)

        # Daten für beide Hände initialisieren
        hand_1_keypoints = [""] * 42  # Platzhalter für 21 Keypoints (x, y) einer Hand
        hand_2_keypoints = [""] * 42

        # Überprüfen, ob Hände erkannt wurden
        if result.multi_hand_landmarks:
            for hand_index, hand_landmarks in enumerate(result.multi_hand_landmarks):
                # Hand-Koordinaten in Pixel umrechnen
                h, w, _ = frame.shape
                keypoints_data = []
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    keypoints_data.append(x)
                    keypoints_data.append(y)

                # Speichere die Keypoints für Hand 1 oder Hand 2
                if hand_index == 0:  # Erste erkannte Hand
                    hand_1_keypoints = keypoints_data
                elif hand_index == 1:  # Zweite erkannte Hand
                    hand_2_keypoints = keypoints_data

                # Zeichne die Hand-Keypoints
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Schreibe Frame-Nummer und Keypoints in die CSV
        writer.writerow([frame_counter] + hand_1_keypoints + hand_2_keypoints)

        # Frame anzeigen
        cv2.imshow("Live Hand Keypoints", frame)

        frame_counter += 1  # Zähler erhöhen

        # Programm mit 'q' beenden
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Ressourcen freigeben
cap.release()
cv2.destroyAllWindows()

print(f"Live-Keypoints-Erfassung abgeschlossen. Ergebnisse gespeichert in: {csv_file_path}")
