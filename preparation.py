import cv2
import mediapipe as mp

# Initialisierung der Mediapipe-Module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.8,
                       min_tracking_confidence=0.8)

mp_draw = mp.solutions.drawing_utils  # Zum Zeichnen der Keypoints

# Kamera starten
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera konnte nicht geöffnet werden!")
    exit()

print("Drücke 'q', um das Programm zu beenden.")
while True:
    success, frame = cap.read()
    if not success:
        print("Fehler beim Lesen des Kamerabildes!")
        break

    # Bilder umwandeln von BGR in RGB (Mediapipe erwartet RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Hand-Keypoints erkennen
    result = hands.process(frame_rgb)

    # Überprüfen, ob Hände erkannt wurden
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Keypoints zeichnen
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Überprüfen, ob der Handgelenk-Punkt (Keypoint 0) existiert
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            h, w, _ = frame.shape
            wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
            print(f"Keypoint 0 (Handgelenk) Position: (x={wrist_x}, y={wrist_y})")

            # Alle Keypoints ausgeben (relativ zu Keypoint 0)
            for idx, landmark in enumerate(hand_landmarks.landmark):
                x, y = int(landmark.x * w), int(landmark.y * h)
                # Berechnung der relativen Koordinaten zum Handgelenk (Keypoint 0)
                relative_x = x - wrist_x
                relative_y = y - wrist_y

                print(f"Keypoint {idx}: (x={relative_x}, y={relative_y})")
                keypoints_pos = 0

                # Markiere Keypoints mit einem Kreis
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

            # Koordinatennetz zeichnen
            for i in range(0, w, 50):  # Netz mit 50 Pixel Abstand
                cv2.line(frame, (i, 0), (i, h), (0, 255, 0), 1)
            for j in range(0, h, 50):
                cv2.line(frame, (0, j), (w, j), (0, 255, 0), 1)

    # Bild anzeigen
    try:
        cv2.imshow("Handerkennung mit Keypoints", frame)
    except Exception as e:
        print(f"Fehler bei der Anzeige: {e}")
        break

    # Programm mit 'q' beenden
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Ressourcen freigeben
cap.release()
cv2.destroyAllWindows()
