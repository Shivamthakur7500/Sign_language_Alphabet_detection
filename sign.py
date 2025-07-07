import cv2
import mediapipe as mp
import numpy as np
import json

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

gesture_dict = {}

cap = cv2.VideoCapture(0)
label = "A"  # Change this for each sign you want to record

print(f"📸 Show gesture '{label}' and press 's' to save landmarks.")

while True:
    success, frame = cap.read()
    if not success:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract landmarks
            landmark_vector = []
            for lm in hand_landmarks.landmark:
                landmark_vector.extend([lm.x, lm.y, lm.z])

            cv2.putText(frame, "Press 's' to save this gesture", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Gesture Template Collector", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('s'):
        gesture_dict[label] = landmark_vector
        print(f"✅ Gesture '{label}' saved.")
    elif key & 0xFF == ord('d'):
        break

cap.release()
cv2.destroyAllWindows()

# Save dictionary to JSON
with open("sign_templates.json", "a") as f:
    json.dump(gesture_dict, f)

print("📁 Saved gesture template to sign_templates.json")
