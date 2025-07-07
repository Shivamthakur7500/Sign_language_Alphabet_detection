import cv2
import mediapipe as mp
import numpy as np
import json

# Load gesture templates
with open("project\sign_templates.json", "r") as f:
    gesture_templates = json.load(f)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Prediction function
def predict_sign(live_landmarks, templates):
    min_dist = float('inf')
    predicted_label = None
    for label, template in templates.items():
        dist = np.linalg.norm(np.array(template) - np.array(live_landmarks))
        if dist < min_dist:
            min_dist = dist
            predicted_label = label
    return predicted_label

# Start webcam
cap = cv2.VideoCapture(0)
while True:
    success, frame = cap.read()
    if not success:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            live_vector = []
            for lm in hand_landmarks.landmark:
                live_vector.extend([lm.x, lm.y, lm.z])

            # Predict the sign
            predicted_label = predict_sign(live_vector, gesture_templates)

            # Display prediction
            cv2.putText(frame, f"Predicted: {predicted_label}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Real-Time Sign Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('d'):
        break

cap.release()
cv2.destroyAllWindows()
