import pickle
import cv2
import mediapipe as mp
import numpy as np
import warnings
import os
import tensorflow as tf

warnings.simplefilter("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

# Load trained model
model = tf.keras.models.load_model('./sign_language_model.h5')

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Ensure labels_dict matches model training
labels_dict = {0: 'Hello!', 1: 'Stop!', 2: 'Good luck!', 3: 'Nice!'}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret or frame is None:
        print("❌ Error: Could not read frame from camera.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Normalize input
        data_aux = np.array(data_aux) / np.max(data_aux)
        input_data = data_aux.reshape(1, 21, 2, 1)

        # Make prediction
        prediction = model.predict(input_data)
        predicted_label = np.argmax(prediction)
        predicted_character = labels_dict.get(predicted_label, "Unknown")

        # Debugging output
        print("Raw prediction probabilities:", prediction)
        print("Predicted index:", predicted_label)
        print("Mapped label:", predicted_character)

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(
            frame,
            predicted_character,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )

    cv2.imshow("frame", frame)

    # Stop the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("🛑 Stopping video capture...")
        break

cap.release()
cv2.destroyAllWindows()
