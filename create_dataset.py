import os
import pickle
import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils  # To draw landmarks

# Dataset directory
DATA_DIR = './isl_dataset'
EXPECTED_FEATURES = 42 * 2  # 21 landmarks * 2 (x, y) for **both** hands

data = []
labels = []
folder_count = 0  # Count folders processed

# Process dataset
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):  # Skip files
        continue

    print(f"📂 Processing folder: {dir_}...")  # ✅ Folder processing message
    folder_data_count = 0  # Track processed images per folder

    for img_path in os.listdir(dir_path):
        img = cv2.imread(os.path.join(dir_path, img_path))

        if img is None:
            print(f"⚠️ Failed to load: {img_path} in {dir_}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            print(f"⚠️ No hands detected in: {img_path}")
            continue

        data_aux = []
        x_ = [] 
        y_ = []

        # Loop through each detected hand
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

        # If only **one hand** detected, pad the remaining 42 features with 0
        if len(data_aux) < EXPECTED_FEATURES:
            data_aux.extend([0] * (EXPECTED_FEATURES - len(data_aux)))

        # If more than **two hands** detected (unexpected case), truncate to 84 features
        elif len(data_aux) > EXPECTED_FEATURES:
            data_aux = data_aux[:EXPECTED_FEATURES]

        data.append(data_aux)
        labels.append(dir_)  # Use folder name as label
        folder_data_count += 1

    print(f"✅ Completed folder: {dir_} ({folder_data_count} images processed)\n")  # ✅ Completion message

    folder_count += 1  # Increment folder count

    # Save data every 5 folders
    if folder_count % 5 == 0:
        with open('data.pickle', 'wb') as f:
            pickle.dump({'data': data, 'labels': labels}, f)
        print(f"💾 Saved dataset after {folder_count} folders!")

# Final save (if remaining folders < 5)
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("🎯 Dataset processing completed!")
print("✅ Final dataset saved as 'data.pickle'.")
