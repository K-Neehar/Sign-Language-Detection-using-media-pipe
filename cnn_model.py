import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'])  # Hand landmarks
labels = np.asarray(data_dict['labels'])  # Alphabet labels

# Encode labels (convert A-Z to numerical values)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Convert labels to categorical (one-hot encoding)
labels = to_categorical(labels)

# Reshape data for CNN (42 landmarks per hand, 2 coordinates each)
data = np.array(data).reshape(-1, 42, 2, 1)  # (samples, landmarks, coordinates, channels)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels)

# Normalize data
x_train = x_train / np.max(x_train)
x_test = x_test / np.max(x_test)

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 1), activation='relu', input_shape=(42, 2, 1)),  # Expecting 42 landmarks
    MaxPooling2D((2, 1)),
    
    Conv2D(64, (3, 1), activation='relu'),
    MaxPooling2D((2, 1)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')  # Output layer
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))

# Save model
model.save('sign_language_model.h5')
with open('label_encoder.p', 'wb') as f:
    pickle.dump(label_encoder, f)

print("✅ Model saved as 'sign_language_model.h5'!")
