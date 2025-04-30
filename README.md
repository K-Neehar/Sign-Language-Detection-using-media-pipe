# Sign Language Detector

A real-time Sign Language Detector using Python, OpenCV, and Mediapipe.

## 🚀 Features
- Detects hand gestures for sign language recognition
- Uses **OpenCV** for image processing
- Utilizes **Mediapipe Hands** for hand landmark detection
- Trained with a custom dataset using **Machine Learning**

## 🛠 Requirements
- Python **3.11** (recommended)
- OpenCV
- Mediapipe
- NumPy
- Scikit-learn

Install dependencies using:
```sh
pip install -r requirements.txt
```

## 📌 Steps to Run
1. **Collect images** for training:
   ```sh
   python collect_imgs.py
   ```
2. **Create dataset** from collected images:
   ```sh
   python create_dataset.py
   ```
3. **Train the model**:
   ```sh
   python train_classifier.py
   ```
4. **Run inference for sign detection**:
   ```sh
   python inference_classifier.py
   ```

## 📝 Notes
- Ensure your **webcam** is connected for real-time detection.
- If the model predicts an unknown label, update `labels_dict` accordingly in `inference_classifier.py`.

---

