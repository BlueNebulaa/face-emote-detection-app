import cv2
import numpy as np
from joblib import load
from Random_forest import RandomForest_manual,DecisionTree,Node
from skimage.feature import hog
import streamlit as st

# Load model dan cascade classifier
face_ref = cv2.CascadeClassifier("face.xml")
model = load("random_forest_model.joblib")

def face_detection(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_ref.detectMultiScale(gray_frame, scaleFactor=1.1, minSize=(48, 48), minNeighbors=3)
    return faces, gray_frame

def extract_face(gray_frame, face_coords):
    x, y, w, h = face_coords
    face_region = gray_frame[y:y+h, x:x+w]
    face_resized = cv2.resize(face_region, (48, 48))
    hog_features = hog(
        face_resized,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False
    )
    face_normalized = hog_features / 255.0
    face_flatten = face_normalized.flatten()
    return face_flatten

def drawer_box(frame):
    faces, gray_frame = face_detection(frame)
    result_frame = frame.copy()
    for (x, y, w, h) in faces:
        # Ekstraksi fitur wajah menggunakan HOG
        hog_features = extract_face(gray_frame, (x, y, w, h)).reshape(1, -1)
        
        # Prediksi emosi menggunakan model
        prediction = model.predict(hog_features)[0]
        
        # Label emosi
        emotion_labels = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: 'Neutral', 5: "Sad", 6: "Surprise"}
        emotion = emotion_labels.get(prediction, "Unknown")
        
        # Gambar kotak dan teks pada frame
        cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cv2.putText(result_frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return result_frame

# Streamlit Application
st.title("Emotion Detection with OpenCV and Random Forest")

# Tempat untuk menampilkan frame
frame_placeholder = st.empty()

# Add a "Start" and "Stop" button and manage their state
start_button_pressed = st.button("Start Webcam")
stop_button_pressed = st.button("Stop")

if start_button_pressed:
    # Capture video from webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened() and not stop_button_pressed:
        ret, frame = cap.read()

        if not ret:
            st.write("The video capture has ended.")
            break

        # Deteksi emosi dan gambar kotak pada frame
        result_frame = drawer_box(frame)

        # Konversi dari BGR ke RGB
        result_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

        # Tampilkan frame
        frame_placeholder.image(result_frame, channels="RGB")

        # Break jika tombol "Stop" ditekan
        if stop_button_pressed:
            break

    cap.release()
    cv2.destroyAllWindows()
