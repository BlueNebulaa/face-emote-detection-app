import cv2
import numpy as np
from joblib import load
from Random_forest import RandomForest_manual,DecisionTree,Node
from skimage.feature import hog
import os
import sys


if getattr(sys, 'frozen', False):  # Jika dibundel dengan PyInstaller
    base_path = sys._MEIPASS
else:
    base_path = os.path.abspath(".")

# Path ke file XML dan model
xml_path = os.path.join(base_path, "face.xml")
model_path = os.path.join(base_path, "random_forest_model.joblib")
face_ref = cv2.CascadeClassifier(xml_path)
camera = cv2.VideoCapture(0)
model = load(model_path)

def face_detection(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
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
    for (x, y, w, h) in faces:
        # Ekstraksi fitur wajah menggunakan HOG
        hog_features = extract_face(gray_frame, (x, y, w, h)).reshape(1, -1)  # Tambah dimensi batch
        
        # Prediksi emosi menggunakan model
        prediction = model.predict(hog_features)[0]
        
        # Label emosi (sesuaikan dengan model Anda)
        emotion_labels = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4:'Neutral', 5:"Sad",6:"Surprise"}
        emotion = emotion_labels.get(prediction, "Unknown")
        
        # Gambar kotak dan teks pada frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

def close_window():
    camera.release()
    cv2.destroyAllWindows()
    exit()
    
def main():
    while True :
        _,frame = camera.read()
        drawer_box(frame)
        cv2.imshow("Project AI", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            close_window()
        

if __name__ == '__main__':
    main()

