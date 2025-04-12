import joblib
import face_recognition
import cv2
import numpy as np
import os

clf = joblib.load("model.pkl")

def detect_faces(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        try:
            name = clf.predict([face_encoding])[0]
        except:
            name = "Unknown"
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return frame, len(face_locations)

def save_output(output_path, frame):
    try:
        cv2.imwrite(output_path, frame)
        print(f"Output berhasil disimpan di: {output_path}")
    except:
        print("Gagal menyimpan output")

input_file = "test_image.png" 

save_option = input("Simpan output? (y/n): ").lower()
output_path = None
if save_option == 'y':
    output_path = input("Masukkan nama file output (contoh: output.jpg): ")
    
if input_file.lower().endswith(('.png', '.jpg', '.jpeg')):
    # Image mode
    frame = cv2.imread(input_file)
    if frame is None:
        print(f"Error: Tidak bisa membaca file gambar {input_file}")
    else:
        processed_frame, num_faces = detect_faces(frame)
        print(f"Detected {num_faces} faces")
        
        cv2.imshow("Face Recognition", processed_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if save_option == 'y' and output_path:
            save_output(output_path, processed_frame)
        
elif input_file.lower().endswith(('.mp4', '.avi', '.mov')) or input_file.isdigit():
    # Video mode
    cap = cv2.VideoCapture(int(input_file) if input_file.isdigit() else input_file)
    
    if not cap.isOpened():
        print(f"Error: Tidak bisa membuka video source {input_file}")
    else:
        writer = None
        if save_option == 'y' and output_path:
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            fps = cap.get(cv2.CAP_PROP_FPS)
            writer = cv2.VideoWriter(output_path, 
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps, 
                                   (frame_width, frame_height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame, num_faces = detect_faces(frame)
            print(f"Detected {num_faces} faces in current frame")
            
            cv2.imshow("Face Recognition", processed_frame)
            
            if writer is not None:
                writer.write(processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
else:
    print("Format file tidak didukung. Gunakan file gambar (jpg/png) atau video (mp4/avi) atau nomor webcam")