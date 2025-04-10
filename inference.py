import joblib
import face_recognition
import cv2
import numpy as np

clf = joblib.load("model.pkl")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) 
if not cap.isOpened():
    print("Error: Kamera tidak bisa dibuka.")
    exit()

# Pengaturan performa
SKIP_FRAMES = 2  
RESIZE_SCALE = 0.5 
SHOW_FPS = True    

print("Real-time Face Recognition - Tekan 'Q' untuk keluar")

frame_count = 0
fps = 0
start_time = cv2.getTickCount()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal mengambil frame")
        break

    frame_count += 1
    if frame_count % SKIP_FRAMES != 0:
        continue

    small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)
    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
    face_locations = face_recognition.face_locations(rgb_small_frame)

    if face_locations:
        try:
            face_encodings = face_recognition.face_encodings(
                rgb_small_frame,
                known_face_locations=face_locations,
                num_jitters=1
            )

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Kembalikan koordinat ke skala asli
                top = int(top / RESIZE_SCALE)
                right = int(right / RESIZE_SCALE)
                bottom = int(bottom / RESIZE_SCALE)
                left = int(left / RESIZE_SCALE)
                name = "Unknown"
                try:
                    name = clf.predict([face_encoding])[0]
                except Exception as e:
                    print(f"Kesalahan prediksi: {e}")
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        except Exception as e:
            print(f"Kesalahan saat proses wajah: {e}")
            continue

    if SHOW_FPS:
        current_time = cv2.getTickCount()
        time_diff = (current_time - start_time) / cv2.getTickFrequency()
        fps = 1 / time_diff
        start_time = current_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Program selesai dengan bersih")
