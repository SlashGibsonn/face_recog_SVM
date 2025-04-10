import face_recognition
from sklearn import svm
import os
import joblib

encodings = []
names = []

train_path = 'train_dir/'
train_dir = os.listdir(train_path)

for person in train_dir:
    person_path = os.path.join(train_path, person)

    if not os.path.isdir(person_path):
        print(f"{person_path}, dilewati karena bukan folder.")
        continue

    person_images = os.listdir(person_path)

    # Loop setiap gambar orang tersebut
    for person_img in person_images:
        face = face_recognition.load_image_file(os.path.join(train_path, person, person_img))
        face_bounding_boxes = face_recognition.face_locations(face)

        # Gunakan hanya gambar dengan 1 wajah
        if len(face_bounding_boxes) == 1:
            face_enc = face_recognition.face_encodings(face)[0]
            encodings.append(face_enc)
            names.append(person)
        else:
            print(person + "/" + person_img + " dilewati karena tidak mengandung tepat satu wajah.")

print(f"Jumlah data training: {len(encodings)}")
print(f"Label unik: {set(names)}")

clf = svm.SVC(gamma='scale')
clf.fit(encodings, names)

joblib.dump(clf, "model.pkl")
