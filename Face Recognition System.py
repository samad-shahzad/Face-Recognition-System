import cv2
import os
import numpy as np

def train_recognizer(data_path="training_data"):
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    faces = []
    labels = []
    label_names = {}
    current_id = 0

    print("Training...")

    for person_name in os.listdir(data_path):
        person_path = os.path.join(data_path, person_name)

        if not os.path.isdir(person_path):
            continue

        label_names[current_id] = person_name

        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)

            img = cv2.imread(img_path)
            if img is None:
                continue  # skip bad images

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces_detected = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces_detected:
                face_roi = gray[y:y + h, x:x + w]

                # normalize face size
                face_roi = cv2.resize(face_roi, (200, 200))

                faces.append(face_roi)
                labels.append(current_id)

        current_id += 1

    if len(faces) == 0:
        print(" No faces found. Check dataset.")
        return None, None, None

    recognizer.train(faces, np.array(labels))
    recognizer.save("trained_model.yml")

    print(f" Trained on {len(faces)} faces for {len(label_names)} people.")
    return recognizer, face_cascade, label_names


# ==========================
# RUN SYSTEM
# ==========================
recognizer, face_cascade, labels = train_recognizer()

cap = cv2.VideoCapture(0)

print("Running... Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]

        # same resize as training
        face_roi = cv2.resize(face_roi, (200, 200))

        label_id, confidence = recognizer.predict(face_roi)

        name = labels.get(label_id, "Unknown")

        # stricter threshold
        if confidence > 69:
            name = "Unknown"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({int(confidence)})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()