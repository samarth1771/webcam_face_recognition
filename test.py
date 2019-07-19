import cv2

import pickle

face_cascade = cv2.CascadeClassifier('haarcascades_xml/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

og_labels = {}

# Fetching data from pickle file and appending labels

with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

cap = cv2.VideoCapture(0)
count = 0
_id = ""
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        # print(x, y, w, h)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        _id, conf = recognizer.predict(roi_gray)
        # print(conf)

        # Put text on the face if confidence is less than 70

        if conf < 70:
            print(_id)
            print(labels[_id])
            cv2.putText(frame, labels[_id], (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (250, 255, 0), 2)

    cv2.imshow("Frames", frame)
    count += 1
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
print(labels)
