import cv2
import os

# Initializing haarcascade
face_cascade = cv2.CascadeClassifier('haarcascades_xml/haarcascade_frontalface_default.xml')
name = input("What is your name")

cap = cv2.VideoCapture(0)
count = 0
max = 150
dir_name = "face_data/" + name      # Directory name as "face_data/abc"
os.mkdir(dir_name)                  # Making new directory
while count < max:                  # Condition for taking 150 pics
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)          # Converting to grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)    # Detecting face with the cascade provided
    for (x, y, w, h) in faces:
        print(x, y, w, h)
        cv2.rectangle(frame, (x, y), (x + h, y + w), (250, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]

        # Saving gray region of interest frames into directory

        if count < max:
            path_name = "face_data/" + name + "/" + str(count) + ".jpg"
            cv2.imwrite(path_name, roi_gray)
    cv2.imshow("Frames", frame)
    count += 1
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
