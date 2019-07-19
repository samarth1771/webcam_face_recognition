import cv2

# Initializing haarcascade
face_cascade = cv2.CascadeClassifier('haarcascades_xml/haarcascade_frontalface_default.xml')
# Initializing Video capture with CV2
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()     # Capturing frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      # converting frame into grayscale  image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5) # Detecting face with the cascade provided
    for (x, y, w, h) in faces:
        # print(x, y, w, h)     # printing coordinates of detected face in image
        roi_color = frame[y:y + h, x:x + w]     # Cropping region of interest in normal image
        cv2.rectangle(frame, (x, y), (x + w, y + h), (250, 255, 0), 2)

    # Printing frames with rectangle on face with imshow

    cv2.imshow("Frames", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
