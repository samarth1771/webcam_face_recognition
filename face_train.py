import os
import cv2
import numpy as np
import pickle


# Initializing directories for making a face detection model

base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir, "face_data")

face_cascade = cv2.CascadeClassifier('haarcascades_xml/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()       # Face recognizer with LBPH method

current_id = 0
y_lables = []
x_train = []
label_id = {}

# Loop for training every directory in face_data folder and give label according to it

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            # print(label, path)
            if not label in label_id:
                label_id[label] = current_id
                current_id += 1
            _id = label_id[label]
            # print(_id)
            # print(label_id)
            # x_train.append(path)  # file name
            # y_lables.append(label)  # foldername
            pil_image = cv2.imread(path, 0)
            image_array = np.array(pil_image, "uint8")
            # print(image_array)
            faces = face_cascade.detectMultiScale(image_array, minNeighbors=5, scaleFactor=1.2)
            for (x, y, w, h) in faces:
                roi = image_array[y:y + h, x:x + w]
                x_train.append(roi)
                y_lables.append(_id)


# Saving data to a pickle

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_id, f)

# Training a model from a recognizer model and saving it

recognizer.train(x_train, np.array(y_lables))
recognizer.save("trainer.yml")
print(label_id)
