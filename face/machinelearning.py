# Import necessary modules
import cv2           # OpenCV for image processing
import numpy as np   # NumPy for numerical operations
import pandas as pd  # pandas for data processing and analysis
import tensorflow as tf  # TensorFlow for deep learning
import os
from retinaface import RetinaFace  # RetinaFace for face detection
import matplotlib.pyplot as plt
from django.conf import settings

STATIC_DIR = settings.STATIC_DIR

model_path = 'C:/Users/ALRYADA/Desktop/img model/CNN_emotion_detection.h5'
emotion_recognition_model = tf.keras.models.load_model(model_path)

prob = []
imgs = []


def faces_detection(image_path):
    # Load the input image
    image = cv2.imread(image_path)

    # Convert the image to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces in the input image using RetinaFace
    faces = RetinaFace.detect_faces(image_rgb)

    # Loop over the detected faces and highlight the facial areas
    for face in faces.keys():
        entity = faces[face]
        facial_area = entity["facial_area"]

        # Highlight the facial area with a white rectangle
        cv2.rectangle(image, (facial_area[2], facial_area[3]),
                      (facial_area[0], facial_area[1]), (0, 255, 0), 4)
       # resized_face_region = cv2.resize(face, (48, 48))
       # preprocessed_face_region = np.expand_dims(
        #    resized_face_region, axis=0) / 255.0

        # Convert the face region from RGB to BGR format
       # new_img = cv2.cvtColor(resized_face_region, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(settings.MEDIA_ROOT,
                                 'mloutput/process.jpg'), image)
        # Make emotion prediction using the pre-trained model and print the result
        # prob_sad = emotion_recognition_model.predict(
        #   preprocessed_face_region)[0][0]
       # print('Probability of being sad:', prob_sad)


def face_emotion(image_path):
    # Load the input image
    image = cv2.imread(image_path)

    # Convert the image to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Extract the face regions from the input image using RetinaFace
    faces = RetinaFace.extract_faces(image_rgb, align=True)
    cv2.imwrite(os.path.join(settings.MEDIA_ROOT,
                             'mloutput/process.jpg'), image)
    # Preprocess each face region and make emotion predictions
    c = 1
    paths = []
    prob = []
    prob_sad = []
    for face in faces:
        # Resize and preprocess the face region for input to the emotion recognition model
        resized_face_region = cv2.resize(face, (48, 48))
        preprocessed_face_region = np.expand_dims(
            resized_face_region, axis=0) / 255.0
        PATH = f"C:/Users/ALRYADA/Desktop/Team_26_Graduation_Project/face/static/images/process{c}.jpg"
        p2 = f"/images/process{c}.jpg"
        paths.append(p2)
        #PATH = os.path.join(settings.MEDIA_ROOT, f'mloutput/process{c}.jpg')
        print(PATH)
        c += 1
        # Convert the face region from RGB to BGR format
        new_img = cv2.cvtColor(resized_face_region, cv2.COLOR_RGB2BGR)
        cv2.imwrite(PATH, new_img)
        # Make emotion prediction using the pre-trained model and print the result
        prob_sad.append(emotion_recognition_model.predict(
            preprocessed_face_region)[0][0])
        percentage = prob_sad[-1] * 100
        formatted_percentage = "{:.2f}%".format(percentage)
      #  imgs.append(new_img)
        prob.append(formatted_percentage)
        # plt.imshow(new_img)
        # plt.axis('off')
        # plt.show()
        # print('Probability of being sad:', prob_sad)
        # print(face)
        faces_detection(image_path)
        print(prob, paths, prob_sad)
    return prob, paths, prob_sad
