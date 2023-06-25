# DeepFace-Test

import cv2
from deepface import DeepFace
import numpy as np

# How to install the APIs libraries for facial recognition
#pip install opencv-python
#pip install deepface
#pip install numpy

# This line detects the face adn the classifier will check for the eyes and once it finds it, it will normalize the face image size
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# This code is using a live feed from webcam to detect the faces and emotions
cap = cv2.VideoCapture(0)

# This set will show real time emotions from the webcam on a loop
while True:
    _, frame = cap.read()
    analyze = DeepFace.analyze(frame, actions=['emotion'])
    print(analyze[0]['dominant_emotion'])

# This line is providing color to the video
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    faces = face_cascade.detectMultiScale(img, 1.1, 5, minSize=(40, 40))

# This function is analyzing the emotions of the person in real time from the webcam.
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
        try:
            analyze = DeepFace.analyze(frame,actions=['emotion'])
            cv2.putText(img, analyze['dominant_emotion'], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (224, 77, 176), 2)
            print(analyze[0]['emotion'])
        except:
            print("no face")

    cv2.imshow('frame', frame)

# This function is were you stop the program when you click on the letter "q"
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


