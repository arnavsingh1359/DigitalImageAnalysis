import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

image = np.array(cv.imread("F:\\IITD\\2022-1\\COL783-Digital Image "
                           "Analysis\\Assignments\\Assignment-1\\pictures\\part1\\img.png"))
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(image, 1.001, 6)

for (x, y, w, h) in faces:
    cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv.imshow('Faces', image)
cv.waitKey(0)