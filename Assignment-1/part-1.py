import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from skimage import exposure

fig, ax = plt.subplots(1, 2)


img = np.array(cv.imread("Pictures\\img_5.png"))

l, a, b = np.array_split(cv.cvtColor(img, cv.COLOR_BGR2LAB), 3, axis=2)


def detect_faces(image, scale_factor, min_neighbor):
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, scale_factor, min_neighbor)
    return faces


faces = detect_faces(img, 1.1, 6)

for (x, y, w, h) in faces:
    ax[0].imshow(l, cmap="gray")
    patch = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='red', facecolor='none')
    ax[0].add_patch(patch)
    hist = exposure.histogram(l[y:y+w, x:x+h], nbins=150)
    ax[1].plot(hist[1], hist[0])
    # cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
    # img1 = l[y:y+w, x:x+h]
    # ax[1].imshow(img1, cmap="gray")
    # cv.imshow("Image1", img1)
    # cv.waitKey(0)

# cv.imshow("Image1", img)

# cv.waitKey(0)
plt.show()