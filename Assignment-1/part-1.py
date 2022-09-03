import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from skimage import exposure
from scipy.signal import argrelextrema
import seaborn as sns

# fig, ax = plt.subplots(1, 2)

img = np.array(cv.imread("Pictures\\img_5.png"))
# lum, alpha, beta = np.array_split(cv.cvtColor(img, cv.COLOR_BGR2LAB), 3, axis=2)
# hue, sat, value = np.array_split(cv.cvtColor(img, cv.COLOR_BGR2HSV), 3, axis=2)


def detect_skin(alpha, beta, hue, sat):
    result = np.zeros((alpha.shape[0], alpha.shape[1]))
    p = 0
    while p < alpha.shape[0]:
        q = 0
        while q < alpha.shape[1]:
            if ((alpha[p, q, 0] - 143) / 6.5) ** 2 + ((beta[p, q, 0] - 148) / 12) ** 12 < 1:
                if 64 <= sat[p, q, 0] <= 192 and hue[p, q, 0] <= 17.1:
                    result[p, q] = 1
            q += 1
        p += 1

    return result


def detect_faces(image, scale_factor, min_neighbor):
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, scale_factor, min_neighbor)
    return faces


def is_bimodal(x, y):
    maxima = argrelextrema(y, np.greater)[0]
    minima = argrelextrema(y, np.less)[0]
    if len(maxima) >= 2:
        i = 0
        while i < len(maxima) - 1 and len(minima) > 0:
            d = y[maxima[i]]
            b = y[maxima[i + 1]]
            m = y[minima[i]]
            if m < 0.8 * d and m < 0.8 * b:
                return int(x[maxima[i]]), int(x[maxima[i + 1]]), int(x[minima[i]])
            i += 1

    return None, None, None


def sidelight_correction(image):
    lum, alpha, beta = np.array_split(cv.cvtColor(image, cv.COLOR_BGR2LAB), 3, axis=2)
    hue, sat, value = np.array_split(cv.cvtColor(image, cv.COLOR_BGR2HSV), 3, axis=2)
    skin_mask = detect_skin(alpha, beta, hue, sat)
    faces = detect_faces(image, 1.001, 5)
    for (i, j, w, h) in faces:
        skin_in_face = skin_mask[j:j + w, i:i + h]
        p = sns.distplot(lum[j:j + w, i:i + h, 0], bins=256)
        x, y = p.get_lines()[0].get_data()
        # plt.plot(x, y)
        plt.xlim((0, 255))
        maxima = argrelextrema(y, np.greater)
        minima = argrelextrema(y, np.less)
        d, b, m = is_bimodal(x, y)
        A = np.ones_like(lum)
        if d and b and m:
            f = (b - d) / (m - d)
            r = 0
            while r < skin_in_face.shape[0]:
                s = 0
                while s < skin_in_face.shape[1]:
                    if skin_in_face[r, s] and lum[j + r, i + s] < m:
                        A[j + r, i + s] = f
                    s += 1
                r += 1
        final = lum * A
        # plt.scatter(x[maxima], y[maxima], color='green')
        # plt.scatter(x[minima], y[minima], color='red')
        plt.imshow(final, cmap='gray')


sidelight_correction(img)

# faces = detect_faces(img, 1.001, 10)

# for (x, y, w, h) in faces:
# face = faces[0, :]
# x = face[0]
# y = face[1]
# w = face[2]
# h = face[3]
# ax[0].imshow(img)
# patch = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor='red', facecolor='none')
# ax[0].add_patch(patch)
# hist = exposure.histogram(l[y:y+w, x:x+h, 0], nbins=1024)
# ax[1].bar(hist[1], hist[0], color='blue')
# ax[1].imshow(detect_skin(alpha, beta, hue, sat), cmap="gray")
    # cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
    # cv.imshow("Image1", img)
    # cv.waitKey(0)
# img1 = l[y:y+w, x:x+h]
# ax[1].imshow(img1, cmap="gray")
# cv.imshow("Image1", img1)
# cv.waitKey(0)

# cv.imshow("Image1", img)

# cv.waitKey(0)


plt.show()
