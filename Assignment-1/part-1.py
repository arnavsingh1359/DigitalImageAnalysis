import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from skimage import exposure
from scipy.signal import argrelextrema
import seaborn as sns

img = np.array(cv.imread("Pictures/part1/img_5.png"))
# lum, alpha, beta = np.array_split(cv.cvtColor(img, cv.COLOR_BGR2LAB), 3, axis=2)
# hue, sat, value = np.array_split(cv.cvtColor(img, cv.COLOR_BGR2HSV), 3, axis=2)




def detected_skin(image, alpha_a=6.5, beta_b=12, sat_min=15, sat_max=170, hue_max=17.1):
    hue, sat, value = np.array_split(cv.cvtColor(image, cv.COLOR_BGR2HSV), 3, axis=2)
    Y,Cr,Cb = np.array_split(cv.cvtColor(image,cv.COLOR_BGR2YCR_CB),3,axis =2)
    hue = hue[:, :, 0]
    sat = sat[:, :, 0]
    value = value[:, :, 0]
    Y = Y[:, :, 0]
    Cr = Cr[:, :, 0]
    Cb = Cb[:,:,0]
    result = np.zeros(hue.shape)			
    p = 0
    while p < result.shape[0]:
        q = 0
        while q < result.shape[1]:            
            if sat_min <= sat[p, q] <= sat_max and hue[p, q] <= hue_max :
                if 135<=Cr[p,q]<=180 and 85<=Cb[p,q]<=135:
                    result[p, q] = 1
            q += 1
        p += 1

    return result

def detect_faces(image, scale_factor=1.01, min_neighbor=6):
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
    lum, alpha, beta = np.array_split(cv.cvtColor(image, cv.COLOR_BGR2Lab), 3, axis=2)
    lum = lum[:, :, 0]
    alpha = alpha[:, :, 0]
    beta = beta[:, :, 0]
    skin_mask = detected_skin(image)
    plt.imshow(skin_mask,cmap ='gray')
    faces = detect_faces(image, 1.1, 5)
    # plt.imshow(skin_mask, cmap='gray')
    for (x, y, w, h) in faces:
        skin_in_face = skin_mask[y:y + w, x:x + h]
        #plt.imshow(skin_in_face,cmap ='gray')
        # p = sns.distplot(lum[y:y + w, x:x + h], bins=256)
        # intensity, density = p.lines[0].get_data()
        # # plt.plot(intensity, density)
        # # plt.xlim((0, 255))
        # maxima = argrelextrema(density, np.greater)
        # minima = argrelextrema(density, np.less)
        # d, b, m = is_bimodal(intensity, density)
        # A = np.ones_like(lum)
        # if d and b and m:
        #     f = (b - d) / (m - d)
        #     r = 0
        #     while r < skin_in_face.shape[0]:
        #         s = 0
        #         while s < skin_in_face.shape[1]:
        #             if skin_in_face[r, s] and lum[y + r, x + s] < m:
        #                 A[y + r, x + s] = f
        #             s += 1
        #         r += 1
        # final = lum * A
        # plt.scatter(intensity[maxima], density[maxima], color='green')
        # plt.scatter(intensity[minima], density[minima], color='red')
        # plt.imshow(final, cmap='gray')


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
