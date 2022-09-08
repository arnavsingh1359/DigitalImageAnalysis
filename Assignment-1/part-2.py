import cv2 as cv
import numpy as np
from skimage import exposure
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns

imgcolor = np.array(cv.imread("Pictures/part2/img_2a.png"))
lum_color, alpha_color, beta_color = np.array_split(cv.cvtColor(imgcolor, cv.COLOR_BGR2LAB), 3, axis=2)
lum_color = lum_color[:, :, 0]
alpha_color = alpha_color[:, :, 0]
beta_color = beta_color[:, :, 0]
imggray = np.array(cv.imread("Pictures/part2/img_2b.png", 0))
lum_gray = imggray
alpha_gray = np.zeros_like(lum_gray)
beta_gray = np.zeros_like(lum_gray)

equalized = exposure.match_histograms(imggray, lum_color).astype('uint8')


def nbd_stat(lum, window_size=5):
    result = lum.copy()
    kernel_mean = np.ones((window_size, window_size)) / (window_size * window_size)
    mean = cv.filter2D(result, ddepth=-1, kernel=kernel_mean)
    result = np.sqrt(np.power(result - mean, 2) / window_size)

    return result


def generate_samples(image, nsamples=200):
    indices = np.empty((nsamples, 2))
    n = 0
    while n < nsamples:
        i = random.randint(0, image.shape[0])
        j = random.randint(0, image.shape[1])
        indices[n] = np.array([i, j])
        n += 1

    return indices


def match_color():
    global lum_color, alpha_color, beta_color, equalized, alpha_gray, beta_gray, nbd_gray, color_samples
    i = 0
    while i < equalized.shape[0]:
        j = 0
        index_closest = [-1, -1]
        while j < equalized.shape[1]:
            l = equalized[i, j] / 2 + nbd_gray[i, j] / 2
            p = 0
            closest = np.inf
            while p < color_samples.shape[0]:
                indices_x, indices_y = color_samples[p].astype('uint8')
                if np.abs(l - lum_color[indices_x, indices_y]) < closest:
                    closest = np.abs(l - lum_color[indices_x, indices_y])
                    index_closest = [indices_x, indices_y]
                p += 1
            alpha_gray[i, j] = alpha_color[index_closest[0], index_closest[1]]
            beta_gray[i, j] = beta_color[index_closest[0], index_closest[1]]
            j += 1
        i += 1


color_samples = generate_samples(imgcolor)
nbd_gray = nbd_stat(equalized)
# print(color_samples)
time_start = time.time()
match_color()
time_end = time.time()
exe_time = time_end - time_start
print(exe_time)
colorized = np.stack((lum_gray, alpha_gray, beta_gray), axis=2)
colorized_bgr = cv.cvtColor(colorized, cv.COLOR_Lab2BGR)
cv.imshow("color matched Image", colorized_bgr)
cv.imshow("Color Image", lum_color)
cv.imshow("Gray Image", imggray)
# cv.imshow("Matched gray Image", equalized)
cv.waitKey(0)
