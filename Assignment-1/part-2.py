import cv2 as cv
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt
import seaborn as sns

imgcolor = np.array(cv.imread("Pictures/part2/img_1a.png"))
lum_color, alpha_color, beta_color = np.array_split(cv.cvtColor(imgcolor, cv.COLOR_BGR2LAB), 3, axis=2)
lum_color = lum_color[:, :, 0]
alpha_color = alpha_color[:, :, 0]
beta_color = beta_color[:, :, 0]
imggray = np.array(cv.imread("Pictures/part2/img_1b.png", 0))
lum_gray = imggray
alpha_gray = np.zeros_like(lum_gray)
beta_gray = np.zeros_like(lum_gray)

matched = exposure.match_histograms(imggray, lum_color).astype('uint8')


def nbd_stat(lum, window_size=5):
    result = lum.copy()
    kernel_mean = np.ones((window_size, window_size)) / (window_size * window_size)
    mean = cv.filter2D(result, ddepth=-1, kernel=kernel_mean)
    result = np.sqrt(np.power(result - mean, 2) / window_size)

    return result


nbd_color = nbd_stat(lum_color)
nbd_gray = nbd_stat(matched)
cv.imshow("STD Image", nbd_gray)
# cv.imshow("Color Image", lum_color)
# cv.imshow("Gray Image", imggray)
cv.imshow("Matched gray Image", matched)
cv.waitKey(0)
