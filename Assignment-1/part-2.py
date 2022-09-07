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
cv.imshow("Luminance Image", lum_color)
cv.imshow("Gray Image", imggray)
cv.imshow("Matched gray Image", matched)
cv.waitKey(0)
