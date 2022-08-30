import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = np.array(cv.imread("lena256.png"))
img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)
y,u,v = cv.split(img_yuv)
cv.imshow('y',y)
cv.waitKey(0)