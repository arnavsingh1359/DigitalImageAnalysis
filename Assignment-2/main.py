import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

import image_pyramid

if __name__ == '__main__':
    name = "apple"
    path = "C:\\Users\\Arnav Singh\\PycharmProjects\\DigitalImageAnalysis\\Assignment-2\\Images\\Part " \
           "1\\" + name + ".png"
    image = cv.imread(path, 0)
    plt.imshow(image, cmap="gray")
    plt.xlim((0, image.shape[1]))
    plt.ylim((image.shape[0], 0))
    reduced = image_pyramid.gaussian_pyramid(image)
    for i in range(3):
        plt.imshow(reduced, cmap="gray")
        reduced = image_pyramid.gaussian_pyramid(reduced)
    plt.show()
