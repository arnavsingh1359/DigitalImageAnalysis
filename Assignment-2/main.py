import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

import image_pyramid

if __name__ == '__main__':
    name = "ryan"
    path = "C:\\Users\\Arnav Singh\\PycharmProjects\\DigitalImageAnalysis\\Assignment-2\\Images\\Part " \
           "1\\" + name + ".png"
    image = cv.imread(path, 0)
    plt.xlim((0, image.shape[1]))
    plt.ylim((image.shape[0], 0))
    lay = 3
    gaussian = image_pyramid.gaussian_pyramid(image, lay)
    laplacian = image_pyramid.laplacian_pyramid(gaussian)
    recons = image_pyramid.reconstruct_lap(laplacian)
    plt.imshow(recons, "gray")
    # for i in range(lay):
    #     plt.imshow(laplacian[i], cmap="gray")
        # reduced = image_pyramid.gaussian_pyramid(reduced)
    plt.show()
