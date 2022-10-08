import math
import cv2 as cv
import numpy as np


def gaussian_pyramid(image, downscale=2, sigma=None):
    out_shape = tuple(math.ceil(d / float(downscale)) for d in image.shape)
    if sigma is None:
        sigma = 2 * downscale / 6.0
    kernel_size = 5
    smoothed = cv.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    out = _bi_inter(smoothed, out_shape)
    return out


def _bi_inter(image, dimension):
    height = image.shape[0]
    width = image.shape[1]
    scale_x = width / (dimension[1])
    scale_y = height / (dimension[0])

    new_image = np.zeros((dimension[0], dimension[1]))
    for i in range(dimension[0]):
        for j in range(dimension[1]):
            x = (j + 0.5) * scale_x - 0.5
            y = (i + 0.5) * scale_y - 0.5

            x_int = int(x)
            y_int = int(y)

            x_int = min(x_int, width - 2)
            y_int = min(y_int, height - 2)

            x_diff = x - x_int
            y_diff = y - y_int

            a = image[y_int, x_int]
            b = image[y_int, x_int + 1]
            c = image[y_int + 1, x_int]
            d = image[y_int + 1, x_int + 1]

            pixel = a * (1 - x_diff) * (1 - y_diff) + b * x_diff * (1 - y_diff) + \
                    c * (1 - x_diff) * y_diff + d * x_diff * y_diff

            new_image[i, j] = pixel
    return new_image.astype(np.uint8)
