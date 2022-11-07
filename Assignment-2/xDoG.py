import cv2 as cv
import numpy as np
from skimage import filters


def thresholder(u, e, phi):
    result = np.ones_like(u)
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            if u[i, j] < e:
                result[i, j] = 1 + np.tanh(phi * (u[i, j] - e))
    return 255 * result.astype("uint8")


def normalize(img):
    mi = img.min()
    ma = img.max()
    result = (img - mi) / (ma - mi) * 255
    return result


if __name__ == "__main__":
    image = cv.imread("Images\\Part 3\\man2.png", 0)
    size = 9
    sigma = 1.3
    k = 1.3
    ks = k * sigma
    p = 18
    blurred = filters.gaussian(image, sigma)
    blurred_2 = filters.gaussian(image, ks)
    difference = blurred - blurred_2
    new = normalize(blurred + p * difference)
    final = thresholder(new, 82.2, 0.06)
    cv.imshow("original", image)
    cv.imshow("blurred", blurred)
    cv.imshow("blurred_2", blurred_2)
    cv.imshow("difference", difference)
    cv.imshow("new", new.astype("uint8"))
    cv.imshow("final", final)
    cv.waitKey(0)
    cv.destroyAllWindows()
