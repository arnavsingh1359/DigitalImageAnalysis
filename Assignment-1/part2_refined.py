import cv2 as cv
import numpy as np
from skimage import exposure
import random
import time

imgcolor = np.array(cv.imread("Pictures/part2/img_1a.png"))
# imgcolorroi = cv.selectROI(imgcolor)
imggray = np.array(cv.imread("Pictures/part2/img_1b.png", 0))
# roi_cropped = imgcolor[int(imgcolorroi[1]):int(imgcolorroi[1] + imgcolorroi[3]),
#               int(imgcolorroi[0]):int(imgcolorroi[0] + imgcolorroi[2])]
# cv.imshow("ROI", roi_cropped)
# cv.waitKey(0)


def make_swatches(color_image, gray_image, n_swatches=2):
    swatches = []

    n = 0
    while n < n_swatches:
        cr1 = cv.selectROI(color_image)
        sw1 = color_image[int(cr1[1]):int(cr1[1] + cr1[3]), int(cr1[0]):int(cr1[0] + cr1[2])]
        cr2 = cv.selectROI(gray_image)
        sw2 = gray_image[int(cr2[1]):int(cr2[1] + cr2[3]), int(cr2[0]):int(cr2[0] + cr2[2])]
        swatches.append([sw1, sw2])
        n += 1

    return swatches


make_swatches(imgcolor, imggray)


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

    return indices.astype("uint8")


def match_color(color_image, gray_image, swatches=None, n_samples=200, window_size=5, iterations=10):
    lum_color, alpha_color, beta_color = np.array_split(cv.cvtColor(color_image, cv.COLOR_BGR2LAB), 3, axis=2)
    lum_color = lum_color[:, :, 0]
    alpha_color = alpha_color[:, :, 0]
    beta_color = beta_color[:, :, 0]
    lum_gray = gray_image
    alpha_gray = np.zeros_like(lum_gray)
    beta_gray = np.zeros_like(lum_gray)

    equalized = exposure.match_histograms(lum_gray, lum_color).astype("uint8")
    nbd_gray = nbd_stat(equalized, window_size=window_size)
    for ite in range(1, iterations + 1):
        print(f"iterations = {ite}")
        # color_samples = generate_samples(color_image, n_samples)
        intensity_hash = {inten: [] for inten in range(0, 256)}
        color_samples = np.empty((n_samples, 2))
        count = 0
        while count <= n_samples:
            i = random.randint(0, lum_color.shape[0] - 1)
            j = random.randint(0, lum_color.shape[1] - 1)
            inten = lum_color[i, j]
            if not intensity_hash[inten]:
                intensity_hash[inten] = [i, j]
                count += 1

        i = 0
        while i < equalized.shape[0]:
            j = 0
            while j < equalized.shape[1]:
                l = int(equalized[i, j] / 2 + nbd_gray[i, j] / 2)
                # print(intensity_hash[l])
                closest = intensity_hash[l]
                l1 = l + 1
                l2 = l - 1
                while not closest:
                    if l1 < 256 and intensity_hash[l1]:
                        closest = intensity_hash[l1]
                    elif l2 > -1 and intensity_hash[l2]:
                        closest = intensity_hash[l2]
                    else:
                        l1 += 1
                        l2 -= 1
                alpha_gray[i, j] = alpha_color[closest[0], closest[1]]
                beta_gray[i, j] = beta_color[closest[0], closest[1]]
                j += 1
            i += 1
        colorized = np.stack((lum_gray, alpha_gray, beta_gray), axis=2)
        colorized_bgr = cv.cvtColor(colorized, cv.COLOR_Lab2BGR)
        filename = f"Pictures/result/img_1re{ite}.png"
        cv.imwrite(filename, colorized_bgr)

# start = time.time()
# match_color(imgcolor, imggray)
# end = time.time()
# print(end - start)
