import cv2 as cv
import numpy as np
from skimage import exposure
import random
from localbinarypatterns import LocalBinaryPatterns
from sklearn.naive_bayes import GaussianNB
import time
import pandas


def make_swatches(color_image, gray_image, n_swatches=2):
    swatches = []

    n = 0
    while n < n_swatches:
        cr1 = cv.selectROI(color_image)
        cr2 = cv.selectROI(gray_image)
        swatches.append([cr1, cr2])
        n += 1
    cv.destroyAllWindows()
    return swatches


def nbd_stat(lum, window_size=5):
    result = lum.copy()
    kernel_mean = np.ones((window_size, window_size)) / (window_size * window_size)
    mean = cv.filter2D(result, ddepth=-1, kernel=kernel_mean)
    result = np.sqrt(np.power(result - mean, 2) / window_size)

    return result


def match_color(color_image, gray_image, n_samples=200, window_size=5):
    lum_color, alpha_color, beta_color = np.array_split(cv.cvtColor(color_image, cv.COLOR_BGR2LAB), 3, axis=2)
    lum_color = lum_color[:, :, 0]
    alpha_color = alpha_color[:, :, 0]
    beta_color = beta_color[:, :, 0]
    lum_gray = gray_image
    alpha_gray = np.zeros_like(lum_gray)
    beta_gray = np.zeros_like(lum_gray)

    equalized = exposure.match_histograms(lum_gray, lum_color).astype("uint8")
    nbd_stats = nbd_stat(equalized, window_size=window_size)
    intensity_hash = {inten: [] for inten in range(0, 256)}
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
            l = int(equalized[i, j] / 2 + nbd_stats[i, j] / 2)
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
    colorized_lab = np.stack((lum_gray, alpha_gray, beta_gray), axis=2)

    return colorized_lab


def match_color_swatch(color_image, gray_image, n_swatches, n_samples=50, window_size=5, nbd_size=3):
    lum_gray = gray_image
    alpha_gray = np.zeros_like(lum_gray)
    beta_gray = np.zeros_like(lum_gray)
    colorized_lab = np.stack((lum_gray, alpha_gray, beta_gray), axis=2)
    swatches = make_swatches(imgcolor, imggray, n_swatches)
    lum_csw = []
    alpha_csw = []
    beta_csw = []
    for swatch in swatches:
        cx, cy, cw, ch = swatch[0]
        gx, gy, gw, gh = swatch[1]
        colorized_swatch = match_color(color_image[cy:cy + ch, cx:cx + cw], gray_image[gy:gy + gh, gx:gx + gw],
                                       n_samples, window_size)
        colorized_lab[gy:gy + gh, gx:gx + gw, :] = colorized_swatch
        # y1 = int(gy + (gw - nbd_size) / 2)
        # y2 = int(gy + (gw + nbd_size) / 2)
        # x1 = int(gx + (gh - nbd_size) / 2)
        # x2 = int(gx + (gh + nbd_size) / 2)
        # lum_csw.append(colorized_lab[y1: y2, x1: x2, 0])
        # alpha_csw.append(colorized_lab[y1: y2, x1: x2, 1])
        # beta_csw.append(colorized_lab[y1: y2, x1: x2, 2])
        lum_csw.append(colorized_swatch[:, :, 0])
        alpha_csw.append(colorized_swatch[:, :, 1])
        beta_csw.append(colorized_swatch[:, :, 2])

    desc = LocalBinaryPatterns(24, 8)
    data = []
    labels = []

    for ind, col_swatch in enumerate(lum_csw):
        index = 0
        while index < 100:
            x = random.randint(0, col_swatch.shape[0] - nbd_size)
            y = random.randint(0, col_swatch.shape[1] - nbd_size)
            sample = col_swatch[y:y + nbd_size, x:x + nbd_size]
            if np.any(sample):
                hist = desc.describe(image=sample)
                labels.append(ind)
                data.append(hist)
                index += 1

    model = GaussianNB()
    model.fit(data, labels)
    c_lab = colorized_lab.copy()
    all_predictions = []
    i = 0
    while i < colorized_lab.shape[0] - nbd_size:
        j = 0
        while j < colorized_lab.shape[1] - nbd_size:
            # nbd = colorized_lab[i:i + nbd_size, j:j + nbd_size, :]
            nbd_gray = colorized_lab[i:i + nbd_size, j:j + nbd_size, :][:, :, 0]
            # min_err = np.inf
            # which = -1
            # for ind, col_swatch in enumerate(lum_csw):
            #     error = np.sum(np.power(nbd[:, :, 0] - col_swatch[0:nbd_size, 0:nbd_size], 2))
            #     if error < min_err:
            #         min_err = error
            #         which = ind
            # colorized_lab[j:j + nbd_size, i:i + nbd_size, :][:, :, 1] = alpha_csw[which]
            # colorized_lab[j:j + nbd_size, i:i + nbd_size, :][:, :, 2] = beta_csw[which]
            hist = desc.describe(nbd_gray)
            pred = model.predict(hist.reshape(1, -1))
            all_predictions.append(pred[0])
            # print(pred[0])
            x = random.randint(0, alpha_csw[pred[0]].shape[0] - nbd_size)
            y = random.randint(0, alpha_csw[pred[0]].shape[1] - nbd_size)
            colorized_lab[i:i + nbd_size, j:j + nbd_size, :][:, :, 1] = alpha_csw[pred[0]][x:x + nbd_size,
                                                                        y:y + nbd_size]
            colorized_lab[i:i + nbd_size, j:j + nbd_size, :][:, :, 2] = beta_csw[pred[0]][x:x + nbd_size,
                                                                        y:y + nbd_size]
            j += 1
        i += 1
    colorized_bgr = cv.cvtColor(colorized_lab, cv.COLOR_Lab2BGR)
    print(all_predictions)
    cv.imshow("Colorized", colorized_bgr)
    cv.waitKey(0)


imgcolor = np.array(cv.imread("F:\\IITD\\2022-1\\COL783-Digital Image "
                              "Analysis\\Assignments\\Assignment-1\\pictures\\part2\\img_2a.png"))
# cv.imshow("Color", imgcolor)
# cv.waitKey(0)
imggray = np.array(cv.imread("F:\\IITD\\2022-1\\COL783-Digital Image "
                             "Analysis\\Assignments\\Assignment-1\\pictures\\part2\\img_2b.png", 0))
# cv.imshow("Gray", imggray)
# cv.waitKey(0)
match_color_swatch(imgcolor, imggray, 2, nbd_size=2)
# iterations = 10
# for ite in range(0, iterations):
#     print(f"iteration = {ite}")
#     start = time.time()
#     colorized_lab = match_color(imgcolor, imggray)
#     end = time.time()
#     colorized_bgr = cv.cvtColor(colorized_lab, cv.COLOR_Lab2BGR)
#     filename = f"F:\\IITD\\2022-1\\COL783-Digital Image Analysis\\Assignments\\Assignment-1\\pictures\\result\\img_4{ite}.png "
#     cv.imwrite(filename, colorized_bgr)
#     print(end - start)
