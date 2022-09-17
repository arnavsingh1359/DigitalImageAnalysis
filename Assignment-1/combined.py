import math
import random
import cv2 as cv
import numpy as np
from scipy import stats
from scipy.signal import argrelextrema
from scipy.signal import medfilt
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
from skimage import exposure
from sklearn.naive_bayes import GaussianNB

from localbinarypatterns import LocalBinaryPatterns


def invert(img):
    h, w = img.shape
    img2 = img.copy()
    for i in range(h):
        for j in range(w):
            if img[i][j] == 255:
                img2[i][j] = 0
            else:
                img2[i][j] = 255
    return img2


def detected_skin(image, alpha_a=6.5, beta_b=12, sat_min=15, sat_max=170, hue_max=17.1):
    hue, sat, value = np.array_split(cv.cvtColor(image, cv.COLOR_BGR2HSV), 3, axis=2)
    Y, Cr, Cb = np.array_split(cv.cvtColor(image, cv.COLOR_BGR2YCR_CB), 3, axis=2)
    hue = hue[:, :, 0]
    sat = sat[:, :, 0]
    value = value[:, :, 0]
    Y = Y[:, :, 0]
    Cr = Cr[:, :, 0]
    Cb = Cb[:, :, 0]
    result = np.zeros(hue.shape)
    p = 0
    while p < result.shape[0]:
        q = 0
        while q < result.shape[1]:
            if sat_min <= sat[p, q] <= sat_max and hue[p, q] <= hue_max:
                if 135 <= Cr[p, q] <= 180 and 85 <= Cb[p, q] <= 135:
                    result[p, q] = 1
            q += 1
        p += 1
    _h, _w = image.shape[:2]
    _kernel = cv.getStructuringElement(cv.MORPH_RECT, (int(_h / 48), int(_w / 48)))
    result = cv.morphologyEx(result, cv.MORPH_CLOSE, _kernel)
    return result


def detect_faces(image, scale_factor=1.01, min_neighbor=6):
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, scale_factor, min_neighbor)
    return faces


def WLS_filter(img, lambda_=0.1, alpha=1.2, eps=10 ^ -4):
    image = img / 255.0

    s = image.shape

    k = np.prod(s)

    dy = np.diff(image, 1, 0)
    dy = -lambda_ / (np.absolute(dy) ** alpha + eps)
    last_y = np.zeros((s[1],))
    dy = np.vstack((dy, last_y))
    dy = dy.flatten('F')

    dx = np.diff(image, 1, 1)
    dx = -lambda_ / (np.absolute(dx) ** alpha + eps)
    last_x = np.zeros((s[0],))
    dx = np.hstack((dx, last_x[:, np.newaxis]))
    dx = dx.flatten('F')

    a = spdiags(np.vstack((dx, dy)), [-s[0], -1], k, k, format="csr")

    d = 1 - (dx + np.roll(dx, s[0]) + dy + np.roll(dy, s[1]))
    a = a + a.T + spdiags(d, 0, k, k)
    _out = spsolve(a, image.flatten('F')).reshape(s[::-1])

    base = np.rollaxis(_out, 1) * 255
    out = np.clip(base, 0, 255)
    detail = image - out
    np.clip(detail, 0, 255, out=detail)
    return out, detail


def eacp(F, L, W=None, lambda_=0.2, alpha=0.3, eps=1e-4):
    if W is None:
        W = np.ones(F.shape)
    f = F.flatten('F')
    w = W.flatten('F')
    s = L.shape

    k = np.prod(s)
    # L_i - L_j along y axis
    dy = np.diff(L, 1, 0)
    dy = -lambda_ / (np.absolute(dy) ** alpha + eps)
    dy = np.vstack((dy, np.zeros(s[1], )))
    dy = dy.flatten('F')
    # L_i - L_j along x axis
    dx = np.diff(L, 1, 1)
    dx = -lambda_ / (np.absolute(dx) ** alpha + eps)
    dx = np.hstack((dx, np.zeros(s[0], )[:, np.newaxis]))
    dx = dx.flatten('F')
    # A case: j \in N_4(i)  (neighbors of diagonal line)
    a = spdiags(np.vstack((dx, dy)), [-s[0], -1], k, k)
    # A case: i=j   (diagonal line)
    d = w - (dx + np.roll(dx, s[0]) + dy + np.roll(dy, 1))
    a = a + a.T + spdiags(d, 0, k, k)  # A: put together
    f = spsolve(a, w * f).reshape(s[::-1])  # slove Af  =  b =w*g and restore 2d
    A = np.rollaxis(f, 1)
    # A = np.clip( _out*255.0, 0, 255).astype('uint8')
    return A


def is_bimodal(x, y):
    maxima = argrelextrema(y, np.greater)[0]
    minima = argrelextrema(y, np.less)[0]
    if len(maxima) >= 2:
        i = 0
        while i < len(maxima) - 1 and len(minima) > 0:
            d = y[maxima[i]]
            b = y[maxima[i + 1]]
            m = y[minima[i]]
            if m < 0.8 * d and m < 0.8 * b:
                return int(x[maxima[i]]), int(x[maxima[i + 1]]), int(x[minima[i]])
            i += 1

    return None, None, None


def mask_skin(img, mask):
    for i in range(len(mask[0])):
        for j in range(len(mask[1])):
            if mask[i][j] == 0:
                img[i, j] = 0
    return img


def percentile(image):
    data = np.zeros(256)
    for i in range(len(image[0])):
        for j in range(len(image[1])):
            data[int(image[i, j])] += 1

    tot_sum = 0
    total = int(np.prod(image.shape) * 0.75)
    cum = np.zeros(256)
    for i in range(len(data)):
        tot_sum += data[i]
        cum[i] = tot_sum
        if cum[i] >= total:
            return i


def all_zero(img):
    h, j = img.shape
    for i in range(h):
        for j in range(j):
            if img[i][j] != 0:
                return False
    return True


def face_correction(image, skin_mask, lambda_=0.2):
    lum, alpha, beta = np.array_split(cv.cvtColor(image, cv.COLOR_BGR2Lab), 3, axis=2)
    lum = lum[:, :, 0]
    alpha = alpha[:, :, 0]
    beta = beta[:, :, 0]
    I_out, detail = WLS_filter(lum)
    # face_detection
    face = detect_faces(image, 1.001, 5)
    # all faces
    faces = [lum[y:y + h, x:x + w] for (x, y, w, h) in face]
    W = np.zeros(lum.shape)
    A = np.ones(lum.shape, dtype="float")
    B = np.ones(lum.shape)
    # face_correction a)sidelight and b)exposure
    for index, (x, y, w, h) in enumerate(face):
        # sidelight correction
        skin_in_face = skin_mask[y:y + h, x:x + w]
        data, bins = exposure.histogram(lum[y:y + h, x:x + w], nbins=256)
        intensity = np.mgrid[0:256]
        kernel = stats.gaussian_kde(data)
        density = kernel(intensity)
        maxima = argrelextrema(density, np.greater)
        minima = argrelextrema(density, np.less)
        d, b, m = is_bimodal(intensity, density)
        # global W
        if d and b and m:
            f = (b - d) / (m - d)
            r = 0
            while r < w:
                s = 0
                while s < h:
                    if skin_in_face[r, s] and (lum[y + r, x + s] < m):
                        A[y + r, x + s] = f
                    s += 1
                r += 1
            miu = (lum[A == f]).mean()
            sig = 255 * 3
            W = np.exp(-(lum - miu) ** 2 / sig ** 2)
            W[...] = 1 - W[...]
            W[...] = 1
    are_all_zero = all_zero(W.copy())
    if are_all_zero:
        W = np.ones(lum.shape)
        A_after = eacp(A, lum, W, lambda_)
    else:
        A_after = eacp(A, lum, W, lambda_)
    I_out *= A_after
    I_out2 = I_out.copy()
    for index, (x, y, w, h) in enumerate(face):

        skin_in_face = skin_mask[y:y + h, x:x + w]
        temp_img = I_out[y:y + h, x:x + w]
        p = percentile(temp_img)
        if p < 120:
            f = (120 + p) / ((2 * p) + 1e-6)
            f = np.clip(f, 1 + 1e-6, 2 - 11e-6)
            B[y:y + h, x:x + w][skin_in_face > 0] = f
            B = eacp(B, lum)
            I_out = I_out2 * B

    final = I_out + detail
    final = np.stack((final, alpha, beta), axis=2)
    final = final.astype("uint8")
    final_bgr = cv.cvtColor(final, cv.COLOR_Lab2BGR)
    return final_bgr


def skyline(mask):
    h, w = mask.shape
    for i in range(w):
        raw = mask[:, i]
        after_median = medfilt(raw, 19)
        try:
            first_zero_index = np.where(after_median == 0)[0][0]
            first_one_index = np.where(after_median == 1)[0][0]
            if first_zero_index > 20:
                mask[first_one_index:first_zero_index, i] = 1
                mask[first_zero_index:, i] = 0
                mask[:first_one_index, i] = 0
        except:
            continue
    return mask


def check_blue(mask):
    b, g, r = np.array_split(mask, 3, axis=2)
    b = b[:, :, 0]
    g = g[:, :, 0]
    r = r[:, :, 0]
    IDEAL_SKY_BGR = (195, 165, 80)
    RANGE = (60, 65, 120)
    h, w = b.shape
    for i in range(h):
        for j in range(w):
            if not (abs(b[i][j] - IDEAL_SKY_BGR[0]) < RANGE[0] and abs(g[i][j] - IDEAL_SKY_BGR[1]) < RANGE[1] and abs(
                    r[i][j] - IDEAL_SKY_BGR[2]) < RANGE[2]):
                b[i][j] = 0
                g[i][j] = 0
                r[i][j] = 0
    return np.stack((b, g, r), axis=2)


def sky_mask(img):
    h, w, _ = img.shape
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = cv.blur(img_gray, (9, 3))
    cv.medianBlur(img_gray, 5)
    lap = cv.Laplacian(img_gray, cv.CV_8U)
    gradient_mask = (lap < 5.9).astype(np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 3))
    mask = cv.morphologyEx(gradient_mask, cv.MORPH_ERODE, kernel)
    mask = skyline(mask)
    after_img = cv.bitwise_and(img, img, mask=mask)
    after_img = check_blue(after_img)
    return after_img


"""Part-2"""


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
    swatches = make_swatches(color_image, gray_image, n_swatches)
    lum_csw = []
    alpha_csw = []
    beta_csw = []
    for swatch in swatches:
        cx, cy, cw, ch = swatch[0]
        gx, gy, gw, gh = swatch[1]
        colorized_swatch = match_color(color_image[cy:cy + ch, cx:cx + cw], gray_image[gy:gy + gh, gx:gx + gw],
                                       n_samples, window_size)
        colorized_lab[gy:gy + gh, gx:gx + gw, :] = colorized_swatch
        lum_csw.append(colorized_swatch[:, :, 0])
        alpha_csw.append(colorized_swatch[:, :, 1])
        beta_csw.append(colorized_swatch[:, :, 2])

    desc = LocalBinaryPatterns(24, 8)
    data = []
    labels = []

    for ind, col_swatch in enumerate(lum_csw):
        index = 0
        while index < 10:
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
            nbd_gray = colorized_lab[i:i + nbd_size, j:j + nbd_size, :][:, :, 0]
            hist = desc.describe(nbd_gray)
            pred = model.predict(hist.reshape(1, -1))
            all_predictions.append(pred[0])
            x = random.randint(0, alpha_csw[pred[0]].shape[0] - nbd_size)
            y = random.randint(0, alpha_csw[pred[0]].shape[1] - nbd_size)
            colorized_lab[i:i + nbd_size, j:j + nbd_size, :][:, :, 1] = alpha_csw[pred[0]][x:x + nbd_size,
                                                                        y:y + nbd_size]
            colorized_lab[i:i + nbd_size, j:j + nbd_size, :][:, :, 2] = beta_csw[pred[0]][x:x + nbd_size,
                                                                        y:y + nbd_size]
            j += 1
        i += 1
    colorized_bgr = cv.cvtColor(colorized_lab, cv.COLOR_Lab2BGR)
    cv.imshow("Colorized", colorized_bgr)
    cv.waitKey(0)


def skyline(mask):
    h, w = mask.shape
    for i in range(w):
        raw = mask[:, i]
        after_median = medfilt(raw, 19)
        try:
            first_zero_index = np.where(after_median == 0)[0][0]
            first_one_index = np.where(after_median == 1)[0][0]
            if first_zero_index > 20:
                mask[first_one_index:first_zero_index, i] = 1
                mask[first_zero_index:, i] = 0
                mask[:first_one_index, i] = 0
        except:
            continue
    return mask


def check_blue(mask):
    b, g, r = np.array_split(mask, 3, axis=2)
    b = b[:, :, 0]
    g = g[:, :, 0]
    r = r[:, :, 0]
    IDEAL_SKY_BGR = (195, 165, 80)
    RANGE = (60, 65, 120)
    h, w = b.shape
    for i in range(h):
        for j in range(w):
            if not (abs(b[i][j] - IDEAL_SKY_BGR[0]) < RANGE[0] and abs(g[i][j] - IDEAL_SKY_BGR[1]) < RANGE[1] and abs(
                    r[i][j] - IDEAL_SKY_BGR[2]) < RANGE[2]):
                b[i][j] = 0
                g[i][j] = 0
                r[i][j] = 0
    return np.stack((b, g, r), axis=2)


def sky_mask(img):
    h, w, _ = img.shape

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    img_gray = cv.blur(img_gray, (9, 3))
    cv.medianBlur(img_gray, 5)
    lap = cv.Laplacian(img_gray, cv.CV_8U)
    gradient_mask = (lap < 5.9).astype(np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (9, 3))
    mask = cv.morphologyEx(gradient_mask, cv.MORPH_ERODE, kernel)
    mask = skyline(mask)
    after_img = cv.bitwise_and(img, img, mask=mask)
    after_img = check_blue(after_img)
    return after_img


"""Part-3"""


def get_w_spatial(n, m):
    xc = n / 2
    yc = m / 2
    maxdE = math.sqrt(xc ** 2 + yc ** 2)

    xy = np.empty([n, m, 2])
    for i in range(n):
        xy[i, :, 0] = i
    for j in range(m):
        xy[:, j, 1] = j

    temp = 1 - (np.sqrt((xy[..., 0] - xc) ** 2 + (xy[..., 1] - yc) ** 2) / maxdE) ** 2

    return temp


def shad_saliency_enhance(energy_map, L, I_bgr):
    dark_mask = (L < 50) & (np.maximum.reduce([I_bgr[..., 0], I_bgr[..., 1], I_bgr[..., 2]])
                            - np.minimum.reduce([I_bgr[..., 0], I_bgr[..., 1], I_bgr[..., 2]]) > 5)
    dark = np.where(dark_mask, L, 0)
    bright = L[~dark_mask]
    dark_smoothed = 255 * (WLS_filter(dark, 10, 10)[0]).reshape(dark.shape)
    print(np.percentile(bright, 35))
    print(np.percentile(dark_smoothed, 95))
    f_sal = min(2.0, 1.0 * np.percentile(bright, 35) / np.percentile(dark_smoothed, 95))

    b, detail = WLS_filter(L, 1000, 1)
    b *= 255
    detail *= 255
    cv.imshow("b", b)
    b_new = f_sal * energy_map * b + (1 - energy_map) * b
    cv.imshow("b new", b_new)
    cv.waitKey(0)
    return b_new + detail


def enhance_em(bgr_image, em):
    lum = cv.cvtColor(bgr_image, cv.COLOR_BGR2Lab)[..., 0]
    dark_mask = (lum < 50) & (np.maximum.reduce([bgr_image[..., 0], bgr_image[..., 1], bgr_image[..., 2]])
                              - np.minimum.reduce([bgr_image[..., 0], bgr_image[..., 1], bgr_image[..., 2]]) > 5)
    dark = np.where(dark_mask, lum, 0)
    mask = detected_skin(bgr_image).astype("float")

    w_spatial = get_w_spatial(lum.shape[0], lum.shape[1])
    em_res = w_spatial * ((em + 10 * mask + 10 * dark) / (em + 10 * mask + 10 * dark).max())
    em_res = eacp(em_res, lum)
    return em_res


def ss_enhance(bgr_image):
    i_lab = cv.cvtColor(bgr_image, cv.COLOR_BGR2Lab)
    saliency = cv.saliency.StaticSaliencyFineGrained_create()
    success, _em = saliency.computeSaliency(bgr_image)
    energy_map = enhance_em(bgr_image, _em)
    energy_map = (energy_map - energy_map.min()) / (energy_map.max() - energy_map.min())
    l_new = shad_saliency_enhance(energy_map, i_lab[..., 0], bgr_image)
    i_lab[..., 0] = l_new
    i_res_bgr = cv.cvtColor(i_lab, cv.COLOR_Lab2BGR)
    return i_res_bgr, energy_map


def run_all_corrections():
    image_bgr = cv.imread("Pictures\\part3\\img_0.png")
    # image_gray = cv.imread("Path\\to\\image")
    """part-1"""
    skin_mask = detected_skin(image_bgr)
    face_corrected = face_correction(image_bgr, skin_mask, 1.2)
    """part-2"""
    # skymask = sky_mask(image_bgr)

    """part-3"""
    image_bgr_enhanced, salience_map = ss_enhance(image_bgr)

    cv.imshow("enhanced", image_bgr_enhanced)
    cv.waitKey(0)


run_all_corrections()
