import math

import cv2 as cv
import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve


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


def WLS_filter(img, lambda_=0.1, alpha=1.2, eps=1e-4):
    image = img / 255.0
    s = image.shape
    k = np.prod(s)
    # print(s, k)
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
    # print(dx.shape)

    a = spdiags(np.vstack((dx, dy)), [-s[0], -1], k, k, format="csr")

    d = 1 - (dx + np.roll(dx, s[0]) + dy + np.roll(dy, s[1]))
    a = a + a.T + spdiags(d, 0, k, k, format="csr")
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
    return A


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


def shad_saliency_enhance(energy_map, L, I_bgr):
    dark_mask = (L < 50) & (np.maximum.reduce([I_bgr[..., 0], I_bgr[..., 1], I_bgr[..., 2]])
                            - np.minimum.reduce([I_bgr[..., 0], I_bgr[..., 1], I_bgr[..., 2]]) > 5)
    dark = np.where(dark_mask, L, 0)
    bright = L[~dark_mask]
    dark_smoothed = 255 * (WLS_filter(dark, 1, 1.5)[1]).reshape(dark.shape)
    print(np.percentile(dark_smoothed, 95))
    print(np.percentile(bright, 35))
    f_sal = min(2.0, 1.0 * np.percentile(bright, 35) / np.percentile(dark_smoothed, 95))

    b, detail = WLS_filter(L, 100, 12)
    b *= 255
    detail *= 255
    cv.imshow("b", b)
    # plt.imshow(dark_smoothed)
    # plt.colorbar()
    # plt.show()
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
    em_res = w_spatial * ((em + 100 * mask + 100 * dark) / (em + 100 * mask + 100 * dark).max())
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


image_bgr = cv.imread("Pictures\\part3\\img_0.png")

image_bgr_result, energy_map = ss_enhance(image_bgr)

# cv.imwrite("Result image.png", image_bgr_result)
# plt.imshow(energy_map)
# plt.show()
cv.imshow("enhanced", image_bgr_result)
cv.waitKey(0)
