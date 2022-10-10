import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def dog(g_s, g_ks):
    return g_s - g_ks


def xdog(g_s, g_ks, p):
    return g_s - p*g_ks


def thresholder(u, e, phi):
    u_ = u/255
    result = np.ones_like(u)
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            if u_[i, j] < e:
                result[i, j] = 1 + np.tanh(phi * (u_[i, j] - e))

    return result*255


def plotter():
    x1 = np.linspace(-10, 10, 100)
    y1 = np.linspace(-10, 10, 100)
    x, y = np.meshgrid(x1, y1)
    s = 1
    k = 1.6
    ks = k * s
    g_s = np.exp(-(x * x + y * y) / (2 * (s ** 2)))
    g_s /= np.sum(g_s)
    g_ks = np.exp(-(x * x + y * y) / (2 * (ks ** 2)))
    g_ks /= np.sum(g_ks)
    d = dog(g_s, g_ks)
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_wireframe(x, y, d)
    plt.show()


# plotter()


image = cv.imread("Images\\Part 3\\man2.png", 0)
size = 5
sigma = 0.5
k = 200
ks = k * sigma
p = 0.98
blurred = cv.GaussianBlur(image,(3,3),sigma)
blurred_2 = cv.GaussianBlur(image,(3,3),ks)
# g_s = get_gaussian_kernel(size, sigma)
# g_ks = get_gaussian_kernel(size, ks)
# blurred = cv.filter2D(image, -1, g_s)
# blurred_2 = cv.filter2D(image, -1, g_ks)
# cv.imshow("image", image)
# cv.imshow("blurred", blurred)
# cv.imshow("blurred_2", blurred_2)
new = np.clip(blurred -p*blurred_2, 0, 255)
final = thresholder(new,0.32,10.3)
cv.imshow("new", new)
cv.imshow("final", final)
cv.waitKey(0)
