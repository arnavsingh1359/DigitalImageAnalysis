import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def dog(g_s, g_ks):
    return g_s - g_ks


def xdog(g_s, g_ks, p):
    return (1 + p) * g_s - p * g_ks


def thresholder(u, e, phi):
    result = np.ones_like(u)
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            if u[i, j] < e:
                result[i, j] = 1 + np.tanh(phi * (u[i, j] - e))

    return result


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


def get_gaussian_kernel(kernel_size, sigma):
    x1 = np.linspace(-2 * sigma, 2 * sigma, kernel_size)
    y1 = np.linspace(-2 * sigma, 2 * sigma, kernel_size)
    x, y = np.meshgrid(x1, y1)
    kernel = np.exp(-(x * x + y * y) / (2 * (sigma ** 2)))
    su = np.sum(kernel)
    kernel /= su
    print(kernel)
    return kernel


image = cv.imread("Images\\Part 3\\man2.png", 0)
size = 5
sigma = 5
k = 1.6
ks = k * sigma
p = 18
g_s = get_gaussian_kernel(size, sigma)
g_ks = get_gaussian_kernel(size, ks)
d = dog(g_s, g_ks)
print(d)
blurred = cv.filter2D(image, -1, g_s)
scaled = cv.filter2D(image, -1, p * d)
s = xdog(g_s, g_ks, p)
i = cv.filter2D(image, -1, s)
cv.imshow("image", image)
cv.imshow("blurred", blurred)
cv.imshow("scaled", scaled)
new = np.clip(blurred + scaled, 0, 255)
cv.imshow("new", i)
cv.waitKey(0)
