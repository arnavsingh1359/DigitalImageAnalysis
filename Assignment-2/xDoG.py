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
    x1 = np.linspace(-1, 1, 25)
    y1 = np.linspace(-1, 1, 25)
    x, y = np.meshgrid(x1, y1)
    s = 5
    k = 1.6
    ks = k * s
    tau = 0
    g_s = 1 / (np.sqrt(2 * np.pi * (s ** 2))) * (np.exp(-(x * x + y * y) / 2 * (s ** 2)))
    g_ks = 1 / (np.sqrt(2 * np.pi * (ks ** 2))) * (np.exp(-(x * x + y * y) / 2 * (ks ** 2)))
    d = dog(g_s, g_ks)
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_wireframe(x, y, d)
    plt.show()


def get_gaussian_kernel(kernel_size, sigma):
    x1 = np.linspace(-1, 1, kernel_size)
    y1 = np.linspace(-1, 1, kernel_size)
    x, y = np.meshgrid(x1, y1)
    kernel = 1 / (np.sqrt(2 * np.pi * (sigma ** 2))) * (np.exp(-(x * x + y * y) / 2 * (sigma ** 2)))
    return kernel


image = cv.imread("Images\\Part 3\\man2.png", 0)
size = 7
s = 5
k = 1.6
ks = k * s
p = 18
g_s = get_gaussian_kernel(size, s)
g_ks = get_gaussian_kernel(size, ks)
d = dog(g_s, g_ks)
blurred = cv.filter2D(image, -1, g_s)
scaled = p * cv.filter2D(image, -1, d)
cv.imshow("image", image)
cv.imshow("blurred", blurred)
cv.imshow("scaled", scaled)
cv.waitKey(0)
