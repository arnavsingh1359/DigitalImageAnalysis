import numpy as np
from skimage import filters


def gaussian_pyramid(image, layers, downscale=2, sigma=None):
    if sigma is None:
        sigma = 2 * downscale / 6.0

    out = [image]
    reduced = image
    for i in range(layers - 1):
        smoothed = filters.gaussian(reduced, sigma)
        reduced = smoothed[::2, ::2]
        out.append(reduced)
    return out


def bi_inter(image, dimension):
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


def laplacian_pyramid(gp, tau):
    p = []
    for i in range(len(gp) - 1):
        g = gp[i]
        g_ = gp[i + 1]
        g_expand = bi_inter(g_, g.shape)
        lap = g - tau * g_expand
        p.append(lap)

    p.append(gp[-1])
    return p


def reconstruct_lap(lp):
    p_img = lp[-1]
    for i in range(len(lp) - 1):
        n_l = lp[-2 - i]
        e_p = bi_inter(p_img, n_l.shape)
        p_img = e_p + n_l
    return p_img
