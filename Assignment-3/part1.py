import cv2 as cv
import numpy as np


def partial_x(img):
    result = img.copy().astype("int32")
    result = np.hstack((result[:, 1:], result[:, 0].reshape(result[:, 0].shape[0], -1))) - result[:, :]

    return result


def partial_y(img):
    result = img.copy().astype("int32")
    result = np.vstack((result[1:, :], result[0, :].reshape(-1, result[: 0].shape[1]))) - result[:, :]

    return result


def normalize(img):
    mi = np.min(img)
    ma = np.max(img)
    print(mi, ma)
    return ((img.astype("int32") - mi) / (ma - mi) * 255).astype("uint8")


def inpaint(img, epochs = 500):
    n = 0
    res = img.copy()
    while n < epochs:
        ix = partial_x(res)
        iy = partial_y(res)
        ixx = partial_x(ix)
        iyy = partial_y(iy)
        L = ixx + iyy
        # n = normalize(L)
        # cv.imshow("L", n)
        dL = np.zeros((img.shape[0], img.shape[1], 2))
        for i in range(L.shape[0]):
            for j in range(L.shape[1]):
                dL[i, j, 0] = L[(i + 1) % L.shape[0], j] - L[(i - 1) % L.shape[0], j]
                dL[i, j, 1] = L[i, (j + 1) % L.shape[1]] - L[i, (j - 1) % L.shape[1]]

        N = np.zeros_like(dL)
        N[:, :, 0] = -iy
        N[:, :, 1] = ix
        norm = np.sqrt((np.power(ix, 2) + np.power(iy, 2)))
        N[:, :, 0] = np.divide(N[:, :, 0], norm)
        N[:, :, 1] = np.divide(N[:, :, 1], norm)
        N = np.nan_to_num(N)
        beta = np.zeros(img.shape)
        for i in range(beta.shape[0]):
            for j in range(beta.shape[1]):
                beta[i, j] = dL[i, j, 0] * N[i, j, 0] + dL[i, j, 1] * N[i, j, 1]

        grad = np.zeros_like(img)
        for i in range(beta.shape[0]):
            for j in range(beta.shape[1]):
                if beta[i, j] > 0:
                    ixbm = np.minimum(0, ix[i, j] - ix[i, (j - 1) % ix.shape[1]])
                    ixfM = np.maximum(0, ix[i, (j + 1) % ix.shape[1]] - ix[i, j])
                    iybm = np.minimum(0, iy[i, j] - iy[(i - 1) % iy.shape[0], j])
                    iyfM = np.maximum(0, iy[(i + 1) % iy.shape[0], j] - iy[i, j])
                    grad[i, j] = np.sqrt((ixbm ** 2 + ixfM ** 2 + iybm ** 2 + iyfM ** 2))
                elif beta[i, j] < 0:
                    ixbM = np.maximum(0, ix[i, j] - ix[i, (j - 1) % ix.shape[1]])
                    ixfm = np.minimum(0, ix[i, (j + 1) % ix.shape[1]] - ix[i, j])
                    iybM = np.maximum(0, iy[i, j] - iy[(i - 1) % iy.shape[0], j])
                    iyfm = np.minimum(0, iy[(i + 1) % iy.shape[0], j] - iy[i, j])
                    grad[i, j] = np.sqrt((ixbM ** 2 + ixfm ** 2 + iybM ** 2 + iyfm ** 2))

        res +=

        if grad.all():
            print("In-painting done.")

        n = normalize(grad)
        cv.imshow("grad", n)


if __name__ == "__main__":
    image = cv.imread("Images/lena256.png", 0)
    inpaint(image.astype("int32"))
    cv.imshow("original", image)
    cv.waitKey(0)
    cv.destroyAllWindows()
