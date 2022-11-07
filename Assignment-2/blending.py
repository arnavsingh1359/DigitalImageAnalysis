import cv2 as cv
import numpy as np
import image_pyramid


def fusion(img1, img2, mask, layers, tau):
    gp1 = image_pyramid.gaussian_pyramid(img1, layers)
    gp2 = image_pyramid.gaussian_pyramid(img2, layers)
    lp1 = image_pyramid.laplacian_pyramid(gp1, tau)
    lp2 = image_pyramid.laplacian_pyramid(gp2, tau)
    b_p = []
    masks = [mask]
    for p in range(layers - 1):
        mask = mask[::2, ::2]
        masks.append(mask)
    for i in range(layers):
        i1 = lp1[i].astype("float32")
        i2 = lp2[i].astype("float32")
        m = masks[i]
        temp = m * i1 + (1 - m) * i2
        b_p.append(temp)

    final_image = image_pyramid.reconstruct_lap(b_p)
    mi = final_image.min()
    m = final_image.max()

    final_image = 255 * (final_image - mi) / (m - mi)
    return final_image.astype("uint8")


if __name__ == "__main__":
    im1 = np.float32(cv.imread("Images\\Part 1\\burt_apple.png", 0))
    im2 = np.float32(cv.imread("Images\\Part 1\\burt_orange.png", 0))

    half = int(im1.shape[1] / 2)
    masks = []
    ma = np.hstack((np.ones((im1.shape[0], half), np.float32), np.zeros((im1.shape[0], half), np.float32)))
    masks.append(ma)
    laye = 7
    ta = 1
    out = fusion(im1, im2, ma, laye, ta)
    cv.imshow("Apple", im1.astype("uint8"))
    cv.imshow("Orange", im2.astype("uint8"))
    cv.imshow("Fusion", out)
    cv.waitKey(0)
    cv.destroyAllWindows()
