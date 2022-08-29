import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = np.array(cv.imread("lena256.png"))
gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


# fig, ax = plt.subplots(1, 2)
# fig.tight_layout()
#
# ax[0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
# ax[0].set_title("Original")

# ax[1].imshow(cv.cvtColor(gray_image, cv.COLOR_BGR2RGB))
# ax[1].set_title("Grayscale")

def bilinear_interpolation(image, dimension, type):
    height = image.shape[0]
    width = image.shape[1]
    scale_x = width / (dimension[1])
    scale_y = height / (dimension[0])

    if type == "color":
        new_image = np.zeros((dimension[0], dimension[1], image.shape[2]))
        for k in range(3):
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

                    a = image[y_int, x_int, k]
                    b = image[y_int, x_int + 1, k]
                    c = image[y_int + 1, x_int, k]
                    d = image[y_int + 1, x_int + 1, k]

                    pixel = a * (1 - x_diff) * (1 - y_diff) + b * x_diff * (1 - y_diff) + \
                            c * (1 - x_diff) * y_diff + d * x_diff * y_diff

                    new_image[i, j, k] = pixel

        return new_image.astype(np.uint8)
    elif type == "gray":
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

                print(pixel)

                new_image[i, j] = pixel

        return new_image.astype(np.uint8)


rescaled1 = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_LINEAR)
rescaled2 = bilinear_interpolation(img, (512, 512), "color")

diff = rescaled1 - rescaled2
cv.imshow("original", img)
cv.imshow("Rescaled1", rescaled1)
cv.imshow("Rescaled2", rescaled2)
cv.imshow("Difference", diff)
cv.waitKey(0)

plt.show()
