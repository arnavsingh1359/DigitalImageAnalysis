import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure

img = np.array(io.imread("night.png", as_gray=True))

x = 4
fig, ax = plt.subplots(2, x, figsize=(6, 6))
fig.tight_layout()

ax[(0, 0)].imshow(img, cmap="gray")
ax[(0, 0)].set_title("Original")
hist = exposure.histogram(img, normalize=True)
ax[(1, 0)].hist(hist[0], hist[1])
ax[(1, 0)].set_title("Original hist")

new_img = exposure.equalize_hist(img)
ax[(0, 1)].imshow(new_img, cmap="gray")
ax[(0, 1)].set_title("bins = 256")
hist = exposure.histogram(new_img, normalize=True)
ax[(1, 1)].hist(hist[0], hist[1])
ax[(1, 1)].set_title("256 hist")

new_img = exposure.equalize_hist(img, 200)
ax[(0, 2)].imshow(new_img, cmap="gray")
ax[(0, 2)].set_title("bins = 200")
hist = exposure.histogram(new_img, normalize=True)
ax[(1, 2)].hist(hist[0], hist[1])
ax[(1, 2)].set_title("200 hist")

new_img = exposure.equalize_hist(img, 50)
ax[(0, 3)].imshow(new_img, cmap="gray")
ax[(0, 3)].set_title("bins = 50")
hist = exposure.histogram(new_img, normalize=True)
ax[(1, 3)].hist(hist[0], hist[1])
ax[(1, 3)].set_title("50 hist")

plt.show()
