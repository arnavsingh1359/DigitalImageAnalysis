import numpy as np
import matplotlib.pyplot as plt
from skimage import io, draw

img = np.array(io.imread("cheetah.png", as_gray=True))

x = 2
y = 3
fig, ax = plt.subplots(x, y, figsize=(6, 6))
fig.tight_layout()
c = 0.25
g = [0.25, 1, 2, 5, 10, 15]
for i in range(x):
    for j in range(y):
        gamma = g[y*i + j]
        new_img = c * np.power(img, gamma)
        maxi = (new_img.max())
        mini = (new_img.min())
        new_img = (new_img - mini) / (maxi - mini) * 255
        ax[(i, j)].imshow(new_img, cmap='gray')
        ax[(i, j)].set_title(f"gamma = {gamma}")

# io.imshow(img)
plt.show()
