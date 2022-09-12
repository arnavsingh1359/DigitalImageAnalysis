import cv2 as cv
import cv2.saliency as saliency
import matplotlib.pyplot as plt

image = cv.imread("Pictures\\part3\\img_1.png")

saliency = saliency.StaticSaliencyFineGrained_create()
success, saliencyMap = saliency.computeSaliency(image)
threshMap = cv.threshold(saliencyMap.astype("uint8"), 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

plt.imshow(saliencyMap)
plt.colorbar()
plt.show()

# cv.imshow("Image", image)
# cv.imshow("Output", saliencyMap)
# cv.imshow("Thresh", threshMap)
# cv.waitKey(0)
