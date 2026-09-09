import cv2
import numpy as np
import matplotlib.pyplot as plt
# Precompute Gamma LUT and setup CLAHE outside the loops for performance
invGamma = 1.0 / 0.8
gamma_table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
clahe8 = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
clahe16 = cv2.createCLAHE(clipLimit=0.5 , tileGridSize=(16,16))

img = cv2.imread("test.png", cv2.IMREAD_GRAYSCALE)
img_resized = cv2.resize(img, (400,400), interpolation=cv2.INTER_AREA)
img_gamma = cv2.LUT(img_resized, gamma_table)
img_enhanced_8 = clahe8.apply(img_gamma)
img_enhanced_16 = clahe16.apply(img_gamma)

plt.figure()
plt.subplot(1,3,1)
plt.title("original image")
plt.imshow(img_resized, cmap="gray")
plt.subplot(1,3,2)
plt.title("clahe 8")
plt.imshow(img_enhanced_8, cmap="gray")
plt.subplot(1,3,3)
plt.title("clahe 16")
plt.imshow(img_enhanced_16, cmap="gray")
plt.savefig("./show.png")

