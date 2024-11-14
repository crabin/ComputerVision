import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('image/03.png', cv2.IMREAD_GRAYSCALE)
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
kernel = np.ones((3, 3), np.uint8)
dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1), plt.imshow(binary_image, cmap='gray'), plt.title('原始二值图像')
plt.subplot(1, 2, 2), plt.imshow(dilated_image, cmap='gray'), plt.title('膨胀后的图像')
plt.show()