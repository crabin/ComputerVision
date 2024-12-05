# erosion dilation

import cv2
import matplotlib.pyplot as plt
import numpy as np
image_path = 'image/04.png'
image = cv2.imread(image_path, 1)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 将图像二值化
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
# 使用8-邻域（3x3方形）结构元素进行腐蚀
kernel = np.ones((3, 3), np.uint8)
eroded_image_8 = cv2.erode(binary_image, kernel, iterations=1)

dilated_image = cv2.dilate(eroded_image_8, kernel, iterations=4)

plt.subplot(1, 3, 1)
plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image')

plt.subplot(1, 3, 2)
plt.imshow(eroded_image_8, cmap='gray')
plt.title('Eroded Image')

plt.subplot(1, 3, 3)
plt.imshow(dilated_image, cmap='gray')
plt.title('Opening Image')
plt.show()