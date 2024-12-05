import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = 'image/03.png'

image = cv2.imread(image_path, 1)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 将图像二值化
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# 使用4-邻域（十字形）结构元素进行腐蚀
kernel_4 = np.array([[0, 1, 0],
                     [1, 1, 1],
                     [0, 1, 0]], dtype=np.uint8)
eroded_image_4 = cv2.erode(binary_image, kernel_4, iterations=1)


# 使用8-邻域（3x3方形）结构元素进行腐蚀
kernel_8 = np.ones((3, 3), np.uint8)
eroded_image_8 = cv2.erode(binary_image, kernel_8, iterations=2)

cv2.imshow('binary_image', binary_image)
cv2.waitKey(0)

plt.figure(figsize=(8, 8))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(eroded_image_4, cmap='gray')
plt.title('eroded_image_4')

plt.subplot(1, 3, 3)
plt.imshow(eroded_image_8, cmap='gray')
plt.title('eroded_image_8')

plt.show()