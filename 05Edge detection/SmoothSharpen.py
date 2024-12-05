import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('image/lenna.png', cv2.IMREAD_GRAYSCALE)

# 平滑操作（高斯滤波）
smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)

# 锐化操作（使用拉普拉斯算子）
laplacian = cv2.Laplacian(image, cv2.CV_64F)
sharpened_image = np.uint8(np.clip(image - laplacian, 0, 255))

# 锐化操作（使用简单的锐化滤波器）
sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
sharpened_image_kernel = cv2.filter2D(image, -1, sharpen_kernel)

# 显示结果
plt.figure(figsize=(12, 12))

# 原始图像
plt.subplot(1, 4, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

# 平滑图像（高斯滤波）
plt.subplot(1, 4, 2)
plt.title('Smoothed Image (Gaussian)')
plt.imshow(smoothed_image, cmap='gray')

# 锐化图像（拉普拉斯算子）
plt.subplot(1, 4, 3)
plt.title('Sharpened Image (Laplacian)')
plt.imshow(sharpened_image, cmap='gray')

# 锐化图像（简单锐化滤波器）
plt.subplot(1, 4, 4)
plt.title('Sharpened Image (Kernel)')
plt.imshow(sharpened_image_kernel, cmap='gray')

plt.tight_layout()
plt.show()
