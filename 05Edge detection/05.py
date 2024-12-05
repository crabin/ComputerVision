import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2

image = cv2.imread('image/lenna.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('image', image)
# 转化为二值图
# _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
image = image.astype(np.float32) / 5.0
diff_image = np.zeros_like(image)
diff_image_x = np.zeros_like(image)
# 对每一行应用np.diff计算水平导数
for i in range(image.shape[0]):  # 遍历每一行
    diff_image[i, 1:] = np.diff(image[i, :])  # 计算当前行的水平差分，并保存到diff_image
    diff_image_x[i, 1:] = np.diff(image[i, :])

diff_image_y = np.zeros_like(image)
for j in range(image.shape[1]):  # 遍历每一列
    diff_image[1:, j] = np.diff(image[:, j])  # 计算当前列的垂直差分，并保存到diff_image
    diff_image_y[1:, j] = np.diff(image[:, j])


# sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # 水平导数
# sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # 垂直导数

# gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)  # 梯度幅值

# 二阶导数（使用拉普拉斯算子）
# laplacian = cv2.Laplacian(image, cv2.CV_64F)

plt.figure(figsize=(10, 7))
plt.subplot(1, 3, 1)
plt.title('Horizontal (1st Derivative)')
plt.imshow(diff_image_x, cmap='gray')
plt.subplot(1, 3, 2)
plt.title('Vertical (1st Derivative)')
plt.imshow(diff_image_y, cmap='gray')
plt.subplot(1, 3, 3)
plt.title('Gradient Magnitude (1st Derivative)')
plt.imshow(diff_image, cmap='gray')
plt.show()