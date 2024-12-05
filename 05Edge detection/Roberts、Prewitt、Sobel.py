import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('image/lenna.png', cv2.IMREAD_GRAYSCALE)


def add_gaussian_noise(image, mean=0, sigma=25):
    row, col = image.shape
    gaussian = np.random.normal(mean, sigma, (row, col))
    noisy_image = np.array(image, dtype=float) + gaussian
    noisy_image = np.clip(noisy_image, 0, 255)  # 限制值在0到255之间
    return noisy_image.astype(np.uint8)

# 高斯噪声图像
# image = add_gaussian_noise(image)
image = np.float32(image)

# Roberts 算子
roberts_x = np.array([[1, 0], [0, -1]])
roberts_y = np.array([[0, 1], [-1, 0]])

roberts_edge_x = cv2.filter2D(image, -1, roberts_x)
roberts_edge_y = cv2.filter2D(image, -1, roberts_y)
roberts_edge = cv2.magnitude(roberts_edge_x, roberts_edge_y)

# Prewitt 算子
prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

prewitt_edge_x = cv2.filter2D(image, -1, prewitt_x)
prewitt_edge_y = cv2.filter2D(image, -1, prewitt_y)
prewitt_edge = cv2.magnitude(prewitt_edge_x, prewitt_edge_y)

# 计算二阶导数：对每个方向进行二阶卷积
# 二阶导数核
prewitt_x2 = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float32)
prewitt_y2 = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32)

# 计算二阶导数的结果
prewitt_x2_edge = cv2.filter2D(prewitt_edge_x, -1, prewitt_x2)
prewitt_y2_edge = cv2.filter2D(prewitt_edge_y, -1, prewitt_y2)
prewitt_edge_2nd = cv2.magnitude(prewitt_x2_edge, prewitt_y2_edge)

# Sobel 算子
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel_edge = cv2.magnitude(sobel_x, sobel_y)

# 绘制结果
plt.figure(figsize=(12, 12))

# 原始图像
plt.subplot(1, 4, 2)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

# Roberts 算子边缘
# plt.subplot(1, 4, 2)
# plt.title('Roberts Edge Detection')
# plt.imshow(roberts_edge, cmap='gray')

# Prewitt 算子边缘
plt.subplot(1, 4, 3)
plt.title('Prewitt Edge Detection')
plt.imshow(prewitt_edge, cmap='gray')

plt.subplot(1, 4, 4)
plt.title('prewitt_edge_2nd Detection')
plt.imshow(prewitt_edge_2nd, cmap='gray')


# Sobel 算子边缘
# plt.subplot(1, 4, 4)
# plt.title('Sobel Edge Detection')
# plt.imshow(sobel_edge, cmap='gray')

# plt.tight_layout()
plt.show()
