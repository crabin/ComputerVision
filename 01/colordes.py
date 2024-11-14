import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread('image/color.png')

# 将图像转换为RGB格式（因为OpenCV默认以BGR读取图像）
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 将图像数据转换为二维矩阵，每个像素点是一个样本
pixels = img_rgb.reshape(-1, 3)

# 使用KMeans将像素聚类为4个颜色
kmeans = KMeans(n_clusters=4)
kmeans.fit(pixels)

# 获取聚类中心的颜色值（即4个颜色）
centers = kmeans.cluster_centers_.astype('uint8')

# 根据聚类结果重新生成图像
new_pixels = centers[kmeans.labels_]
new_img = new_pixels.reshape(img_rgb.shape)

# 显示原始图像和新的颜色量化后的图像
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title('Original Image ')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(new_img)
plt.title('Quantized Image (4 colors)')
plt.axis('off')

plt.show()
