import cv2 
import numpy as np

image = cv2.imread('images/kan.png')

tx = 50
ty = 50
M = np.float32([[1,0,tx],[0,1,ty]])

shifted = cv2.warpAffine(image,M,(image.shape[1],image.shape[0]))

width = int(image.shape[1]*1.2)
height = int(image.shape[0] * 0.5)
d = (width, height)
resized = cv2.resize(image,d, interpolation = cv2.INTER_AREA)

rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)


# 定义原始坐标点和目标坐标点（例如，进行仿射变换）
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])  # 原图中的三个点
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])  # 目标图中的三个点

# 计算仿射变换矩阵
M = cv2.getAffineTransform(pts1, pts2)

# 应用仿射变换
result = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))


cv2.imshow('rotated_image',rotated_image)

cv2.waitKey(0)