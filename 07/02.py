

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('images/04.png')


# 定义两个卷积核，用于腐蚀和膨胀操作
kernel1 = np.ones((4, 4), np.uint8)

dilate = cv.dilate(img, kernel1, iterations=1)
dilate = cv.dilate(dilate, kernel1, iterations=1)

erosion = cv.erode(dilate, kernel1, iterations=1)
erosion = cv.erode(erosion, kernel1, iterations=1)
erosion = cv.erode(erosion, kernel1, iterations=1)
erosion = cv.erode(erosion, kernel1, iterations=1)

dilate = cv.dilate(erosion, kernel1, iterations=1)
dilate = cv.dilate(dilate, kernel1, iterations=1)
dilate = cv.dilate(dilate, kernel1, iterations=1)
dilate = cv.dilate(dilate, kernel1, iterations=1)
dilate = cv.dilate(dilate, kernel1, iterations=1)

erosion = cv.erode(dilate, kernel1, iterations=1)
erosion = cv.erode(erosion, kernel1, iterations=1)
erosion = cv.erode(erosion, kernel1, iterations=1)
erosion = cv.erode(erosion, kernel1, iterations=1)





# print(dilate)
# mask = np.where(dilate > 0, 255, 0).astype(np.uint8)

# result = np.copy(dilate)  # 复制图像
# result[mask == 255] = 255  # 非黑色区域改为白色


# 4 图像展示
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 8), dpi=100)
axes[0].imshow(img[:, :, ::-1])
axes[0].set_title("img")
axes[1].imshow(erosion[:, :, ::-1])
axes[1].set_title("last")
plt.show()
