

# 导入cv2库，用于图像处理
import cv2 as cv
# 导入matplotlib.pyplot库，用于图像展示
import matplotlib.pyplot as plt
# 导入numpy库，用于数值计算
import numpy as np

# 读取图像
img = cv.imread('images/01.png')


# 定义两个卷积核，用于腐蚀和膨胀操作
kernel1 = np.ones((6, 6), np.uint8)
kernel2 = np.ones((5, 5), np.uint8)

# Apply erosion to remove small noise
erosion = cv.erode(img, kernel1, iterations=1)
# erosion = cv.erode(erosion, kernel2, iterations=1)

# Apply dilation to restore the background
dilate = cv.dilate(erosion, kernel1, iterations=1)
dilate = cv.dilate(dilate, kernel2, iterations=1)
# dilate = cv.dilate(dilate, kernel2, iterations=1)
# # dilate = cv.dilate(dilate, kernel2, iterations=1)
# dilate = cv.dilate(dilate, kernel2, iterations=1)



# print(dilate)
# mask = np.where(dilate > 0, 255, 0).astype(np.uint8)

# result = np.copy(dilate)  # 复制图像
# result[mask == 255] = 255  # 非黑色区域改为白色


# 4 图像展示
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 8), dpi=100)
axes[0].imshow(img[:, :, ::-1])
axes[0].set_title("img")
axes[1].imshow(erosion[:, :, ::-1])
axes[1].set_title("erosion")
axes[2].imshow(dilate[:, :, ::-1])
axes[2].set_title("dilate")
plt.show()
