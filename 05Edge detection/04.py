import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2

# 创建一个图形和轴
# 创建一个空白图像
image = np.ones((200, 400, 3), dtype=np.uint8) * 255  # 创建白色背景图像

# 定义三个正方形的颜色和位置
color1 = (0, 255, 0)  # 绿色
color2 = (0, 255, 255)  # 黄色
color3 = (0, 255, 0)  # 绿色

# 绘制三个正方形
cv2.rectangle(image, (50, 50), (150, 150), color1, -1)  # 第一个绿色正方形
cv2.rectangle(image, (150, 50), (250, 150), color2, -1)  # 第二个黄色正方形
cv2.rectangle(image, (250, 50), (350, 150), color3, -1)  # 第三个绿色正方形

cv2.imshow("Image with Squares", image)

image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)  # 转换为灰度图像

image = cv2.GaussianBlur(image, (5, 5), 0)  # 高斯模糊

slice_index = image.shape[0] // 2
slice = image[slice_index,:]

diff_slice = np.diff(slice)
diff_slice2 = np.diff(diff_slice)

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.plot(slice)
plt.title('Gray Value Variation along the Slice')
plt.xlabel('Position along the Slice')
plt.ylabel('Gray Value')

# 绘制微分后的灰度值变化曲线
plt.subplot(1, 3, 2)
plt.plot(diff_slice)
plt.title('Differentiated Gray Value Variation')
plt.xlabel('Position along the Slice')
plt.ylabel('Gray Value Change Rate')
plt.tight_layout()

plt.subplot(1, 3, 3)
plt.plot(diff_slice2)
plt.title('Differentiated2 Gray Value Variation')
plt.xlabel('Position along the Slice')
plt.ylabel('Gray Value Change Rate')
plt.tight_layout()
plt.show()
