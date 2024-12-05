import cv2
import matplotlib.pyplot as plt
import numpy as np

width, height = 256, 256
# 创建一个对称的线性渐变
x = np.linspace(-1, 1, width)

# 使用高斯函数来生成两边变化慢，中间变化快的效果
gradient = np.exp(-x**2 * 10)  # 高斯函数的宽度可以调整，例如通过乘以10来使变化更明显

# 通过广播扩展到整个二维图像
gradient_image = np.tile(gradient, (height, 1))
# show(gradient_image)

plt.imshow(gradient_image, cmap='gray', origin='upper')
plt.axis('off')  # 不显示坐标轴
plt.show()

slice_index = gradient_image.shape[0] // 2
slice = gradient_image[slice_index,:]

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
