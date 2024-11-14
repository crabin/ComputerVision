import numpy as np
import cv2
import matplotlib.pyplot as plt

# 定义递归标记函数
def label_component_recursive(x, y, label, labels, binary_image):
    # 如果超出边界，返回
    if x < 0 or x >= binary_image.shape[0] or y < 0 or y >= binary_image.shape[1]:
        return
    # 如果已经标记或者不是白色像素，返回
    if binary_image[x, y] == 0 or labels[x, y] > 0:
        return

    # 分配当前像素的标签
    labels[x, y] = label

    # 递归检查8个相邻的像素
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx != 0 or dy != 0:  # 跳过自己
                label_component_recursive(x + dx, y + dy, label, labels, binary_image)

# 递归连通分量标记
def connected_component_recursive(binary_image):
    labels = np.zeros(binary_image.shape, dtype=int)
    current_label = 1

    for i in range(binary_image.shape[0]):
        for j in range(binary_image.shape[1]):
            if binary_image[i, j] == 255 and labels[i, j] == 0:  # 未标记的白色像素
                label_component_recursive(i, j, current_label, labels, binary_image)
                current_label += 1
    return labels

# 读取图像
img = cv2.imread('image/03.png', cv2.IMREAD_GRAYSCALE)
_, binary_image = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

# 使用递归方法进行连通分量标记
labeled_image = connected_component_recursive(binary_image)

# 显示结果
plt.imshow(labeled_image, cmap='nipy_spectral')
plt.colorbar()
plt.title("Labeled Image with Recursive 8-connectivity")
plt.show()
