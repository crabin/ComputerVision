import numpy as np
import cv2
from matplotlib import pyplot as plt

# 初始化
def find_neighbors(x, y, labels):
    neighbors = []
    for i in range(x-1, x+2):
        for j in range(y-1, y+2):
            if i == x and j == y:
                continue  # 跳过自己
            if 0 <= i < labels.shape[0] and 0 <= j < labels.shape[1]:
                neighbors.append(labels[i, j])
    return neighbors

def connected_component_labeling(binary_image):
    labels = np.zeros(binary_image.shape, dtype=int)
    current_label = 1
    # 扫描每个像素
    for i in range(binary_image.shape[0]):
        for j in range(binary_image.shape[1]):
            if binary_image[i, j] == 255:  # 检查是否是白色像素
                neighbors = find_neighbors(i, j, labels)
                # 过滤掉0（未标记的部分）
                labeled_neighbors = [n for n in neighbors if n > 0]
                if len(labeled_neighbors) > 0:
                    labels[i, j] = min(labeled_neighbors)  # 使用最小标签
                else:
                    labels[i, j] = current_label  # 分配新标签
                    current_label += 1
    return labels

# 读取图像
img = cv2.imread('image/03.png', 0)

_, binary_image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# _, binary_image = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

# 应用连通分量标记
labeled_image = connected_component_labeling(binary_image)

# 绘制标记后的图像
plt.imshow(labeled_image, cmap='nipy_spectral')
plt.colorbar()
plt.title("Labeled Image with 8-connectivity")
plt.show()
