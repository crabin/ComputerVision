import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image

image = cv2.imread('images/kan.png')


# 获取图像的高度和宽度
height, width = image.shape[:2]
# 设置旋转角度 (45度)
theta = 45  # 旋转角度，单位为度
theta_rad = np.deg2rad(theta)  # 转换为弧度

# 计算旋转矩阵
cos_theta = np.cos(theta_rad)
sin_theta = np.sin(theta_rad)

# 旋转矩阵 (中心为图像的中心)
rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

# 图像的中心点
center = np.array([width // 2, height // 2])

# 创建新的空图像
euclidean_image = np.zeros_like(image)

# 对图像进行旋转操作
for y in range(height):
    for x in range(width):
        # 计算当前点相对于中心的偏移量
        offset = np.array([x, y]) - center
        
        # 应用旋转矩阵
        new_offset = np.dot(rotation_matrix, offset)  # 旋转后的偏移量
        
        # 计算新坐标
        new_x, new_y = new_offset + center
        
        # 使用最近邻插值，确保新坐标在图像范围内
        new_x, new_y = int(round(new_x)), int(round(new_y))
        
        # 如果新坐标在图像内，赋值
        if 0 <= new_x < width and 0 <= new_y < height:
            euclidean_image[new_y, new_x] = image[y, x]


# 放大因子（例如，放大2倍）
scale_x = 1.5
scale_y = 1.5

# 创建缩放矩阵
scaling_matrix = np.array([[scale_x, 0], [0, scale_y]])

# 创建一个空图像用于存储缩放后的结果
similarity_image = np.zeros_like(euclidean_image)

# 对旋转后的图像进行缩放
for y in range(height):
    for x in range(width):
        # 计算当前点相对于图像中心的偏移量
        offset = np.array([x, y]) - center
        
        # 应用缩放矩阵
        new_offset = np.dot(scaling_matrix, offset)  # 缩放后的偏移量
        
        # 计算新坐标
        new_x, new_y = new_offset + center
        
        # 使用最近邻插值，确保新坐标在图像范围内
        new_x, new_y = int(round(new_x)), int(round(new_y))
        
        # 如果新坐标在图像内，赋值
        if 0 <= new_x < width and 0 <= new_y < height:
            similarity_image[new_y, new_x] = euclidean_image[y, x]


# 旋转矩阵 R180
rotation_matrix = np.array([[-1, 0], [0, -1]])

# 图像的中心
center = np.array([width // 2, height // 2])

# 创建一个空图像来存储旋转后的结果
rotated_image = np.zeros_like(euclidean_image)

# 对图像中的每个像素进行旋转
for y in range(height):
    for x in range(width):
        # 计算当前像素相对于图像中心的偏移量
        offset = np.array([x, y]) - center
        
        # 应用旋转矩阵，计算旋转后的偏移量
        new_offset = np.dot(rotation_matrix, offset)
        
        # 计算新坐标
        new_x, new_y = new_offset + center
        
        # 使用最近邻插值，确保新坐标在图像范围内
        new_x, new_y = int(round(new_x)), int(round(new_y))
        
        # 如果新坐标在图像内，赋值
        if 0 <= new_x < width and 0 <= new_y < height:
            rotated_image[new_y, new_x] = euclidean_image[y, x]


fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs[0, 0].imshow(cv2.cvtColor(euclidean_image, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title('Euclidean Transformation')
axs[0, 1].imshow(cv2.cvtColor(similarity_image, cv2.COLOR_BGR2RGB))
axs[0, 1].set_title('Similarity 1.5 Transformation')
axs[0, 2].imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
axs[0, 2].set_title('Affine rotated 180 Transformation')

for ax in axs.flat:
    ax.axis('off')  
plt.tight_layout()
plt.show()