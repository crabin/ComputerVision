import cv2
import numpy as np
from matplotlib import pyplot as plt


# image_path = "image/gra.png" 
# image = cv2.imread(image_path, 0)


# plt.hist(image.ravel(), 256, [0, 256])
# plt.title('Histogram')
# plt.xlabel('grayscale')
# plt.ylabel('frequency')
# plt.show()

image_path2 = "image/gra2.png"

# image2 = cv2.imread(image_path2)
# # 分离 BGR 通道
# channels = cv2.split(image2)
# colors = ('b', 'g', 'r')
# channel_names = ('Blue', 'Green', 'Red')

# # 创建一个图像窗口
# plt.figure(figsize=(10, 6))

# # 对每个通道计算直方图并显示
# for i, (channel, color, name) in enumerate(zip(channels, colors, channel_names)):
#     plt.subplot(1, 3, i+1)
#     plt.hist(channel.ravel(), 256, [0, 256], color=color)
#     plt.title(f'{name} Channel Histogram')
#     plt.xlim([0, 256])
#     plt.xlabel('Pixel Intensity')
#     plt.ylabel('Frequency')

# # 显示结果
# plt.tight_layout()
# plt.show()

gray_image = cv2.imread(image_path2, 0)

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.hist(gray_image.ravel(), 256, [0, 256])
plt.title('Histogram2')
plt.xlabel('grayscale')
plt.ylabel('frequency')


# 调整亮度 - 增加亮度（可以修改常量来增加或减少亮度）
# brightness_offset = -40  # 可以调整的亮度值，正数增加亮度，负数减少亮度
# bright_image = cv2.add(gray_image, brightness_offset)

# # 确保像素值在有效范围（0 到 255）
# bright_image = np.clip(bright_image, 0, 255).astype(np.uint8)


# plt.subplot(1, 2, 2)
# plt.hist(bright_image.ravel(), 256, [0, 256])
# plt.title(f'Brightness Adjusted (Offset: {brightness_offset})')
# plt.xlabel('grayscale')
# plt.ylabel('frequency')
# plt.show()
# # 显示原图和亮度调整后的图片
# plt.figure(figsize=(10, 6))

# plt.subplot(1, 2, 1)
# plt.imshow(gray_image, cmap='gray')
# plt.title('Original Grayscale Image')
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.imshow(bright_image, cmap='gray')
# plt.title(f'Brightness Adjusted (Offset: {brightness_offset})')
# plt.axis('off')

# plt.tight_layout()
# plt.show()


# 设置对比度增益 (α) 和亮度偏移 (β)
alpha = 1  # 对比度，1.0 保持不变，>1 增加对比度，<1 降低对比度
beta = 40     # 亮度，通常设为0

# 调整对比度
contrast_image = cv2.convertScaleAbs(gray_image, alpha=alpha, beta=beta)

plt.subplot(1, 2, 2)
plt.hist(contrast_image.ravel(), 256, [0, 256])
plt.title(f'Contrast Adjusted (Alpha: {alpha}, Beta: {beta})')
plt.xlabel('grayscale')
plt.ylabel('frequency')
plt.tight_layout()
plt.show()


# 显示原图和对比度调整后的图片
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(contrast_image, cmap='gray')
plt.title(f'Contrast Adjusted (Alpha: {alpha}, Beta: {beta})')
plt.axis('off')

plt.tight_layout()
plt.show()