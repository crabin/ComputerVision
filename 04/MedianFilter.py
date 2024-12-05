
import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = 'image/01.png'

original_image = cv2.imread(image_path, 1)
# original_image = cv2.resize(original_image, (256, 256))  # 调整图像大小


median = cv2.medianBlur(original_image, 5)
# 显示结果
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].imshow(original_image, cmap='gray')
axs[0].set_title("Original Image")
axs[0].axis("off")

axs[1].imshow(median, cmap='gray')
axs[1].set_title("median Noise Removed")
axs[1].axis("off")
plt.show()
