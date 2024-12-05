import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载图像
image = cv2.imread('image/turumai-132.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

# 1. 生成带噪声的图像
# 5% 高斯噪声函数
def add_gaussian_noise(image, percentage=0.9, mean=0, std=25):
    noisy_image = image.copy()
    num_noisy_pixels = int(percentage * image.size / 3)  # 计算要添加噪声的像素数量（每个像素包含RGB三个值）

    for _ in range(num_noisy_pixels):
        # 随机选择一个像素位置
        x = np.random.randint(0, image.shape[0])
        y = np.random.randint(0, image.shape[1])

        # 为每个通道添加高斯噪声
        noisy_image[x, y, 0] = np.clip(noisy_image[x, y, 0] + np.random.normal(mean, std), 0, 255)
        noisy_image[x, y, 1] = np.clip(noisy_image[x, y, 1] + np.random.normal(mean, std), 0, 255)
        noisy_image[x, y, 2] = np.clip(noisy_image[x, y, 2] + np.random.normal(mean, std), 0, 255)

    return noisy_image

# 1. 随机二值噪声
def add_large_binary_noise(image, percentage=0.0005, noise_size=3):
    noisy_image = image.copy()
    num_noisy_pixels = int(percentage * image.size / 3)  # 计算噪声区域数量

    for _ in range(num_noisy_pixels):
        # 随机选择一个像素位置
        x = np.random.randint(0, image.shape[0] - noise_size)
        y = np.random.randint(0, image.shape[1] - noise_size)
        
        # 随机选择黑色或白色噪声
        color = [0, 0, 0] if np.random.rand() > 0.5 else [255, 255, 255]

        # 将noise_size x noise_size区域设置为黑色或白色
        noisy_image[x:x + noise_size, y:y + noise_size] = color
    
    return noisy_image

# 生成带有随机二值噪声的图像
binary_noisy_image = add_large_binary_noise(image)

noisy_image = add_gaussian_noise(image, percentage=0.05)

# 2. 使用均值滤波
mean_filtered = cv2.blur(noisy_image, (5, 5))

# 3. 使用加权均值滤波（高斯滤波）
gaussian_filtered = cv2.GaussianBlur(noisy_image, (5, 5), 0)

median_filtered = cv2.medianBlur(noisy_image, 5)

# 显示结果
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(noisy_image)
axes[0].set_title("gaussia image")
axes[0].axis("off")

axes[1].imshow(mean_filtered)
axes[1].set_title("average filter")
axes[1].axis("off")

# axes[2].imshow(gaussian_filtered)
# axes[2].set_title("gaussian filter")
# axes[2].axis("off")

axes[2].imshow(median_filtered)
axes[2].set_title("median filter")
axes[2].axis("off")
plt.show()
