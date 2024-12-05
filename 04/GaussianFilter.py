
import cv2
import matplotlib.pyplot as plt
import numpy as np

# image_path = 'image/01.png'
image_path = 'image/gaussian_noisy_image.png'

original_image = cv2.imread(image_path, 1)
# original_image = cv2.resize(original_image, (256, 256))  # 调整图像大小

dst_image = original_image.copy()


def get_gaussian_kernel(size, sigma):
    # 创建高斯核
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    for x in range(-center, center + 1):
        for y in range(-center, center + 1):
            kernel[x + center, y + center] = (
                (1 / (2 * np.pi * sigma ** 2)) *
                np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            )
    # 归一化高斯核
    kernel /= np.sum(kernel)
    return kernel

def gaussian_blur(src_image, size, sigma):
    img = src_image.copy()
    dst_image = np.zeros_like(img, dtype=np.float32)  # 创建目标图像

    # 计算高斯核
    gaus = get_gaussian_kernel(size, sigma)
    center = size // 2

    # 扩展边界
    img_padded = cv2.copyMakeBorder(img, center, center, center, center, cv2.BORDER_REPLICATE)

    for i in range(center, img_padded.shape[0] - center):
        for j in range(center, img_padded.shape[1] - center):
            for x in range(-center, center + 1):
                for y in range(-center, center + 1):
                    # 卷积权重改变像素
                    dst_image[i - center, j - center] += img_padded[i + x, j + y] * gaus[x + center, y + center]

    dst_image = np.clip(dst_image, 0, 255).astype(np.uint8)
    # 保存结果图像
    # cv2.imwrite("../img-1and2/GaussianBlur.jpeg", dst_image)
    return dst_image

    shape = image.shape
    if filter_type == 'gaussian':
        # 高斯滤波，适合处理高斯噪声
        print('高斯滤波')
        for i in range(shape[0]):
            for j in range(shape[1]):
                dst_image[i, j, 0] = compute_pixel_value(image, i, j, ksize, 0)
                dst_image[i, j, 1] = compute_pixel_value(image, i, j, ksize, 1)
                dst_image[i, j, 2] = compute_pixel_value(image, i, j, ksize, 2)
        
        return dst_image
    elif filter_type == 'median':
        # 中值滤波，适合处理椒盐噪声
        return cv2.medianBlur(image, 5)
    else:
        return image

# denoise = cv2.GaussianBlur(original_image, ksize=3)
denoise = gaussian_blur(original_image, 5, 1.5)

# 显示结果
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].imshow(original_image, cmap='gray')
axs[0].set_title("Original Image")
axs[0].axis("off")

axs[1].imshow(denoise, cmap='gray')
axs[1].set_title("Gaussian Noise Removed")
axs[1].axis("off")
plt.show()
