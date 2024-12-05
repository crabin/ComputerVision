import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载图像
image = cv2.imread('image/lenna.png', 0)

plt.imshow(image)
plt.axis('off')
plt.show()

def add_gaussian_noise(image, percentage=0.05, mean=0, std=25):
    # 确保图像是灰度图像
    if len(image.shape) != 2:
        raise ValueError("Input image must be a grayscale image.")

    noisy_image = image.copy()
    num_noisy_pixels = int(percentage * image.size)  # 计算要添加噪声的像素数量

    for _ in range(num_noisy_pixels):
        # 随机选择一个像素位置
        x = np.random.randint(0, image.shape[0])
        y = np.random.randint(0, image.shape[1])

        # 为灰度值添加高斯噪声
        noisy_image[x, y] = np.clip(noisy_image[x, y] + np.random.normal(mean, std), 0, 255)

    return noisy_image.astype(np.uint8)  # 返回无符号8位整数类型的图像

noisy_image = add_gaussian_noise(image, percentage=0.9, mean=0, std=25)
psnr_value = cv2.PSNR(image, noisy_image)
print(f"PSNR: {psnr_value:.2f} dB")

edge_image = cv2.bilateralFilter(noisy_image, 9, 75, 75)
edge_psnr_value = cv2.PSNR(image, edge_image)
print(f"edge_image PSNR: {edge_psnr_value:.2f} dB")

gaussian_filter = cv2.GaussianBlur(noisy_image, (5 ,5), 0)
gaussian_psnr_value = cv2.PSNR(image, gaussian_filter)
print(f"gaussian_filterPSNR: {gaussian_psnr_value:.2f} dB")

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title('noisy Image'+f"PSNR: {psnr_value:.2f} dB")
plt.imshow(noisy_image)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('gaussian Filter' +f"PSNR: {gaussian_psnr_value:.2f} dB")
plt.imshow(gaussian_filter)
plt.axis('off')


plt.subplot(1, 3, 3)
plt.title('Bilateral Filter' + f"PSNR: {edge_psnr_value:.2f} dB")
plt.imshow(edge_image)
plt.axis('off')

plt.show()