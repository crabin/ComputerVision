
import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = 'image/movie.png'

original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
original_image = cv2.resize(original_image, (256, 256))  # 调整图像大小

def add_gaussian_noise(image, mean=0, sigma=25):
    gaussian_noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 255)  # 保持像素值在0-255之间
    return noisy_image.astype(np.uint8)

# 添加块状噪声（模拟压缩/传输时发生的噪声）
def add_block_noise(image, quality=30):
    # 将图像编码为JPEG格式以引入压缩伪影
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded_img = cv2.imencode('.jpg', image, encode_param)
    decoded_img = cv2.imdecode(encoded_img, 1)
    return cv2.cvtColor(decoded_img, cv2.COLOR_BGR2GRAY)

# 创建带有不同噪声的图像
gaussian_noisy_image = add_gaussian_noise(original_image)
block_noisy_image = add_block_noise(original_image)

cv2.imwrite('image/gaussian_noisy_image.png', gaussian_noisy_image)

# 显示结果
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(original_image, cmap='gray')
axs[0].set_title("Original Image")
axs[0].axis("off")

axs[1].imshow(gaussian_noisy_image, cmap='gray')
axs[1].set_title("Gaussian Noise (Acquisition)")
axs[1].axis("off")

axs[2].imshow(block_noisy_image, cmap='gray')
axs[2].set_title("Block Noise (Compression/Transmission)")
axs[2].axis("off")

plt.show()
