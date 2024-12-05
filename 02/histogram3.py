import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图像
img = cv2.imread( "image/gra2.png",0)  # 替换为实际的图像路径


def stretch_histogram_preserve_counts(img, low_in, high_in, low_out, high_out):

    normalized = np.clip((img - low_in) / (high_in - low_in), 0, 1)

    stretched_img = normalized * (high_out - low_out) + low_out
    return np.uint8(stretched_img)

# Stretch the values from 0-30 to 0-255
def apply_gamma_transformation(img, gamma):
    # Normalize the image to range [0, 1]
    normalized_img = img / 255.0
    # Apply the gamma correction
    gamma_corrected = np.power(normalized_img, gamma)
    # Scale back to range [0, 255]
    transformed_img = np.uint8(gamma_corrected * 255)
    return transformed_img


def log(c, img):
    c = 255 / np.log(1 + 255)

    # 计算对数变换后的输出值
    output = c * np.log(1 + img)
    output = np.uint8(output)
    return output

def log2(img):
    # 定义最大灰度值 z_m
    z_m = 255

    # 定义输入的灰度值 z
    z = img

    # 初始化输出灰度值 z_prime
    z_prime = np.zeros_like(z)

    # 计算公式1：对于 0 <= z <= z_m / 2
    mask1 = z <= z_m / 2
    z_prime[mask1] = 2 * z_m * (z[mask1] / z_m) ** 2

    # 计算公式2：对于 z_m / 2 <= z <= z_m
    mask2 = z > z_m / 2
    z_prime[mask2] = (z_m / 2) * (1 + np.sqrt((2 * z[mask2] - z_m) / z_m))

    return z_prime



# Set a gamma value for transformation (from the curve in the image it looks like gamma < 1)
# gamma_value = 2.2

# Apply the non-linear gamma transformation
# stretched_img =  apply_gamma_transformation(img, gamma_value)
stretched_img = log2(img)


# Display the original and stretched histograms
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Original histogram
axes[0].hist(img.ravel(), bins=256, range=[0, 256])
axes[0].set_title("Original Histogram")

# Stretched histogram
axes[1].hist(stretched_img.ravel(), bins=256, range=[0, 256])
axes[1].set_title("Stretched Histogram")

plt.show()

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')

plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(stretched_img, cmap='gray')
plt.title('Histogram2')
plt.xlabel('grayscale')
plt.ylabel('frequency')
plt.tight_layout()
plt.show()


