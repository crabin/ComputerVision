import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = 'image/lenna.png'

image = cv2.imread(image_path, 1)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def otsu_binarization(image):
    # 计算灰度直方图
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    # 总像素数
    total = image.shape[0] * image.shape[1]
    # 初始化类内和类间变量
    current_max = 0
    threshold = 0
    sum_total = 0
    for t in range(256):
        sum_total += t * hist[t]  # 灰度值的加权和
    sumB = 0  # 背景像素灰度值的加权和
    wB = 0  # 背景像素数量
    wF = 0  # 前景像素数量
    var_between = 0  # 类间方差
    # 遍历所有可能的阈值
    for t in range(256):
        # 更新背景像素数
        wB += hist[t]
        if wB == 0:  # 没有背景像素，跳过
            continue
        # 更新前景像素数
        wF = total - wB
        if wF == 0:  # 没有前景像素，跳过
            break
        # 更新背景灰度和
        sumB += t * hist[t]
        # 背景和前景的平均灰度
        meanB = sumB / wB
        meanF = (sum_total - sumB) / wF
        # 计算类间方差
        var_between = wB * wF * (meanB - meanF) ** 2
        # 如果当前类间方差大于之前的最大值，则更新阈值
        if var_between > current_max:
            current_max = var_between
            threshold = t
    return threshold


# _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                    cv2.THRESH_BINARY, 11, 2)
# 调用 Otsu's 二值化方法
threshold = otsu_binarization(gray_image)

 # 使用找到的最佳阈值对图像进行二值化
_, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)


print(f"最佳阈值: {threshold}")
plt.figure(figsize=(10, 6))
plt.subplot(1, 3, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('gray Image')
plt.subplot(1, 3, 2)
plt.hist(gray_image.ravel(), 256, [0, 256])
plt.subplot(1, 3, 3)
plt.imshow(binary_image, cmap='gray')
plt.title(f'binary Image threshold:{threshold}')
plt.show()
cv2.waitKey(0)
