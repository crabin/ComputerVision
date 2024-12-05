import cv2
import numpy as np

# 读取图像
image_path = "image/2D2.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 转为灰度图像

# 调整图像大小，确保点阵效果明显
height, width = image.shape
scale_factor = 0.1  # 缩放因子，调整点的密度
resized = cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)))

# 创建一个空白画布
output = np.ones_like(image) * 255  # 创建一个全白的画布

# 定义点的大小和绘制的密度
dot_size = 2
spacing = 10  # 点与点之间的间隔

# 遍历缩小后的图像，并在亮度较暗的位置绘制点
for y in range(0, resized.shape[0], 1):
    for x in range(0, resized.shape[1], 1):
        brightness = resized[y, x]
        if brightness < 128:  # 如果亮度较低（暗的地方），绘制点
            # 计算点的位置在原图上的映射
            cv2.circle(output, (int(x / scale_factor), int(y / scale_factor)), dot_size, (0, 0, 0), -1)

# 显示结果
cv2.imshow('原始图像', image)
cv2.imshow('点阵效果', output)

# 保存结果
cv2.imwrite("/mnt/data/dot_pattern_output.png", output)

# 等待按键并关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
