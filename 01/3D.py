import cv2
import numpy as np

# 步骤 1: 读取图像
image_path = "D:\WorkSpace\class\computer vision\image\\2D2.png"  # 图片路径
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# 步骤 2: 转换为灰度图，用于亮度分析
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 亮度: 计算灰度图的平均像素值
brightness = np.mean(gray)
print(f"图像的平均亮度: {brightness}")

# 纹理提取: 使用拉普拉斯滤波器计算纹理
texture = cv2.Laplacian(gray, cv2.CV_64F)
texture_variance = np.var(texture)
print(f"图像的纹理方差: {texture_variance}")

# 边缘检测: 使用Canny边缘检测提取轮廓
edges = cv2.Canny(gray, 100, 200)

# 模拟运动检测: 如果是视频，可以使用光流法分析
# 这里是静态图像，无法进行运动检测，暂时跳过

# 步骤 3: 显示结果
cv2.imshow('原始图像', image)
cv2.imshow('边缘检测（轮廓）', edges)

# 等待按键关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()

# 后续步骤:
# - 双眼立体视觉：需要两幅图像，使用 cv2.StereoBM_create 或 cv2.StereoSGBM_create 计算视差图
# - 运动检测：对于视频，可以使用 cv2.calcOpticalFlowPyrLK
