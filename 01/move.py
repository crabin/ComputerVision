import cv2
import numpy as np

# 读取图像
img = cv2.imread('image/move.png')

# 获取图像的行数和列数
rows = len(img)
cols = len(img[0])
# 定义原图像的三个点
p1 = np.array([[0, 0], [cols - 1, 0], [0, rows - 1]], dtype=np.float32)
# 定义变换后的三个点
p2 = np.array([[150, 0], [cols - 1, 0], [0, rows - 1]], dtype=np.float32)
# 获取仿射变换矩阵
M = cv2.getAffineTransform(p1, p2)

# 对图像进行仿射变换
dst = cv2.warpAffine(img, M, (cols, rows))
# 显示原图像和处理后的图像
cv2.imshow('img', img)
cv2.imshow('dst', dst)
# 保存处理后的图像
cv2.imwrite('save/move_output.png', dst)
# 等待按键
cv2.waitKey()
# 关闭所有窗口
cv2.destroyAllWindows()

