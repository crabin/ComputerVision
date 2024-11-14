import cv2

# 读取图像
image = cv2.imread('image/flaw.png')

# 获取图像的宽度和高度
width, height = image.shape[:2]

# 降采样（缩小图像）
small_image = cv2.resize(image, (width // 2, height // 2))

# 上采样（放大图像）
large_image = cv2.resize(image, (width * 2, height * 2))

# 显示图像
cv2.imshow('Original', image)
cv2.imshow('Small Image', small_image)
cv2.imshow('Large Image', large_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
