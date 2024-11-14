import cv2
import numpy as np


img = cv2.imread('image/tonumber16.png', cv2.IMREAD_GRAYSCALE)


resized_img = cv2.resize(img, (16, 16), interpolation=cv2.INTER_NEAREST)


def pixel_to_value(pixel):
    if pixel < 50:
        return 5
    elif pixel < 100:
        return 3
    elif pixel < 150:
        return 2
    elif pixel < 200:
        return 1
    else:
        return 0

value_matrix = np.vectorize(pixel_to_value)(resized_img)

# 输出结果
print("数值矩阵表示：")
print(value_matrix)
