
import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('image/sailboat.jpg')


def motion_blur(image, degree=12, angle=45):
    image = np.array(image)
    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred


img_ = motion_blur(image, degree=30, angle=45)
blurred_image = cv2.GaussianBlur(image, (25, 25), 0)
 
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title('Image')
plt.imshow(image)
plt.subplot(1, 3, 2)
plt.title('GaussianBlur' )
plt.imshow(blurred_image)
plt.subplot(1, 3, 3)
plt.title('motion_blur' )
plt.imshow(img_)
plt.show()


