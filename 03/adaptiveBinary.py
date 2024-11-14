import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = 'image/02.jpeg'

image = cv2.imread(image_path, 1)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# adaptivate Mean
adaptiveMean_binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

# adaptivate Gaussian
adaptiveGaussian_binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

plt.figure(figsize=(8, 8))
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.imshow(binary_image, cmap='gray')
plt.title('binary_image Image (v=127)')

plt.subplot(2, 2, 3)
plt.imshow(adaptiveMean_binary_image, cmap='gray')
plt.title('adaptiveMean_binary_image')

plt.subplot(2, 2, 4)
plt.imshow(adaptiveGaussian_binary_image, cmap='gray')
plt.title('adaptiveGaussian_binary_image')


plt.show()