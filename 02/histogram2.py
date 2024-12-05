import cv2
import numpy as np
from matplotlib import pyplot as plt

image_path = "image/gra5.png"

gray_image = cv2.imread(image_path)

plt.figure(figsize=(10, 6))


plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title(f'{image_path}')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.hist(gray_image.ravel(), 256, [0, 256])
plt.title('Histogram2')
plt.xlabel('grayscale')
plt.ylabel('frequency')


plt.tight_layout()
plt.show()