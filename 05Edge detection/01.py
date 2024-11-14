import cv2
import matplotlib.pyplot as plt
import numpy as np


image_path = 'image/01.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


slice_index = image.shape[0] // 2
slice = image[slice_index,:]

diff_slice = np.diff(slice)
diff_slice2 = np.diff(diff_slice)

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.plot(slice)
plt.title('Gray Value Variation along the Slice')
plt.xlabel('Position along the Slice')
plt.ylabel('Gray Value')

# 绘制微分后的灰度值变化曲线
plt.subplot(1, 3, 2)
plt.plot(diff_slice)
plt.title('Differentiated Gray Value Variation')
plt.xlabel('Position along the Slice')
plt.ylabel('Gray Value Change Rate')
plt.tight_layout()

plt.subplot(1, 3, 3)
plt.plot(diff_slice2)
plt.title('Differentiated2 Gray Value Variation')
plt.xlabel('Position along the Slice')
plt.ylabel('Gray Value Change Rate')
plt.tight_layout()
plt.show()
