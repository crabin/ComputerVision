# python实现零均值归一化交叉相关
import cv2 as cv
import numpy as np

img = cv.imread('images/13.png').astype(np.float32)
H, W, C = img.shape
mi = np.mean(img)

temp = cv.imread('images/12.png').astype(np.float32)
Ht, Wt, Ct = temp.shape
mt = np.mean(temp)

i, j = -1, -1

v = -1

for y in range(H - Ht):
    for x in range(W - Wt):
        # _v = np.sum((img[y:y+Ht, x:x+Wt] - temp) ** 2)
        # _v = np.sum(np.abs(img[y:y+Ht, x:x+Wt] - temp))
        _v = np.sum(img[y:y+Ht, x:x+Wt] * temp)
        _v /= (np.sqrt(np.sum(img[y:y+Ht, x:x+Wt]**2)) * np.sqrt(np.sum(temp**2)))
        # _v = np.sum((img[y:y+Ht, x:x+Wt]-mi) * (temp-mt))
        # _v /= (np.sqrt(np.sum((img[y:y+Ht, x:x+Wt]-mi)**2)) * np.sqrt(np.sum((temp-mt)**2)))
        if _v < v or v == -1:
            v = _v
            i, j = x, y

out = img.copy()
# rectangle draw function in opencv
cv.rectangle(out, (j, i), (j+Wt, i+Ht), (0, 0, 255), 2)
out = out.astype(np.uint8)
cv.imshow('result', out)
cv.waitKey(0)
cv.destroyAllWindows()
