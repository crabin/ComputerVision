
import matplotlib.pyplot as plt
import cv2
import numpy as np
 
img0 = cv2.imread('image/flaw.png')
img1 = cv2.resize(img0, fx = 0.5, fy = 0.5, dsize = None)  
# img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)             
height = img1.shape[0]                                    
width = img1.shape[1]                                      
print(img1.shape)
print(width, height)
cv2.namedWindow("W0")
cv2.imshow("W0", img1)
cv2.waitKey(delay = 0)
 
img8 = img1[0:-1:2, 0:-1:2]
img9 = img1[0:-1:4, 0:-1:4]
img10 = img1[0:-1:8, 0:-1:8]
img11 = img1[0:-1:16, 0:-1:16]
titles = ['origin image', '128*128', '64*64', '32*32', '16*16']
image = [img1, img8, img9, img10, img11]    

# 保存处理后的图像
cv2.imwrite('save/tonumber.png', img9)
cv2.imwrite('save/tonumber16.png', img11)

for j in range(5):
    plt.subplot(2, 3, j + 1)
    if j == 0:
        plt.imshow(image[j])
    else:
        plt.imshow(image[j], 'gray')
    plt.title(titles[j])
    plt.xticks([]), plt.yticks([])
plt.savefig('save/quan.png')
plt.show()