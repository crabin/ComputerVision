
from PIL import Image
from matplotlib.pyplot import show
from numpy import average, dot, linalg
import numpy as np



def get_thum(image, size=(64, 64), greyscale=False):
    # 利用image对图像大小重新设置, Image.ANTIALIAS为高质量的
    image = image.resize(size, Image.LANCZOS)
    if greyscale:
        # 将图片转换为L模式，其为灰度图，其每个像素用8个bit表示
        image = image.convert('L')
    return image

def MSE(image1,image2):
    
    image1 = get_thum(image1)
    image2 = get_thum(image2)
    image1 = np.array(image1)
    image2 = np.array(image2)
    mse = np.mean( (image1 - image2) ** 2 )
    return mse


def image_similarity_voctors_via_numpy(image1, image2):
    image1 = get_thum(image1)
    image2 = get_thum(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        # linalg=linear+algebra,norm 
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    res = dot(a / a_norm, b / b_norm)
    return res

image1 = Image.open('images/1.png')
image2 = Image.open('images/2.png')
image3 = Image.open('images/3.png')
image4 = Image.open('images/4.png')

print(" 1 and 2 similarity:",image_similarity_voctors_via_numpy(image1, image2))

print(" 1 and 3 similarity:",image_similarity_voctors_via_numpy(image1, image3))

print(" 1 and 4 similarity:",image_similarity_voctors_via_numpy(image1, image4))

print(" 1 and 2 similarity:",MSE(image1, image2))

print(" 1 and 3 similarity:",MSE(image1, image3))

print(" 1 and 4 similarity:",MSE(image1, image4))