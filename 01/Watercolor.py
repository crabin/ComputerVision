import cv2

# read image
img = cv2.imread('image/Watercolor.png')

# Use the oilPainting function in the xphoto module to process the image
res = cv2.xphoto.oilPainting(img, 4, 1)

cv2.imshow('cartoon_image', res)

# save
cv2.imwrite('save/cartoon_image.png', res)

cv2.waitKey(0)
cv2.destroyAllWindows()
