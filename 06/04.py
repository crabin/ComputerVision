import cv2
import matplotlib.pyplot as plt
import numpy as np
image_path = 'images/number.png'

image = cv2.imread(image_path, 1)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def p_tile_method(image, p=50):
    # Flatten the image to get pixel values
    flattened = image.flatten()
    # Calculate the intensity value corresponding to p%
    threshold = np.percentile(flattened, p)
    # Apply threshold
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary_otsu = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

def minimum_error_threshold(image):
    # Calculate normalized histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()
    bins = np.arange(256)

    # Compute initial values
    total_mean = (bins * hist).sum()
    min_error = float('inf')
    best_threshold = 0

    for t in bins:
        w0 = hist[:t].sum()
        w1 = hist[t:].sum()
        if w0 == 0 or w1 == 0:
            continue
        mean0 = (bins[:t] * hist[:t]).sum() / w0
        mean1 = (bins[t:] * hist[t:]).sum() / w1
        error = w0 * (mean0 - total_mean)**2 + w1 * (mean1 - total_mean)**2
        if error < min_error:
            min_error = error
            best_threshold = t

    print(f"Best threshold: {best_threshold}")
    _, binary_image = cv2.threshold(image, best_threshold, 255, cv2.THRESH_BINARY)
    return binary_image, best_threshold

# Apply Minimum Error method
binary_min_error, binary_min_error_threshold = minimum_error_threshold(gray_image)

def differential_histogram_method(image):
    # Calculate histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    diff = np.diff(hist)
    # Find minimum index in the derivative (local minimum)
    threshold = np.argmin(diff)
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image, threshold

binary_diff_hist, binary_diff_hist_threshold = differential_histogram_method(gray_image)

def laplacian_histogram_method(image):
    # Apply Laplacian operator
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    # Threshold using Otsu's method
    _, binary_image = cv2.threshold(laplacian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image 

binary_laplacian = laplacian_histogram_method(gray_image)


plt.figure(figsize=(10, 8))

plt.subplot(2, 3, 1)
plt.hist(gray_image.ravel(), 256, [0, 256])

plt.subplot(2, 3, 2)
plt.imshow(binary_laplacian, cmap='gray')
plt.title('binary_laplacian Image')

plt.subplot(2, 3, 3)
plt.imshow(p_tile_method(gray_image, 50), cmap='gray')
plt.title('p_tile Image')
plt.subplot(2, 3, 4)
plt.imshow(binary_otsu, cmap='gray')
plt.title('binary_otsu')
plt.subplot(2, 3, 5)
plt.imshow(binary_min_error, cmap='gray')
plt.title(f'binary_min_error, threshold: {binary_min_error_threshold}')

plt.subplot(2, 3, 6)
plt.imshow(binary_diff_hist, cmap='gray')
plt.title(f'binary_diff_hist, threshold: {binary_diff_hist_threshold}')

plt.show()
cv2.waitKey(0)