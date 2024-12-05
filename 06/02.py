import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image

image = cv2.imread('images/kan.png')

# Get the image size
height, width = image.shape[:2]

# Define the transformation matrix for rotation and scaling
M = cv2.getRotationMatrix2D((width//2, height//2), -45, 1)  # Rotate by 45 degrees
euclidean_image  = cv2.warpAffine(image, M, (width, height))

# Define the similarity transformation matrix (scaling, rotation, translation)
M_similarity = cv2.getRotationMatrix2D((width // 2, height // 2), 0, 1.5)
similarity_image = cv2.warpAffine(euclidean_image, M_similarity, (width, height))

rotated_image = cv2.rotate(euclidean_image, cv2.ROTATE_180)

# Define the source and destination points for projective transformation
pts1 = np.float32([[50, 50], [200, 50], [50, 200], [200, 200]])
pts2 = np.float32([[50, 100], [200, 50], [50, 250], [200, 180]])


# Compute the homography matrix
M = cv2.getPerspectiveTransform(pts1, pts2)
projective_image = cv2.warpPerspective(image, M, (width, height))

# Example of a simple topological transformation: image warping
map_x, map_y = np.indices((height, width), dtype=np.float32)
map_x = map_x + 20 * np.sin(map_y / 20)  # Apply a sine distortion in the x-direction
map_y = map_y + 20 * np.cos(map_x / 20)  # Apply a cosine distortion in the y-direction

topological_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)


fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs[0, 0].imshow(cv2.cvtColor(euclidean_image, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title('Euclidean Transformation')
axs[0, 1].imshow(cv2.cvtColor(similarity_image, cv2.COLOR_BGR2RGB))
axs[0, 1].set_title('Similarity 1.5 Transformation')
axs[0, 2].imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
axs[0, 2].set_title('Affine rotated 180 Transformation')
axs[1, 0].imshow(cv2.cvtColor(projective_image, cv2.COLOR_BGR2RGB))
axs[1, 0].set_title('Projective Transformation')
axs[1, 1].imshow(cv2.cvtColor(topological_image, cv2.COLOR_BGR2RGB))
axs[1, 1].set_title('Topological Transformation')
axs[1, 2].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[1, 2].set_title('Original Image')
for ax in axs.flat:
    ax.axis('off')  
plt.tight_layout()
plt.show()