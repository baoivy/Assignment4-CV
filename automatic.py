import cv2
import numpy as np 

# Load the images
source_image = cv2.imread('/mnt/d/Ass4-CV/cat_224x224.jpg', cv2.IMREAD_GRAYSCALE)  # Image to project
destination_image = cv2.imread('/mnt/d/Ass4-CV/background.jpg', cv2.IMREAD_GRAYSCALE)  # Image with the target plane

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect and compute keypoints and descriptors
kp1, des1 = sift.detectAndCompute(source_image, None)
kp2, des2 = sift.detectAndCompute(destination_image, None)

# Match features using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Extract matched points
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Compute the homography matrix
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

# Warp the source image onto the destination
height, width = destination_image.shape
warped_image = cv2.warpPerspective(cv2.imread('/mnt/d/Ass4-CV/cat_224x224.jpg'), H, (width, height))

# Overlay the warped image onto the original destination image
destination_color = cv2.imread('/mnt/d/Ass4-CV/background.jpg')  # Load the color version
mask = (warped_image > 0).astype(np.uint8) * 255
destination_color[mask > 0] = warped_image[mask > 0]

# Save and display the result
cv2.imwrite('result_automatic.jpg', destination_color)
cv2.imshow("Automatic Transformation", destination_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
