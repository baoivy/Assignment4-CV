import numpy as np
from PIL import Image

def compute_homography(src_points, dst_points):
    """
    Compute the homography matrix H such that dst_points ~ H * src_points.
    """
    assert len(src_points) == len(dst_points) and len(src_points) >= 4, "Need at least 4 points."

    A = []
    for (x, y), (x_prime, y_prime) in zip(src_points, dst_points):
        A.append([-x, -y, -1, 0, 0, 0, x * x_prime, y * x_prime, x_prime])
        A.append([0, 0, 0, -x, -y, -1, x * y_prime, y * y_prime, y_prime])

    A = np.array(A)
    _, _, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)
    return H / H[2, 2] 

def apply_homography_to_image(H, img):
    """
    Apply a homography transformation to an image.

    :param H: Homography matrix (3x3).
    :param img: NumPy array representing the image (height x width x channels).
    :return: Transformed NumPy array of the image.
    """
    height, width, channels = img.shape  # Extract dimensions
    
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    coords = np.stack([x.ravel(), y.ravel(), np.ones_like(x.ravel())])  # Homogeneous coordinates
    
    transformed_coords = (H @ coords).T
    transformed_coords = transformed_coords[:, :2] / transformed_coords[:, 2:3]
    
    transformed_img = np.zeros_like(img)
    
    for i in range(height):
        for j in range(width):
            tx, ty = transformed_coords[i * width + j]
            tx, ty = int(round(tx)), int(round(ty))
            if 0 <= tx < width and 0 <= ty < height:  # Ensure coordinates are within bounds
                transformed_img[i, j] = img[ty, tx]  # Map pixel values

    return transformed_img

src_points = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
dst_points = np.array([[70, 80], [150, 70], [140, 150], [30, 140]], dtype=np.float32)

points = Image.open('dog.jpg').convert("RGB")
points_np = np.array(points)

H = compute_homography(src_points, dst_points)
transformed_points = apply_homography_to_image(H, points_np)

transform_img = Image.fromarray(transformed_points.astype(np.uint8))
transform_img.save('dog1.jpg')