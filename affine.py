import numpy as np
from PIL import Image

# Define the affine transformation matrix 
theta = np.deg2rad(40)  
scale = 1.2  
tx, ty = 70, 30 

M = np.array([
    [scale * np.cos(theta), -scale * np.sin(theta), tx],
    [scale * np.sin(theta), scale * np.cos(theta), ty],
    [0, 0, 1]
])

# Compute the inverse of the transformation matrix
M_inverse = np.linalg.inv(M)

def apply_affine_manual(img_np, M_inverse):
    """
    Apply an affine transformation to an image using a transformation matrix.

    :param img_np: NumPy array of the image (height x width x channels).
    :param M_inverse: Inverse affine transformation matrix (3x3).
    :return: Transformed NumPy array of the image.
    """
    height, width, channels = img_np.shape  

    transformed_img = np.zeros_like(img_np)

    for y in range(height):
        for x in range(width):
            src_coords = np.array([x, y, 1])
            dest_coords = M_inverse @ src_coords
            dest_x, dest_y = dest_coords[:2] / dest_coords[2]

            if 0 <= dest_x < width and 0 <= dest_y < height:
                dest_x, dest_y = int(round(dest_x)), int(round(dest_y))
                transformed_img[y, x] = img_np[dest_y, dest_x] 

    return transformed_img

img = Image.open('/mnt/d/Ass4-CV/dog.jpg').convert("RGB")
img_np = np.array(img)

transformed_img_np = apply_affine_manual(img_np, M_inverse)
transformed_img = Image.fromarray(transformed_img_np.astype(np.uint8))
transformed_img.save('transformed_image4.jpg')
