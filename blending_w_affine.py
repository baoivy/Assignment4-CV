import numpy as np
from PIL import Image

def compute_perspective_transform(src_points, dst_points):
    """
    Compute the perspective transform matrix from src_points to dst_points.
    """
    assert len(src_points) == 4 and len(dst_points) == 4, "Need exactly 4 points."

    A = []
    for (x, y), (x_, y_) in zip(src_points, dst_points):
        A.append([-x, -y, -1, 0, 0, 0, x * x_, y * x_, x_])
        A.append([0, 0, 0, -x, -y, -1, x * y_, y * y_, y_])

    A = np.array(A)
    _, _, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)
    return H / H[2, 2]  # Normalize so H[2, 2] == 1


def apply_perspective_transform(src_img, dst_img, H):
    """
    Apply a perspective transform to the source image and paste it on the destination image.
    """
    dst_w, dst_h = dst_img.shape[:2]
    dst_coords = np.indices((dst_h, dst_w)).reshape(2, -1).T  # (y, x)
    dst_coords = np.hstack([dst_coords[:, ::-1], np.ones((dst_coords.shape[0], 1))])  # (x, y, 1)

    src_coords = (np.linalg.inv(H) @ dst_coords.T).T
    src_coords = src_coords[:, :2] / src_coords[:, 2:]  # Normalize coordinates


    transformed_img = dst_img.copy()
    for i, (x, y) in enumerate(src_coords):
        src_x, src_y = int(round(x)), int(round(y))
        dst_x, dst_y = int(round(dst_coords[i][0])), int(round(dst_coords[i][1]))
        if 0 <= src_y < src_img.shape[0] and 0 <= src_x < src_img.shape[1]:
            transformed_img[dst_y, dst_x] = src_img[src_y, src_x]

    print(transformed_img.shape)
    return transformed_img


src_path = "/mnt/d/Ass4-CV/cat_224x224.jpg"  
dst_path = "/mnt/d/Ass4-CV/background.jpg"  

src_img = np.array(Image.open(src_path).convert("RGB"))
dst_img = np.array(Image.open(dst_path).convert("RGB").rotate(-90))

print(src_img.shape)
print(dst_img.shape)
src_points = [[0, 0], [src_img.shape[1] - 1, 0], [src_img.shape[1] - 1, src_img.shape[0] - 1], [0, src_img.shape[0] - 1]]
dst_points = [[1210, 840], [1950, 1100], [1900, 2530], [1070, 2460]]

H = compute_perspective_transform(src_points, dst_points)


result_img_np = apply_perspective_transform(src_img, dst_img, H)
result_img = Image.fromarray(result_img_np)
result_img.save("result_image.jpg")
