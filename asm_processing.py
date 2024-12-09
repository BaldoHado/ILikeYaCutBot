import cv2
import imageio.v3
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from scipy.spatial import procrustes

# ======================================== ASM Model ========================================
def asm_predict(imagepath, resize=True, show=True):
    """
    Input:
    - imagepath: filepath to image; fed into ASM model to predict points
    - resize: set True to resize new_image
    - show: set True to show the overlay of the image and the predicted points
        - Note, the visualization will have 2 keypoints labeled but they are both identical

    Returns:
    - tuple of the new image (resized if resize=True) and the predicted keypoints
    """
    loaded_model = load_model("./facial_keypoints_model_combined_dataset.h5")
    new_image = imageio.v2.imread(imagepath)
    if resize:
        new_image = cv2.resize(new_image, (250, 300)) / 255.0
    new_image_keypoints = loaded_model.predict(np.expand_dims(new_image, axis=0))
    new_image_keypoints_processed = resize_keypoints(new_image_keypoints)
    return new_image, new_image_keypoints_processed


def visualize_keypoints(
    image,
    keypoints1,
    keypoints2,
    image2=None,
    label1="Keypoints 1",
    label2="Keypoints 2",
    resize=True,
    show=True,
):
    """
    Overlay keypoints and predicted_keypoints on top of image

    Input:
    - image: NumPy array of an image; used to display the inputted image
    - keypoints1: ASM keypoints
    - keypoints2: ASM keypoints
    - label1: title for keypoints1
    - label2: title for keypoints2
    - resize: set True to scale keypoints and predicted_keypoints to resized dimensions
    - show: set True to make a plot of the overlay

    Returns:
    - keypoints2 (if resize=False, this is the same as the input)
    """
    if resize:
        keypoints1 = resize_keypoints(keypoints1)
        keypoints2 = resize_keypoints(keypoints2)

    # Display the image
    plt.imshow(image)
    if image2 is not None:
        plt.imshow(image2)

    if show:
        plt.scatter(
            keypoints1[:, 0], keypoints1[:, 1], c="blue", label=label1, s=12
        ) 
        plt.scatter(
            keypoints2[:, 0], keypoints2[:, 1], c="red", label=label2, s=10
        )  
    plt.legend()
    plt.axis("off")
    plt.show()
    return keypoints2


# Convert normalized keypoints back to the resized image dimensions
def resize_keypoints(keypoints, dimensions=[250, 300]):
    return (
        keypoints.reshape(-1, 2) * dimensions
    )  # Scale keypoints to resized dimensions


# Load .pts files
def load_pts_file(filepath):
    with open(filepath, "r") as file:
        lines = file.readlines()
        points = []
        for line in lines[3:-1]:  # skip header lines
            x, y = line.strip().split()
            points.append((float(x), float(y)))
        return np.array(points)

# ======================================== ASM Transformation ========================================
def align_points(reference, to_align):
    """
    Homography transformation to make to_align keypoints look more like reference keypoints
    """
    # 1) Order the points in both sets
    reference_ordered = order_points(reference)
    to_align_ordered = order_points(to_align)

    # 2) Compute the centroid of the ground truth
    reference_centroid = np.mean(reference_ordered, axis=0)
    to_align_centroid = np.mean(to_align_ordered, axis=0)

    # 3) Procrustes analysis for similarity transformation
    ground_truth_centered = reference_ordered - reference_centroid
    predicted_centered = to_align_ordered - to_align_centroid

    _, aligned_keypoints, similarity_disparity = procrustes(
        ground_truth_centered, predicted_centered
    )

    # 4) Restore the aligned predicted points to the ground truth centroid
    aligned_keypoints += reference_centroid

    # 5) Homography transformation for finer alignment
    homography_matrix, mask = cv2.findHomography(
        aligned_keypoints, reference_ordered, cv2.RANSAC
    )

    # Filter out outliers
    mask = mask.ravel().astype(bool)  # Convert mask to boolean
    if not np.any(mask):  # If no inliers are found
        print("No inliers found. Skipping homography refinement.")
        return aligned_keypoints, similarity_disparity, None

    # Apply mask
    aligned_keypoints = aligned_keypoints[mask]
    reference_ordered = reference_ordered[mask]

    # Recompute homography with inliers only
    homography_matrix, _ = cv2.findHomography(aligned_keypoints, reference_ordered, 0)

    aligned_predicted_homo = np.hstack(
        (aligned_keypoints, np.ones((aligned_keypoints.shape[0], 1)))
    )  # Homogeneous coordinates
    aligned_points_homo = aligned_predicted_homo @ homography_matrix.T
    aligned_points = (
        aligned_points_homo[:, :2] / aligned_points_homo[:, 2:]
    )  # Back to Cartesian coordinates

    return aligned_points, similarity_disparity, homography_matrix


def align_points_affine(reference, to_align):
    """
    Aligns `to_align` points to `reference` points using an affine transformation.

    Parameters:
        reference (np.ndarray): Ground truth points (N x 2 array, unordered).
        to_align (np.ndarray): Points to align (N x 2 array, unordered).

    Returns:
        aligned_points (np.ndarray): Aligned points after applying Procrustes and affine transformation.
        similarity_disparity (float): Disparity after Procrustes alignment.
        affine_matrix (np.ndarray): Affine transformation matrix.
    """
    # 1) Order the points in both sets
    reference_ordered = order_points(reference)
    to_align_ordered = order_points(to_align)

    # 2) Compute the centroid of the ground truth and align set
    reference_centroid = np.mean(reference_ordered, axis=0)
    to_align_centroid = np.mean(to_align_ordered, axis=0)

    # 3) Procrustes analysis for similarity transformation
    reference_centered = reference_ordered - reference_centroid
    to_align_centered = to_align_ordered - to_align_centroid
    _, aligned_keypoints, similarity_disparity = procrustes(
        reference_centered, to_align_centered
    )

    # 4) Restore the aligned predicted points to the reference centroid
    aligned_keypoints += reference_centroid

    # 5) Compute affine transformation
    affine_matrix, inliers = cv2.estimateAffinePartial2D(
        aligned_keypoints, reference_ordered, method=cv2.RANSAC
    )

    if affine_matrix is None:
        print("Affine transformation failed. Returning Procrustes-aligned points.")
        return aligned_keypoints, similarity_disparity, None

    # Apply affine transformation
    aligned_points = cv2.transform(
        np.expand_dims(aligned_keypoints, axis=0), affine_matrix
    )[0]

    return aligned_points, similarity_disparity, affine_matrix
    # affine_matrix, _ = cv2.estimateAffinePartial2D(to_align_ordered, reference_ordered, method=cv2.RANSAC)
    # aligned_points = cv2.transform(np.expand_dims(to_align_ordered, axis=0), affine_matrix)[0]
    # return aligned_points, 0, affine_matrix

def order_points(points):
    """
    Order list of points based on angular position relative to the centroid (1st) and distance to centroid (2nd)
    Inputs:
    - points: 2D NumPy array of keypoints

    Returns:
    - ordered 2D NumPy array of keypoints
    """
    centroid = np.mean(points, axis=0)  # centroid

    angles = np.arctan2(
        points[:, 1] - centroid[1], points[:, 0] - centroid[0]
    )  # angle about the centroid

    distances = np.linalg.norm(points - centroid, axis=1)  # distance to centroid

    # sort by angle first, then distance
    criteria = np.column_stack((angles, distances))
    ordered_indices = np.lexsort((criteria[:, 1], criteria[:, 0]))
    ordered_points = points[ordered_indices]

    return ordered_points

def compute_procrustes_matrix(reference_ordered, to_align_ordered):
    """
    Compute the Procrustes affine transformation matrix that maps to_align_ordered to 
    reference_ordered using rotation, scaling, and translation derived from Procrustes analysis.
    
    Parameters:
        reference_ordered (np.ndarray): Nx2 array of reference points (already ordered)
        to_align_ordered (np.ndarray): Nx2 array of points to align (already ordered)
        
    Returns:
        P (np.ndarray): 2x3 affine matrix representing the Procrustes transform
        scale (float): scaling factor found by Procrustes
        R (np.ndarray): 2x2 rotation matrix
        t (np.ndarray): translation vector (1x2)
    """
    # Compute centroids
    reference_centroid = np.mean(reference_ordered, axis=0)
    to_align_centroid = np.mean(to_align_ordered, axis=0)

    # Center points at their centroids
    reference_centered = reference_ordered - reference_centroid
    to_align_centered = to_align_ordered - to_align_centroid

    # Compute matrix for SVD
    A = reference_centered.T @ to_align_centered
    U, S, Vt = np.linalg.svd(A)

    # Compute rotation
    R = U @ Vt
    # Ensure a proper rotation (check for reflection)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    # Compute scale
    # scale = sum of singular values / sum of squared distances in to_align_centered
    var_y = np.sum(to_align_centered**2)
    scale = np.sum(S) / var_y

    # Translation
    t = reference_centroid - scale * (to_align_centroid @ R)

    # Construct the Procrustes affine matrix P:
    # P maps original to_align_ordered points to their Procrustes-aligned counterpart.
    P = np.array([
        [scale * R[0,0], scale * R[0,1], t[0]],
        [scale * R[1,0], scale * R[1,1], t[1]]
    ])

    return P, scale, R, t

def to_homogeneous(T):
    # T is 2x3
    # Convert to 3x3 homogeneous form
    return np.vstack([T, [0, 0, 1]])

def from_homogeneous(T_hom):
    # T_hom is 3x3
    # Convert back to 2x3
    return T_hom[:2, :]

def recenter_warped_image(image, affine_matrix, output_size):
    """
    Applies an affine transformation and recenters the result.

    Parameters:
        image (np.ndarray): The input image.
        affine_matrix (np.ndarray): 2x3 affine transformation matrix.
        output_size (tuple): (width, height) of the output image.

    Returns:
        recentered_image (np.ndarray): The recentered transformed image.
    """
    # Get image dimensions
    h, w = image.shape[:2]

    # Define the four corners of the input image
    corners = np.array([
        [0, 0],  # Top-left
        [w, 0],  # Top-right
        [0, h],  # Bottom-left
        [w, h]   # Bottom-right
    ], dtype=np.float32)

    # Transform the corners using the affine matrix
    transformed_corners = cv2.transform(np.array([corners]), affine_matrix)[0]

    # Compute the bounding box of the transformed corners
    x_min, y_min = transformed_corners.min(axis=0)
    x_max, y_max = transformed_corners.max(axis=0)

    # Compute the center of the bounding box
    bbox_center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])

    # Compute the center of the output image
    output_center = np.array([output_size[0] / 2, output_size[1] / 2])

    # Compute the translation needed to recenter the image
    translation = output_center - bbox_center

    # Adjust the affine matrix to include the recentering translation
    recenter_matrix = affine_matrix.copy()
    recenter_matrix[:, 2] += translation

    # Apply the adjusted affine transformation
    recentered_image = cv2.warpAffine(image, recenter_matrix, output_size, flags=cv2.INTER_LINEAR)

    return recentered_image

def plot_transformations(
    hair_points, hair_points_aligned, predicted, similarity_disparity
):
    """
    Helper function to show the ground truth keypoints, predicted keypoints, and aligned predicted keypoints

    Inputs:
    - hair_points: 2D array of hair keypoints
    - hair_points_aligned: 2D array of hair keypoints after affine transformation
    - predicted: 2D array of predicted keypoints
    - similarity_disparity: float metric of the similarity between predicted_aligned and ground_truth

    Returns:
    - None
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(
        hair_points[:, 0],
        hair_points[:, 1],
        label="Hair Points (Unaligned)",
        c="blue",
        marker="o",
    )
    plt.scatter(
        hair_points_aligned[:, 0],
        hair_points_aligned[:, 1],
        label="Hair Points (Aligned)",
        c="green",
        marker="+",
    )
    plt.scatter(
        predicted[:, 0], predicted[:, 1], label="Predicted", c="red", marker="x"
    )
    plt.legend()
    plt.title(
        f"Alignment with Combined Ordering\nSimilarity Disparity: {similarity_disparity:.4f}"
    )
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.axis("equal")
    plt.show()