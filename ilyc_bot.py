import cv2
import io
import imageio.v3
import keyboard
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from rembg import remove
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from scipy.spatial import procrustes
import keras
from typing import Literal
from textwrap import dedent

FACE_SHAPE_CLASS_MAPPING = {
    "0": "heart",
    "1": "oblong",
    "2": "oval",
    "3": "round",
    "4": "square",
}


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
    loaded_model = load_model("./asm_data/model/facial_keypoints_model_50.h5")
    new_image = cv2.imread(imagepath)
    if resize:
        new_image = cv2.resize(new_image, (250, 300)) / 255.0
    new_image_keypoints = loaded_model.predict(np.expand_dims(new_image, axis=0))
    new_image_keypoints_processed = visualize_keypoints(
        new_image, new_image_keypoints[0], new_image_keypoints[0], show=show
    )
    return new_image, new_image_keypoints_processed


def visualize_keypoints(
    image,
    keypoints1,
    keypoints2,
    label1="Keypoints 1",
    label2="Keypoints 2",
    resize=True,
    debug=False,
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
    - debug: set True to print the predicted_keypoints matrix
    - show: set True to make a plot of the overlay

    Returns:
    - keypoints2 (if resize=False, this is the same as the input)
    """
    if resize:
        keypoints1 = resize_keypoints(keypoints1)
        keypoints2 = resize_keypoints(keypoints2)
    if debug:
        print(f"hello this is the keypoints matrix\n{keypoints2}")

    # Display the image
    if show:
        plt.imshow(image)
        plt.scatter(
            keypoints1[:, 0], keypoints1[:, 1], c="blue", label=label1, s=12
        )  # Ground truth
        plt.scatter(
            keypoints2[:, 0], keypoints2[:, 1], c="red", label=label2, s=10
        )  # Predictions
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


# ======================================== Face Shape Classification Model ========================================
def get_face_shape_model():
    return keras.saving.load_model(
        "face_shape_models/face_shape_identifier.keras", safe_mode=False
    )


def predict_face_shape(model, img):
    predicted_label = np.argmax(model.predict(np.expand_dims(img, axis=0)), axis=1)[0]
    return FACE_SHAPE_CLASS_MAPPING[str(predicted_label)]


def get_optimal_hairstyle(
    gender: Literal["male", "female"],
    face_shape: Literal["oval", "heart", "oblong", "round", "square"],
):
    if gender == "male":
        if face_shape == "oval":
            return {
                "recommendations": [
                    "buzz",
                    "textured quiff",
                    "pompadour",
                    "man bun",
                    "comb over",
                ],
                "explanation": dedent(
                    """
                    Oval faces tend to be symmetric and balanced, so most 
                    hairstyles will look good on oval faces. It's best to pick 
                    hairstyles that draw attention to your features and keep 
                    the hair off of your face. Avoid hairstyles that cover your 
                    forehead, as it will make you face look rounder.
                    """
                ),
            }
        elif face_shape == "round":
            return {
                "recommendations": ["slick back", "skin fades", "spiky hair"],
                "explanation": dedent(
                    """
                    Round faces are best fitted with hairstyles that are 
                    short on the sides but add height or volume, 
                    to create the illusion that the face is longer.
                    """
                ),
            }
        elif face_shape == "oblong":
            return {
                "recommendations": ["side part", "brush up", "short spiky hair"],
                "explanation": dedent(
                    """
                    Oblong faces are long, so you should try 
                    shorter haircuts to avoid making your face look even longer.
                    """
                ),
            }
        elif face_shape == "heart":
            return {
                "recommendations": [
                    "textured crop",
                    "side swept bangs",
                    "pompadour",
                    "crew cut",
                    "ivy league",
                ],
                "explanation": dedent(
                    """
                    Heart-shaped faces tend to have a wider forehead 
                    and a narrower chin, so hairstyles that add volume around the 
                    chin or soften the forehead are ideal. Side-swept styles or 
                    longer hair on the sides can help balance the proportions of 
                    the face, whiletextured or voluminous styles on top can draw 
                    attention away from the forehead.
                    """
                ),
            }
        elif face_shape == "square":
            return {
                "recommendations": [
                    "textured quiff",
                    "side part",
                    "ivy league",
                    "fade",
                    "comb over",
                ],
                "explanation": dedent(
                    """
                    Square faces are angular, so softening the sharp 
                    lines with styles that add height and texture is a good idea. 
                    Styles that are longer on top with short sides or fades work well 
                    to balance the strong jawline. Avoid heavy bangs or blunt cuts 
                    that emphasize the square shape.
                    """
                ),
            }

    if gender == "female":
        if face_shape == "oval":
            return {
                "recommendations": [
                    "long waves",
                    "bob cut",
                    "pixie cut",
                    "soft curls",
                    "side-swept bangs",
                ],
                "explanation": dedent(
                    """
                    Oval faces are well-balanced, so most hairstyles 
                    work. However, styles that highlight the features, such as soft 
                    curls or waves, will enhance the natural symmetry. Bobs and pixie 
                    cuts are great for accentuating facial structure. Side-swept bangs 
                    can add a bit of volume and texture without overwhelming the face.
                    """
                ),
            }
        elif face_shape == "round":
            return {
                "recommendations": [
                    "long layers",
                    "side part",
                    "angular bob",
                    "textured waves",
                ],
                "explanation": dedent(
                    """Round faces are best suited for hairstyles that 
                    add height and volume on top, creating the illusion of a longer 
                    face. Long layers or waves add dimension and texture, while side 
                    parts and angular bobs can add sharpness to the face and draw  
                    attention away from the width.
                    """
                ),
            }
        elif face_shape == "oblong":
            return {
                "recommendations": [
                    "soft waves",
                    "layered bob",
                    "shaggy bob",
                    "side-swept bangs",
                ],
                "explanation": dedent(
                    """For oblong faces, it's best to avoid too much 
                    length. Hairstyles that add volume around the cheeks, like soft 
                    waves or layered bobs, work best. Side-swept bangs can help 
                    balance the long shape of the face and create a softer silhouette.
                    """
                ),
            }
        elif face_shape == "heart":
            return {
                "recommendations": [
                    "long layers with volume",
                    "bob with soft waves",
                    "side-swept bangs",
                    "pixie cut with volume",
                    "asymmetrical lob",
                ],
                "explanation": dedent(
                    """Heart-shaped faces can benefit from hairstyles 
                    that balance the broader forehead and narrow chin. Long layers 
                    with volume around the jawline help to soften sharp features. A 
                    bob with soft waves or a pixie cut with added volume creates harmony 
                    in the face's proportions. Side-swept bangs or asymmetrical lobs can 
                    also help minimize the width of the forehead.
                    """
                ),
            }
        elif face_shape == "square":
            return {
                "recommendations": [
                    "soft curls",
                    "layered lob",
                    "textured waves",
                    "asymmetrical bob",
                    "side-swept bangs",
                ],
                "explanation": dedent(
                    """Square faces benefit from softening the angular features. 
                    Soft curls or waves work well to create texture and volume, which helps 
                    round out the jawline.  Layered lobs or asymmetrical bobs also help soften 
                    the square face. Side-swept bangs add movement and balance the strong features.
                    """
                ),
            }


# ======================================== Data Preprocessing ========================================
def take_image_from_filepath(filepath):
    """
    Takes an image filepath, reads it, removes the background, and reshapes to (128, 128, 3)
    Input:
    - filepath: string destination of the image

    Returns:
    - preprocessed image as a NumPy array
    """
    with open(filepath, "rb") as f:
        input_img = f.read()

    # remove background image
    img_no_bg = remove(input_img)

    # reconvert to NumPy array
    img = imageio.v3.imread(io.BytesIO(img_no_bg), format_hint=".png")
    # img = imageio.v3.imread(filepath)

    if img.ndim == 2:  # grayscale image
        img = np.stack((img,) * 3, axis=-1)
    elif img.shape[-1] == 4:  # alpha
        img = img[..., :3]

    img = resize(img, (128, 128, 3), mode="reflect", anti_aliasing=True).astype(
        "float32"
    )
    return img


def take_image_from_camera(output_file="camera_out.jpg"):
    """
    Opens the computers face cam and takes a picture
    NOTE: still needs testing
    Input:
    - output_file (optional): str for the name of the image

    Returns:
    - NumPy array of an image
    """
    # Open the camera
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Unable to access the camera.")
        return

    print("Camera is ready. Press 'q' to quit or any other key to continue.")

    captured_frame = None
    while True:
        # Read frame from the camera
        ret, frame = camera.read()
        if not ret:
            print("Error: Unable to read from the camera.")
            break

        # Convert frame to RGB for matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame using matplotlib
        plt.imshow(frame_rgb)
        plt.axis("off")
        plt.show(block=False)
        plt.pause(0.001)

        # Prompt user for input
        if keyboard.is_pressed("q"):
            print("Quit without capturing.")
            break

        captured_frame = resize(
            frame_rgb, (250, 300, 3), mode="reflect", anti_aliasing=True
        ).astype("float32")
        captured_frame = (captured_frame * 255).astype(
            np.uint8
        )  # Convert back to uint8
        print(f"Image captured and saved to {output_file}")

    # Save the captured frame if available
    if captured_frame is not None:
        cv2.imwrite(output_file, captured_frame)

    # Release the camera
    camera.release()
    plt.close()
    return captured_frame


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


# ======================================== Interface ========================================
def ilyc_interface():
    """
    Interface for program.
    Prompts user to input image filepath or use camera option
    Then, applies preprocessing to bring size to (250, 300, 3) and remove background
    """
    instructions = """
    For best results, please adhere to the following guidelines:
      1) Image must be front facing
      2) Image must be mostly your face
      3) Preferrably no smiling
    """
    print(instructions)
    image_mode = input(
        "Select input mode. Press 'c' to use your computer camera, 'f' to enter a filepath, or 'q' to quit: "
    )
    if image_mode == "c":
        filepath = "camera_out.jpg"
        img = take_image_from_camera(filepath)
    elif image_mode == "f":
        # NOTE: currently crashes if user quits while in camera mode :/
        filepath = input("Enter your filepath: ")
        img = take_image_from_filepath(filepath)
    elif image_mode == "q":
        return
    else:
        ilyc_interface()
        return

    # Preview image after removing background and resizing; prompt for validity
    plt.imshow(img)
    plt.show()
    valid_input = input("Is this a valid picture? (y/n) ")
    if valid_input.lower() != "y":
        ilyc_interface()
        return

    fs_model = get_face_shape_model()
    fs_pred = predict_face_shape(fs_model, img)
    gender = ''
    while gender not in ["male", "female"]:
        gender = input("What is your gender? ( Male / Female ) ").lower()
    hair_style_recomm = get_optimal_hairstyle(gender, fs_pred)
    print(
        f"""
Your Predicted Face Shape: {fs_pred}\n
Optimal Hairstyles: {', '.join(hair_style_recomm["recommendations"])}
{hair_style_recomm["explanation"]}
        """
    )
    hair_selection = ''
    while hair_selection not in hair_style_recomm["recommendations"]:
        hair_selection = input(
            f"Pick one of the following to apply ({', '.join(hair_style_recomm["recommendations"])}): "
        ).lower()
    hair_points_file = "./asm_data/frontalshapes_manuallyannotated_46points/83a.pts"  # TODO: link hair_selection to ASM of that hair style
    hair_image = cv2.imread(
        "./asm_data/frontalimages_spatiallynormalized/83a.jpg"
    )  # TODO: replace with actual hair image
    hair_points = load_pts_file(
        hair_points_file
    )  # TODO: replace points_path with the hairstyle points

    # Apply model to generate predicted ASM on inputted image
    resized_image, predicted_points = asm_predict(filepath, show=False)
    with open("points.txt", "a") as f:
        f.write(str(predicted_points))

    # Make the hair points align with the predicted points
    aligned_hair, similarity_disparity, affine_matrix = align_points_affine(
        predicted_points, hair_points
    )
    print(affine_matrix)

    # Apply the affine transformation on hair image
    output_size = (resized_image.shape[1], resized_image.shape[0])
    transformed_image = cv2.warpAffine(
        hair_image, affine_matrix, output_size, flags=cv2.INTER_LINEAR
    )
    plt.imshow(transformed_image)
    plt.show()

    # OPTIONAL: Plots (plots upside down because origin is at the bottom left but in images its at the top left)
    plot_transformations(
        hair_points, aligned_hair, predicted_points, similarity_disparity
    )

    # Visualize
    # NOTE: first function call is with the transformation stuff, second is without
    visualize_keypoints(
        resized_image,
        predicted_points,
        aligned_hair,
        label1="Predicted",
        label2="Aligned Hair",
        resize=False,
    )
    visualize_keypoints(
        resized_image,
        predicted_points,
        hair_points,
        label1="Predicted",
        label2="Unaligned Hair",
        resize=False,
    )
    return ilyc_interface()


def main():
    # TESTING: ./asm_data/frontalimages_spatiallynormalized/190b.jpg
    ilyc_interface()


if __name__ == "__main__":
    main()
