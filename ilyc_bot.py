import os
import cv2
import io
import imageio.v3
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from rembg import remove
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from scipy.spatial import procrustes


# ======================================== ASM Model ======================================== 
def asm_predict(new_image, resize=True, show=True):
    '''
    Input:
    - new_image: NumPy array of an image; fed into ASM model to predict points
    - resize: set True to resize new_image
    - show: set True to show the overlay of the image and the predicted points
        - Note, the visualization will have 2 keypoints labeled but they are both identical

    Returns:
    - tuple of the new image (resized if resize=True) and the predicted keypoints
    '''
    loaded_model = load_model("./asm_data/model/facial_keypoints_model_50.h5")
    new_image = cv2.imread(new_image)
    if resize:
      new_image = cv2.resize(new_image, (250, 300)) / 255.0
    new_image_keypoints = loaded_model.predict(np.expand_dims(new_image, axis=0))
    new_image_keypoints_processed = visualize_keypoints(new_image, new_image_keypoints[0], new_image_keypoints[0], show=show)
    return new_image, new_image_keypoints_processed

def visualize_keypoints(image, keypoints, predicted_keypoints, resize=True, debug=False, show=True):
    '''
    Overlay keypoints and predicted_keypoints on top of image

    Input:
    - image: NumPy array of an image; used to display the inputted image
    - keypoints: the ground truth or reference ASM keypoints
    - predicted_keypoints: keypoints predicted by our ASM model
    - resize: set True to scale keypoints and predicted_keypoints to resized dimensions
    - debug: set True to print the predicted_keypoints matrix
    - show: set True to make a plot of the overlay

    Returns:
    - predicted_keypoints (if resize=False, this is the same as the input)
    '''
    if resize:
      keypoints = resize_keypoints(keypoints)
      predicted_keypoints = resize_keypoints(predicted_keypoints)
    if debug:
      print(f"hello this is the keypoints matrix\n{predicted_keypoints}")

    # Display the image
    if show:
      plt.imshow(image)
      plt.scatter(keypoints[:, 0], keypoints[:, 1], c='blue', label='Hair Points', s=12)  # Ground truth
      plt.scatter(predicted_keypoints[:, 0], predicted_keypoints[:, 1], c='red', label='Prediction', s=10)  # Predictions
      plt.legend()
      plt.axis("off")
      plt.show()
    return predicted_keypoints

# Convert normalized keypoints back to the resized image dimensions
def resize_keypoints(keypoints, dimensions=[250, 300]):
    return keypoints.reshape(-1, 2) * dimensions  # Scale keypoints to resized dimensions

# Load .pts files
def load_pts_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
        points = []
        for line in lines[3:-1]:  # skip header lines
            x, y = line.strip().split()
            points.append((float(x), float(y)))
        return np.array(points)

# ======================================== Data Preprocessing ======================================== 
def take_image_from_filepath(filepath):
    '''
    Takes an image filepath, reads it, removes the background, and reshapes to (300, 250, 3)
    Input:
    - filepath: string destination of the image

    Returns:
    - preprocessed image as a NumPy array
    '''
    with open(filepath, 'rb') as f:
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

    img = resize(img, (300, 250, 3), mode='reflect', anti_aliasing=True).astype('float32')
    return img

def take_image_from_camera(output_file='camera_out.jpg'):
    '''
    Opens the computers face cam and takes a picture
    NOTE: still needs testing
    Input:
    - output_file (optional): str for the name of the image

    Returns:
    - NumPy array of an image
    '''
    # Open the camera
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Unable to access the camera.")
        return

    print("Camera is ready. Press 'q' to quit.")

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
        plt.axis('off')
        plt.show(block=False)
        plt.pause(0.001)

        # Prompt user for input
        user_input = input("Press 'c' to capture an image, 'a' to take another frame, or 'q' to quit: ").strip().lower()
        if user_input == 'c':
            captured_frame = resize(frame, (300, 250, 3), mode='reflect', anti_aliasing=True).astype('float32')
            print(f"Image captured and saved to {output_file}")
            break
        elif user_input == 'a':
            continue
        elif user_input == 'q':
            print("Quit without capturing.")
            break

        # NOTE: doesn't work on jupyter notebooks
        # # Check for user input
        # user_input = plt.waitforbuttonpress(timeout=0)
        # if user_input:
        #     # Get the key pressed
        #     key = plt.get_current_fig_manager().canvas.manager.keypress
        #     if key == 'c':
        #         # Capture the current frame
        #         captured_frame = frame
        #         print(f"Image captured and saved to {output_file}")
        #         break
        #     elif key == 'q':
        #         # Quit without capturing
        #         print("Quit without capturing.")
        #         break

    # Save the captured frame if available
    if captured_frame is not None:
        cv2.imwrite(output_file, captured_frame)

    # Release the camera
    camera.release()
    plt.close()
    return captured_frame

# ======================================== ASM Transformation ======================================== 
def align_points(reference, to_align):
    '''
    Homography transformation to make to_align keypoints look more like reference keypoints
    '''
    # 1) Order the points in both sets
    reference_ordered = order_points(reference)
    to_align_ordered = order_points(to_align)

    # 2) Compute the centroid of the ground truth
    reference_centroid = np.mean(reference_ordered, axis=0)
    to_align_centroid = np.mean(to_align_ordered, axis=0)

    # 3) Procrustes analysis for similarity transformation
    ground_truth_centered = reference_ordered - reference_centroid
    predicted_centered = to_align_ordered - to_align_centroid

    _, aligned_keypoints, similarity_disparity = procrustes(ground_truth_centered, predicted_centered)

    # 4) Restore the aligned predicted points to the ground truth centroid
    aligned_keypoints += reference_centroid

    # 5) Homography transformation for finer alignment
    homography_matrix, _ = cv2.findHomography(aligned_keypoints, reference_ordered, cv2.RANSAC)
    aligned_predicted_homo = np.hstack((aligned_keypoints, np.ones((aligned_keypoints.shape[0], 1))))  # Homogeneous coordinates
    aligned_points_homo = aligned_predicted_homo @ homography_matrix.T
    aligned_points = aligned_points_homo[:, :2] / aligned_points_homo[:, 2:]  # Back to Cartesian coordinates

    return aligned_points, similarity_disparity, homography_matrix

def order_points(points):
    '''
    Order list of points based on angular position relative to the centroid (1st) and distance to centroid (2nd)
    Inputs:
    - points: 2D NumPy array of keypoints

    Returns:
    - ordered 2D NumPy array of keypoints
    '''
    centroid = np.mean(points, axis=0)  # centroid

    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])  # angle about the centroid

    distances = np.linalg.norm(points - centroid, axis=1)  # distance to centroid

    # sort by angle first, then distance
    criteria = np.column_stack((angles, distances))
    ordered_indices = np.lexsort((criteria[:, 1], criteria[:, 0]))
    ordered_points = points[ordered_indices]

    return ordered_points

def plot_transformations(ground_truth, predicted, predicted_aligned, similarity_disparity):
  '''
  Helper function to show the ground truth keypoints, predicted keypoints, and aligned predicted keypoints
  Input:
  - ground_truth: 2D array of ground truth keypoints
  - predicted: 2D array of predicted keypoints
  - predicted_aligned: 2D array of aligned predicted keypoints
  - similarity_disparity: float metric of the similarity between predicted_aligned and ground_truth
  
  Returns:
  - None
  '''
  plt.figure(figsize=(8, 6))
  plt.scatter(ground_truth[:, 0], ground_truth[:, 1], label='Ground Truth (Unordered)', c='blue', marker='o')
  plt.scatter(predicted[:, 0], predicted[:, 1], label='Predicted (Unordered)', c='red', marker='x')
  plt.scatter(predicted_aligned[:, 0], predicted_aligned[:, 1], label='Predicted (Aligned)', c='green', marker='+')
  plt.legend()
  plt.title(f"Alignment with Combined Ordering\nSimilarity Disparity: {similarity_disparity:.4f}")
  plt.xlabel("X-axis")
  plt.ylabel("Y-axis")
  plt.axis('equal')
  plt.show()

# ======================================== Interface ======================================== 
def ilyc_interface():
    '''
    Interface for program.
    Prompts user to input image filepath or use camera option
    Then, applies preprocessing to bring size to (250, 300, 3) and remove background
    '''
    instructions = '''
    For best results, please adhere to the following guidelines:
      1) Image must be front facing
      2) Image must be mostly your face
      3) Preferrably no smiling
    '''
    print(instructions)
    image_mode = input("Select input mode. Press 'c' to use your computer camera, 'f' to enter a filepath, or 'q' to quit: ")
    if image_mode == 'c':
        filepath = 'camera_out.jpg'
        img = take_image_from_camera(filepath)
    elif image_mode == 'f':
        # NOTE: currently crashes if user quits while in camera mode :/
        filepath = input("Enter your filepath: ")
        print(filepath)
        print(type(filepath))
        img = take_image_from_filepath(filepath)
    elif image_mode == 'q':
        return
    else:
        ilyc_interface()
        return

    # Preview image and prompt for validity
    plt.imshow(img)
    plt.show()
    # sleep(2)  # there's some delay for showing the image --> this accounts for it
    valid_input = input("Is this a valid picture? (y/n) ")
    if valid_input.lower() != 'y':
        ilyc_interface()
        return

    # TODO: connect to Adam's face shape, prompt user for hairstyle
    hair_selection = "fade"  # TODO: replace with Adam's stuff
    hair_points_file = './asm_data/frontalshapes_manuallyannotated_46points/23a.pts'  # TODO: link hair_selection to ASM of that hair style
    hair_points = load_pts_file(hair_points_file)  # TODO: replace points_path with the hairstyle points

    # Apply model to generate predicted ASM on inputted image
    resized_image, predicted_points = asm_predict(filepath, show=False)

    # Make the hair points align with the predicted points
    aligned_points, similarity_disparity, homography_matrix = align_points(predicted_points, hair_points)

    # OPTIONAL: Plots (plots upside down because origin is at the bottom left but in images its at the top left)
    # plot_transformations(ground_truth_points, predicted_points, predicted_aligned, similarity_disparity)

    # Visualize
    # NOTE: first function call is with the homography stuff, second is without
    visualize_keypoints(resized_image, aligned_points, hair_points, resize=False)
    visualize_keypoints(resized_image, predicted_points, hair_points, resize=False)


def main():
    ilyc_interface()

if __name__ == "__main__":
    main()