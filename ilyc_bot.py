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
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Lambda
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from scipy.spatial import procrustes
import keras
import skimage
from typing import Literal
from textwrap import dedent
from data_preprocessing import *
from face_shape_classifier import *
from asm_processing import *

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
    gender = ""
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
    hair_selection = ""
    while hair_selection not in hair_style_recomm["recommendations"]:
        hair_selection = input(
            f"Pick one of the following to apply ({', '.join(hair_style_recomm['recommendations'])}): "
        ).lower()
    hair_selection = "_".join(hair_selection.split())
    hair_points_file = f"./hair_points/{hair_selection}.pts" 
    print("Selection:", f"./hair_templates/{hair_selection}_hair.png")
    hair_image = cv2.imread(f"./hair_templates/{hair_selection}_hair.png", flags=cv2.IMREAD_UNCHANGED)
    hair_image = cv2.cvtColor(hair_image, cv2.COLOR_BGRA2RGBA)
    hair_points = load_pts_file(hair_points_file)

    # Apply model to generate predicted ASM on inputted image
    resized_image, predicted_points = asm_predict(filepath, show=False)
    with open("points.txt", "a") as f:
        f.write(str(predicted_points))

    image_width, image_height = resized_image.shape[1], resized_image.shape[0]
    predicted_points /= np.array([image_width, image_height])
    hair_points /= np.array([image_width, image_height])

    # Make the hair points align with the predicted points
    aligned_hair, similarity_disparity, affine_matrix = align_points_affine(
        predicted_points, hair_points
    )
    print(affine_matrix)


    # Apply affine transformation
    output_size = (image_width, image_height)
    hair_image = cv2.resize(hair_image, (250, 300)) / 255.0
    transformed_hair = recenter_warped_image(hair_image, affine_matrix, output_size)

    # plt.imshow(hair_image)
    # plt.show()
    # plt.imshow(transformed_hair)
    # plt.show()

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
        image2=transformed_hair,
        label1="Predicted",
        label2="Aligned Hair",
        resize=True,
        show=True
    )
    return ilyc_interface()


def main():
    # TESTING: ./asm_data/frontalimages_spatiallynormalized/190b.jpg
    ilyc_interface()
    # get_face_shape_model()


if __name__ == "__main__":
    main()
