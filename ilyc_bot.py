import cv2
import numpy as np
import matplotlib.pyplot as plt
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
        img, img_no_bg = take_image_from_camera(filepath)
    elif image_mode == "f":
        # NOTE: currently crashes if user quits while in camera mode :/
        filepath = input("Enter your filepath: ")
        img = take_image_from_filepath(filepath)
        img_no_bg = img
    elif image_mode == "q":
        return
    else:
        ilyc_interface()
        return

    # Preview image after removing background and resizing; prompt for validity
    plt.imshow(img_no_bg)
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

    # Reprompt until user quits
    while True:
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
        hair_image = cv2.imread(f"./hair_templates/{hair_selection}_hair.png", flags=cv2.IMREAD_UNCHANGED)
        hair_image = cv2.cvtColor(hair_image, cv2.COLOR_BGRA2RGBA)
        hair_points = load_pts_file(hair_points_file)

        # Apply model to generate predicted ASM on inputted image
        resized_image, predicted_points = asm_predict(filepath, show=False)
        with open("points.txt", "a") as f:
            f.write(str(predicted_points))

        # Normalize points for transformation
        image_width, image_height = resized_image.shape[1], resized_image.shape[0]
        predicted_points /= np.array([image_width, image_height])
        hair_points /= np.array([image_width, image_height])

        # Apply transformations
        transformed_hair, aligned_hair, similarity_disparity = align_hair(hair_image, predicted_points, hair_points)

        # OPTIONAL: Plots (plots upside down because origin is at the bottom left but in images its at the top left)
        plot_transformations(
            hair_points, aligned_hair, predicted_points, similarity_disparity
        )

        # Visualize
        # NOTE: first function call just shows the hair on the face, second includes keypoints
        visualize_keypoints(
            resized_image,
            predicted_points,
            aligned_hair,
            image2=transformed_hair,
            label1="Predicted",
            label2="Aligned Hair",
            resize=True,
            show=False
        )

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

        reprompt = input("Try another hair? 'y' to continue, 'n' to input a new image, and anything else to quit: ").strip().lower()
        if reprompt == 'y':
            continue
        elif reprompt == 'n':
            return ilyc_interface()
        else:
            return


def main():
    ilyc_interface()


if __name__ == "__main__":
    main()
