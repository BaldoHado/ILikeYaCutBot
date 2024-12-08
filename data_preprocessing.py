import cv2
import io
import imageio.v3
import keyboard
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from rembg import remove

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
        elif keyboard.is_pressed("c"):
            captured_frame = resize(
                frame_rgb, (128, 128, 3), mode="reflect", anti_aliasing=True
            ).astype("float32")
            captured_frame = (captured_frame * 255).astype(
                np.uint8
            )  # Convert back to uint8
            print(f"Image captured and saved to {output_file}")
            break
        else:
            continue

    # Save the captured frame if available
    if captured_frame is not None:
        cv2.imwrite(output_file, captured_frame)

    # Release the camera
    camera.release()
    plt.close()
    return captured_frame