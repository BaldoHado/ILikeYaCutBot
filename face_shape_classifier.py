import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Lambda
from tensorflow.keras.optimizers import Adam
from typing import Literal
from textwrap import dedent

# ======================================== Face Shape Classification Model ========================================

FACE_SHAPE_CLASS_MAPPING = {
    "0": "heart",
    "1": "oblong",
    "2": "oval",
    "3": "round",
    "4": "square",
}

def get_face_shape_model():
    # return keras.saving.load_model(
    #     "face_shape_models/face_shape_identifier.keras",
    #     custom_objects={"lambda": lambda x: x / 128 - 1},
    # )
    model = Sequential(
        [
            Input(shape=(128, 128, 3)),
            Lambda(lambda x: x / 128 - 1),
            Conv2D(32, [3, 3], activation="relu"),
            MaxPooling2D([2, 2], 2),
            Dropout(0.3),
            Conv2D(64, [3, 3], activation="relu"),
            MaxPooling2D([2, 2], 2),
            Dropout(0.3),
            Conv2D(128, [3, 3], activation="relu"),
            MaxPooling2D([2, 2], 2),
            Dropout(0.3),
            Conv2D(256, [3, 3], activation="relu"),
            MaxPooling2D([2, 2], 2),
            Dropout(0.3),
            Conv2D(512, [3, 3], activation="relu"),
            MaxPooling2D([2, 2], 2),
            Dropout(0.3),
            Flatten(),
            Dense(5, activation="softmax"),
        ]
    )
    model.load_weights("fs_weights.weights.h5")
    return model


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
                    "asymmetrical bob",
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
