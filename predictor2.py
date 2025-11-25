import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import os

saved_model = load_model("bonefracture.h5")


def check(input_img):

    print("your image is:", input_img)

    img_path = os.path.join("images", input_img)

    img = tf.keras.utils.load_img(
        img_path,
        target_size=(180, 180)
    )

    img_array = keras.preprocessing.image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = img_array.reshape(1, 180, 180, 3)

    print(img_array.shape)

    output = saved_model.predict(img_array)

    print("Prediction:", output)

    if output[0] > 0.5:
        status = False
    else:
        status = True

    return status