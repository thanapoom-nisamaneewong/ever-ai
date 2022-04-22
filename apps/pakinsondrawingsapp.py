import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import tensorflow
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
def predict(img, model_class):
    # check label
    if model_class == 'Spiral':
        model = load_model('models/pakinson/spiral_parkinson_v1.h5')
    else:
        model = load_model('models/pakinson/wave_parkinson_v1.h5')

    img = img.resize((128, 128))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = np.array(img, np.float32) / 255.0

    # predict score
    predict = model.predict(img)

    score = predict[0][0] * 100
    score = round(score, 2)

    result = "Your Parkinson's Disease Risk is {}%".format(score)
    return result


def app():
    st.title("'Parkinson's Disease Prediction")
    st.write("check out this [link](https://ever-parkinson.herokuapp.com)")


