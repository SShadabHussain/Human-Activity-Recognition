import os

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from django.core.wsgi import get_wsgi_application


# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

# application = get_wsgi_application()

# from django.contrib.auth import authenticate


# def check_password():
#     """Returns `True` if the user had a correct password."""

#     def password_entered():
#         """Checks whether a password entered by the user is correct."""
#         user = authenticate(
#             username=st.session_state["username"], password=st.session_state["password"]
#         )

#         if user is not None:
#             st.session_state["password_correct"] = True
#             del st.session_state["password"]  # don't store username + password
#             del st.session_state["username"]
#         else:
#             st.session_state["password_correct"] = False

#     if "password_correct" not in st.session_state:
#         # First run, show inputs for username + password.
#         st.text_input("Username", on_change=password_entered, key="username")
#         st.text_input(
#             "Password", type="password", on_change=password_entered, key="password"
#         )
#         return False
#     elif not st.session_state["password_correct"]:
#         # Password not correct, show input + error.
#         st.text_input("Username", on_change=password_entered, key="username")
#         st.text_input(
#             "Password", type="password", on_change=password_entered, key="password"
#         )
#         st.error("😕 User not known or password incorrect")
#         return False
#     else:
#         # Password correct.
#         return True


# if check_password():

st.title("Image Classification")
upload_file = st.sidebar.file_uploader("Upload Images", type="jpg")
generate_pred = st.sidebar.button("Predict")
model = tf.keras.models.load_model("HAR_model.h5")


def import_n_pred(image_data, model):
        size = (224, 224)
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        img = np.asarray(image)
        reshape = img[np.newaxis, ...]
        pred = model.predict(reshape)
        return pred


if generate_pred:
        image = Image.open(upload_file)
        with st.expander("image", expanded=True):
            st.image(image, use_column_width=True)
        pred = import_n_pred(image, model)
        labels = [
            "calling",
            "hugging",
            "laughing",
            "texting",
            "using_laptop",
            "clapping",
            "drinking",
            "sleeping",
            "eating",
            "sitting",
            "running",
            "listening_to_music",
            "dancing",
            "cycling",
            "fighting",
        ]
        st.title("prediction of image is {}".format(labels[np.argmax(pred)]))
