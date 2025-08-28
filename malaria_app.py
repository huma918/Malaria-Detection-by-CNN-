import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image
import matplotlib.pyplot as plt

# CSS Styling
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .uploaded-image {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%;
    }
    .prediction {
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Model loading in Cache
@st.cache_resource
def load_keras_model():
    model = load_model('malaria_detection_model_keras.h5')
    return model

model = load_keras_model()

# Image tranformation
def transform_image(image):
    image = image.resize((150, 150))
    img_array = keras_image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Application
st.title("Malaria Detection in Blood Cell Images")

tab1, tab2 = st.tabs(["Single Image Prediction", "Bulk Image Prediction"])

## Single Image Prediction
with tab1:
    st.header("Upload a Blood Cell Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        ## Loading Image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)

        ## Tranforming Image
        img_array = transform_image(image)

        ## Model Prediction
        pred_probabilities = model.predict(img_array).flatten()
        prediction = (pred_probabilities > 0.5).astype(np.int32)[0]

        ## Model Result
        if prediction == 1:
            st.markdown('<div class="prediction"><span style="color:#4bff4b;">Uninfected</span>.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction"><span style="color:#ff4b4b;">Parasitized</span> with Malaria.</div>', unsafe_allow_html=True)

## Multi Image Prediction
with tab2:
    st.header("Upload Multiple Blood Cell Images")
    ## Multi Image Upload
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "png"], accept_multiple_files=True)

    if uploaded_files is not None:
        uninfected_count = 0
        parasitized_count = 0

        ## Loading Image one by one
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            img_array = transform_image(image)

            ## Predicting
            pred_probabilities = model.predict(img_array).flatten()
            prediction = (pred_probabilities > 0.5).astype(np.int32)[0]

            ## Count Of Prediction
            if prediction == 1:
                uninfected_count += 1
            else:
                parasitized_count += 1

        st.write(f"Uninfected: {uninfected_count}")
        st.write(f"Parasitized: {parasitized_count}")

        labels = ['Uninfected', 'Parasitized']
        counts = [uninfected_count, parasitized_count]

        fig, ax = plt.subplots()
        ax.bar(labels, counts, color=['#4bff4b', '#ff4b4b'])
        ax.set_xlabel('Category')
        ax.set_ylabel('Count')
        ax.set_title('Count of Predictions for Multiple Images')

        st.pyplot(fig)
