import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load the trained model
model_path = "C:/Users/jagad/Simple_cnn"
model = load_model(model_path)

# Define the class labels
class_labels = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro']

# Set up the Streamlit app
st.title("Rice Leaf Disease Classification")


# Upload image for classification
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)


    # Preprocess the image for prediction
    img = image.load_img(uploaded_file, target_size=(256,256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize the image
    model=load_model(r"C:\Users\jagad\Simple_cnn")
    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_label = class_labels[predicted_class]

    st.write('Predicted disease:', predicted_label)
