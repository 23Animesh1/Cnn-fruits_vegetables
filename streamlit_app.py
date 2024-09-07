import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Set the title and favicon that appear in the browser's tab bar.
st.set_page_config(
    page_title='Fruit and Vegetable Classifier',
    page_icon='🍎',  # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_resource
def load_model():
    """Load the pre-trained CNN model."""
    model = tf.keras.models.load_model('/content/drive/MyDrive/Deep learning project/trained_model.h5')
    return model

def preprocess_image(image):
    """Preprocess the image for prediction."""
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img_array = cv2.resize(img_array, (64, 64))  # Resize to the model's input size
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_image(model, img_array):
    """Make a prediction for a single image."""
    predictions = model.predict(img_array)
    result_index = np.argmax(predictions)  # Return index of max element
    return result_index, predictions

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# 🍎 Fruit and Vegetable Classifier

Upload an image of a fruit or vegetable, and this app will classify it!
'''

# Load the pre-trained model
cnn_model = load_model()

# File uploader for image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image and predict
    img_array = preprocess_image(image)
    class_index, prediction = predict_image(cnn_model, img_array)

    # Display prediction result
    st.write(f"Predicted class index: {class_index}")
    
    # Assuming you have a list of class names to map index to name
    class_names = ['Apple', 'Banana', 'Tomato', 'Potato', 'Carrot', 'etc.']  # Add your actual class names here
    st.write(f"Predicted class: {class_names[class_index]}")
    
    # Display prediction probabilities
    st.write("Prediction Probabilities:")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {prediction[0][i] * 100:.2f}%")
