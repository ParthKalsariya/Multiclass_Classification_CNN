import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("multiclass_cnn_model(Car, Bus or Ship).h5")

# Define class labels
class_labels = ['It\'s a airplane', 'It\'s a car', 'It\'s a ship']

def preprocess_image(img, target_size=(224, 224)):
    """Preprocess the image for model prediction."""
    img = img.resize(target_size)  # Resize image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array

def predict_image(img):
    """Predict the class of the given image."""
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]  # Get class with max probability
    confidence = np.max(predictions)  # Get confidence score
    return predicted_class, confidence

# Streamlit UI
st.set_page_config(page_title="Image Classifier", page_icon="🚀", layout="wide")
# Custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f5f5;
    }
    .header {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #333;
    }
    .subheader {
        text-align: center;
        font-size: 18px;
        color: #555;
    }
    .upload-box {
        border: 2px dashed #aaa;
        padding: 20px;
        text-align: center;
        background-color: #fff;
        border-radius: 10px;
    }
    .result {
        font-size: 22px;
        font-weight: bold;
        color: #2e7d32;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown('<div class="header">🚀 Image Classification (Car, Airplane, Ship) </div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Upload an image and let the AI classify it!</div>', unsafe_allow_html=True)
st.write("")

# Upload Image
# st.markdown('<div class="upload-box">📂 Drag and drop an image here or click to upload</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1])  # Create two columns
    with col1:
        st.image(img, caption="Uploaded Image"
                #  , use_column_width=True
                 )

    # Predict button
    with col2:
        if st.button("🔍 Predict", use_container_width=True):
            predicted_class, confidence = predict_image(img)
            st.markdown(f'<div class="result">✅ Predicted Class: {predicted_class} </div>', unsafe_allow_html=True)
