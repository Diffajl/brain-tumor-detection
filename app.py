import streamlit as st
import numpy as np
from PIL import Image
import io
import tensorflow as tf

model = tf.keras.models.load_model('tumor_otak.h5')

def load_uploaded_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((224, 224))  
    img_array = np.array(img)
    img_array = img_array / 255.0 
    return img_array

def predict_image(image_bytes):
    img_array = load_uploaded_image(image_bytes)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        return "Tumor detected"
    else:
        return "No tumor detected"

st.title("Brain Tumor Detection using CNN")

uploaded_file = st.file_uploader("Upload a brain MRI image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    img = Image.open(io.BytesIO(image_bytes))
    st.image(img, caption="Uploaded Brain MRI Image", use_column_width=True)

    # Predict the result
    result = predict_image(image_bytes)
    st.write("Prediction: ", result)
