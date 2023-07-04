import streamlit as st
from PIL import Image
import numpy as np
from tensorflow import keras

# Load the model
model = keras.models.load_model('best_model.h5')

# Define the class-to-denomination mapping
class_to_denomination = {
    0: 10,
    1: 100,
    2: 20,
    3: 200,
    4: 50,
    5: 500
}

# Define the prediction function
def predict_currency(image):
    # Preprocess the image
    image = image.resize((256, 256))
    input_arr = np.array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    input_arr = input_arr / 255.0  # Normalize pixel values

    # Make predictions
    predictions = model.predict(input_arr)
    predicted_class_index = np.argmax(predictions[0])
    predicted_denomination = class_to_denomination[predicted_class_index]
    
    return predicted_denomination

# Streamlit app
st.title("Currency Classification")
st.write("Upload an image and let the model predict the currency denomination.")

uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Predict and display the result
    prediction = predict_currency(image)
    st.write("Predicted denomination:", prediction)
