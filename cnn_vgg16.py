import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pickle
import matplotlib.pyplot as plt

# Load models and histories
model_cnn = tf.keras.models.load_model('model_cnn.keras')
model_vgg16 = tf.keras.models.load_model('model_vgg16.keras')

with open('history_cnn.pkl', 'rb') as f:
    history_cnn = pickle.load(f)

with open('history_vgg16.pkl', 'rb') as f:
    history_vgg16 = pickle.load(f)

# Streamlit web app
st.title("Image Classification with CNN and VGG16-based Model")

# Model selection dropdown
model_choice = st.selectbox("Select a model for classification:", ('CNN', 'VGG16'))

# Load the selected model and history based on user choice
model, history = (model_cnn, history_cnn) if model_choice == 'CNN' else (model_vgg16, history_vgg16)

# Image upload and preprocessing
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image to match the input shape of the selected model
    if model_choice == 'CNN':
        # For the CNN model, convert to grayscale and resize
        image = image.convert('L').resize((28, 28))  # Convert to grayscale and resize to 28x28
        image = np.array(image).astype('float32') / 255.0
        image = np.expand_dims(image, axis=-1)  # Add channel dimension
    else:
        # For the VGG16 model, ensure it's RGB and resize
        image = image.convert('RGB').resize((48, 48))  # Ensure image is in RGB and resize to 48x48
        image = np.array(image).astype('float32') / 255.0

    image = np.expand_dims(image, axis=0)  # Add batch dimension for both models
    st.image(image, caption='Resized and normalized image', use_column_width=True)

    # Predict the class of the image
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    class_probabilities = predictions[0]

    # Display prediction results
    st.write(f"Predicted Class: {predicted_class}")
    st.write("Class Probabilities:")
    for i, prob in enumerate(class_probabilities):
        st.write(f"Class {i}: {prob * 100:.2f}%")


    # Plot loss and accuracy graphs
    st.write("Model Training History:")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.set_title('Loss')
    ax1.plot(history['loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.legend()

    ax2.set_title('Accuracy')
    ax2.plot(history['accuracy'], label='Training Accuracy')
    ax2.plot(history['val_accuracy'], label='Validation Accuracy')
    ax2.legend()

    st.pyplot(fig)
