import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the model
model = tf.keras.models.load_model("pneumonia_model_partial.h5")

# Function to predict pneumonia
def predict_pneumonia(img):
    try:
        # Preprocess the image for the model
        img = img.resize((224, 224))  # Resize the image to 224x224 pixels to match model's expected input size
        img_array = np.array(img)  # Convert the image to a numpy array
        img_array = img_array.astype('float32')  # Ensure the data type is float32
        img_array = img_array / 255.0  # Normalize the image values between 0 and 1
        img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension to the image
        
        # Get the model's prediction
        prediction = model.predict(img_array)
        
        # Return the result
        if prediction[0] > 0.5:
            return "Pneumonia Detected"
        else:
            return "No Pneumonia"
    except Exception as e:
        return f"Error: {e}"  # Return the error if something goes wrong

# Create Gradio Interface
interface = gr.Interface(fn=predict_pneumonia,
                         inputs=gr.Image(type="pil"),
                         outputs="text",
                         live=True)

# Launch the interface
interface.launch(share=True)
