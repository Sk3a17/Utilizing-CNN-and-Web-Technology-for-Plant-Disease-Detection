import os
from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the pre-trained model
model = tf.keras.models.load_model('/Users/shikhar/Desktop/FINALPROJECT/mymodel.h5')

# Define class labels
class_labels = ['Apple_Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_Healthy', 'Cotton_Army_worm',
                'Cotton_Bacterial_blight', 'Cotton_Healthy', 'Cotton_Target_spot', 'Grape_Black_Measles',
                'Grape_Healthy', 'Grape_Leaf_blight', 'Potato_Early_Blight', 'Potato_Healthy', 'Potato_Late_Blight',
                'Sugarcane_Healthy', 'Sugarcane_RedRot', 'Sugarcane_Rust', 'Sugarcane_Yellow', 'Tomato_Early_blight',
                'Tomato_healthy', 'Tomato_Late_Blight', 'Tomato_Leaf_Mold', 'Tomato_mosaic_virus', 'Tomato_Target_Spot',
                'Tomato_YellowLeaf_Curl_Virus']


# Function to preprocess the image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(256, 256))  # Adjust target_size according to your model input shape
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Function to make predictions
def predict_image(image_path):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label


# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')


# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', prediction="No file part")

        file = request.files['file']

        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', prediction="No selected file")

        if file:
            # Save the uploaded file to a temporary location
            filename = file.filename
            file_path = os.path.join('uploads', filename)
            file.save(file_path)

            # Make prediction
            prediction = predict_image(file_path)
            return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
