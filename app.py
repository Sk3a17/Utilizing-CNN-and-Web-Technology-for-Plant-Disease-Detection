import os
from flask import Flask, request, render_template, url_for
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import urllib.parse

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('/Users/shikhar/Desktop/FINALPROJECT/mymodel.h5')

# Define class labels
class_labels = ['Apple_Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_Healthy', 'Cotton_Army_worm', 'Cotton_Bacterial_blight', 'Cotton_Healthy', 'Cotton_Target_spot', 'Grape_Black_Measles', 'Grape_Healthy', 'Grape_Leaf_blight', 'Potato_Early_Blight', 'Potato_Healthy', 'Potato_Late_Blight', 'Sugarcane_Healthy', 'Sugarcane_RedRot', 'Sugarcane_Rust', 'Sugarcane_Yellow', 'Tomato_Early_blight', 'Tomato_healthy', 'Tomato_Late_Blight', 'Tomato_Leaf_Mold', 'Tomato_mosaic_virus', 'Tomato_Target_Spot', 'Tomato_YellowLeaf_Curl_Virus']

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



def get_symptoms_and_prevention(predicted_class_label):
    symptoms = ""
    prevention = ""

    if predicted_class_label == 'Apple_Apple_scab':
        symptoms = "Dark, scabby spots on leaves and fruit, causing them to become deformed."
        prevention = "Apply fungicides early in the growing season, remove infected leaves, and maintain good air circulation around the trees."
    elif predicted_class_label == 'Apple_Black_rot':
        symptoms = "Brownish-black lesions on fruit, causing them to shrivel and become mummified."
        prevention = "Remove infected fruit, prune trees to improve air circulation, and apply fungicides during the growing season."
    elif predicted_class_label == 'Apple_Cedar_apple_rust':
        symptoms = "Yellow spots on upper surfaces of leaves, orange pustules on undersides, leading to leaf drop."
        prevention = "Remove nearby cedar trees, apply fungicides before rust symptoms appear, and keep the area clean."
    elif predicted_class_label == 'Apple_Healthy':
        symptoms = "No visible symptoms."
        prevention = "Regularly inspect plants for any signs of diseases or pests and maintain proper care."
    elif predicted_class_label == 'Cotton_Army_worm':
        symptoms = "Defoliation, chewing damage on leaves, fruit, and stems."
        prevention = "Plant early, use resistant varieties, implement cultural control methods, and apply insecticides if necessary."
    elif predicted_class_label == 'Cotton_Bacterial_blight':
        symptoms = "Angular water-soaked lesions on leaves, stems, and bolls, leading to defoliation and yield loss."
        prevention = "Rotate crops, use disease-free seeds, avoid overhead irrigation, and apply copper-based fungicides."
    elif predicted_class_label == 'Cotton_Healthy':
        symptoms = "No visible symptoms."
        prevention = "Regularly inspect plants for any signs of diseases or pests and maintain proper care."
    elif predicted_class_label == 'Cotton_Target_spot':
        symptoms = "Circular to angular spots on leaves, with concentric rings, leading to defoliation."
        prevention = "Avoid planting susceptible varieties, implement crop rotation, and apply fungicides if necessary."
    elif predicted_class_label == 'Grape_Black_Measles':
        symptoms = "Dark brown to black circular lesions on leaves, leading to leaf drop and reduced fruit quality."
        prevention = "Prune infected canes, apply fungicides before bloom, and manage vineyard to reduce humidity."
    elif predicted_class_label == 'Grape_Healthy':
        symptoms = "No visible symptoms."
        prevention = "Regularly inspect plants for any signs of diseases or pests and maintain proper care."
    elif predicted_class_label == 'Grape_Leaf_blight':
        symptoms = "Irregularly shaped brown lesions on leaves, leading to defoliation and reduced yield."
        prevention = "Prune infected canes, apply fungicides during the growing season, and maintain good airflow."
    elif predicted_class_label == 'Potato_Early_Blight':
        symptoms = "Dark brown lesions with yellow margins on lower leaves, leading to defoliation."
        prevention = "Rotate crops, space plants properly, remove infected leaves, and apply fungicides."
    elif predicted_class_label == 'Potato_Healthy':
        symptoms = "No visible symptoms."
        prevention = "Regularly inspect plants for any signs of diseases or pests and maintain proper care."
    elif predicted_class_label == 'Potato_Late_Blight':
        symptoms = "Irregularly shaped brown lesions on leaves, stems, and tubers, leading to rapid plant death."
        prevention = "Avoid overhead irrigation, remove infected plants, apply fungicides preventatively."
    elif predicted_class_label == 'Sugarcane_Healthy':
        symptoms = "No visible symptoms."
        prevention = "Regularly inspect plants for any signs of diseases or pests and maintain proper care."
    elif predicted_class_label == 'Sugarcane_RedRot':
        symptoms = "Reddish discoloration at the base of the stem, internal decay, stunted growth, and yield loss."
        prevention = "Plant disease-free seedlings, remove infected plants, and manage soil moisture."
    elif predicted_class_label == 'Sugarcane_Rust':
        symptoms = "Small yellow-orange pustules on leaves, leading to premature leaf shedding and yield loss."
        prevention = "Plant resistant varieties, remove infected leaves, and apply fungicides."
    elif predicted_class_label == 'Sugarcane_Yellow':
        symptoms = "Yellowing of leaves, stunted growth, and reduced sugar content in the stalks."
        prevention = "Use disease-free seedlings, control aphid populations, and maintain proper field hygiene."
    elif predicted_class_label == 'Tomato_Early_blight':
        symptoms = "Dark brown lesions with concentric rings on lower leaves, leading to defoliation."
        prevention = "Rotate crops, space plants properly, remove infected leaves, and apply fungicides."
    elif predicted_class_label == 'Tomato_healthy':
        symptoms = "No visible symptoms."
        prevention = "Regularly inspect plants for any signs of diseases or pests and maintain proper care."
    elif predicted_class_label == 'Tomato_Late_Blight':
        symptoms = "Dark brown lesions with greasy texture on leaves and stems, leading to rapid plant death."
        prevention = "Avoid overhead irrigation, remove infected plants, and apply fungicides preventatively."
    elif predicted_class_label == 'Tomato_Leaf_Mold':
        symptoms = "Yellowing and distortion of leaves with fuzzy gray or brown patches."
        prevention = "Increase air circulation, avoid overhead watering, and apply fungicides preventatively."
    elif predicted_class_label == 'Tomato_mosaic_virus':
        symptoms = "Mottled or distorted leaves, stunted growth, and reduced fruit quality."
        prevention = "Use virus-free seeds, control insect vectors, and remove infected plants."
    elif predicted_class_label == 'Tomato_Target_Spot':
        symptoms = "Circular lesions with concentric rings on leaves, leading to defoliation."
        prevention = "Rotate crops, space plants properly, remove infected leaves, and apply fungicides."
    elif predicted_class_label == 'Tomato_YellowLeaf_Curl_Virus':
        symptoms = "Yellowing and curling of leaves, stunted growth, and reduced fruit yield."
        prevention = "Control whitefly populations, use resistant varieties, and remove infected plants."

    return symptoms, prevention



# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/code')
def code():
    return render_template('code.html')


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
            # Ensure that the 'uploads' folder exists in the 'static' directory, create if necessary
            uploads_folder = os.path.join(app.root_path, 'static', 'uploads')
            if not os.path.exists(uploads_folder):
                os.makedirs(uploads_folder)

            # Save the uploaded file to the uploads folder
            filename = file.filename
            # Ensure that the filename is properly URL encoded
            filename = urllib.parse.quote(filename)
            file_path = os.path.join(uploads_folder, filename)
            file.save(file_path)

            # Make prediction
            prediction = predict_image(file_path)

            symptoms, prevention = get_symptoms_and_prevention(prediction)

            # Pass the file path to the HTML template
            uploaded_file = url_for('static', filename='uploads/' + filename)
            return render_template('index.html', prediction=prediction, uploaded_file=uploaded_file, symptoms=symptoms, prevention=prevention)


if __name__ == '__main__':
    app.run(debug=True, port=5001)
