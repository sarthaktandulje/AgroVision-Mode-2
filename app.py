from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Paths
MODEL_PATH = "model/AgroVision_model.h5"   # relative path
UPLOAD_FOLDER = "static/uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model safely
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Class names
CLASS_NAMES = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus'
]

# Preventive measures
PREVENTION = {
    'Tomato___Bacterial_spot': "Remove infected leaves, avoid overhead watering, use copper-based fungicides.",
    'Tomato___Early_blight': "Use disease-free seeds, rotate crops, and apply fungicides like chlorothalonil.",
    'Tomato___Late_blight': "Destroy infected plants, avoid water on leaves, and use resistant varieties.",
    'Tomato___Leaf_Mold': "Ensure good air circulation, avoid wetting leaves, and apply sulfur sprays.",
    'Tomato___Septoria_leaf_spot': "Remove infected debris, water at the base, and use fungicides like mancozeb.",
    'Tomato___Spider_mites Two-spotted_spider_mite': "Spray neem oil or insecticidal soap; maintain humidity.",
    'Tomato___Target_Spot': "Avoid overhead watering, remove infected leaves, and apply preventive fungicides.",
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Control whiteflies, use resistant varieties, and remove infected plants.",
    'Tomato___Tomato_mosaic_virus': "Disinfect tools, avoid tobacco use near plants, and use resistant cultivars."
}

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        return f"Template error: {e}"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded!"})

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded!"})

    file = request.files['file']
    if file.filename == "":
        return jsonify({"error": "Empty filename!"})

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.root_path, UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array)
        confidence = np.max(predictions)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]

        if confidence < 0.55:
            return jsonify({
                "disease": "Unknown or Not a Tomato Leaf 🧐",
                "prevention": "Try uploading a clearer image of a tomato leaf."
            })

        prevention = PREVENTION.get(predicted_class, "No prevention info available.")
        return jsonify({
            "disease": predicted_class,
            "prevention": prevention,
            "confidence": f"{confidence*100:.2f}%"
        })
    except Exception as e:
        return jsonify({"error": f"Prediction error: {e}"})

if __name__ == "__main__":
    app.run(debug=True)
