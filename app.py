from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# -------------------------------------------------------------
# 🧠 Model and Paths
# -------------------------------------------------------------
MODEL_PATH = "model/AgroVision_model.h5"
# Auto-download model from Google Drive if not found or corrupted
if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 100000:
    import gdown
    print("⚠️ Model not found or incomplete — downloading from Drive...")
    os.makedirs("model", exist_ok=True)
    drive_url = "https://drive.google.com/uc?id=1ITThppMgw-CNIVkCjd8VAlkrDbgiPazt" # your model's Drive link
    gdown.download(drive_url, MODEL_PATH, quiet=False)

UPLOAD_FOLDER = "static/uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------------------------------------------
# ⚙️ Load the Trained Model
# -------------------------------------------------------------
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# -------------------------------------------------------------
# 🌱 Class Names (must match training order)
# -------------------------------------------------------------
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

# -------------------------------------------------------------
# 🧬 Deep Preventive Measures
# -------------------------------------------------------------
PREVENTION = {
    'Tomato___Bacterial_spot': (
        "🦠 **Bacterial Spot:** Caused by *Xanthomonas* bacteria. Remove infected leaves and avoid overhead watering. "
        "Use copper-based fungicides weekly in humid weather and sanitize your tools with a 10% bleach solution."
    ),

    'Tomato___Early_blight': (
        "🌿 **Early Blight:** Fungal disease caused by *Alternaria solani*. Rotate crops every 2–3 years. "
        "Spray fungicides like chlorothalonil or mancozeb at first signs. Ensure good airflow and remove lower leaves as they age."
    ),

    'Tomato___Late_blight': (
        "🌧️ **Late Blight:** The same fungus that caused the Irish Potato Famine — *Phytophthora infestans*. "
        "Destroy infected plants immediately. Avoid wet leaves, and spray copper-based fungicides preventively every 7 days."
    ),

    'Tomato___Leaf_Mold': (
        "🍃 **Leaf Mold:** Loves humidity! Caused by *Passalora fulva*. Improve ventilation, water early mornings, "
        "and use sulfur-based fungicides. Remove old leaves near soil level."
    ),

    'Tomato___Septoria_leaf_spot': (
        "🌱 **Septoria Leaf Spot:** Caused by *Septoria lycopersici*. Remove infected leaves quickly. "
        "Water only at the base, and apply mancozeb fungicide every 7–10 days during rainy periods."
    ),

    'Tomato___Spider_mites Two-spotted_spider_mite': (
        "🕷️ **Spider Mites:** Tiny sap-sucking insects that love heat. Increase humidity, spray neem oil or insecticidal soap, "
        "and introduce ladybugs or predatory mites as natural defenders."
    ),

    'Tomato___Target_Spot': (
        "🎯 **Target Spot:** Caused by *Corynespora cassiicola*. Prune for airflow, avoid wetting leaves, "
        "and spray preventive fungicides like difenoconazole or azoxystrobin when humidity rises."
    ),

    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': (
        "🦟 **TYLCV:** Spread by whiteflies! Use yellow sticky traps, neem oil, or imidacloprid. "
        "Remove infected plants immediately, and always plant resistant varieties."
    ),

    'Tomato___Tomato_mosaic_virus': (
        "🧫 **Tomato Mosaic Virus:** Extremely contagious. Disinfect tools with bleach and avoid tobacco near plants. "
        "Use resistant varieties and wash hands before handling plants. Never compost infected material."
    )
}

# -------------------------------------------------------------
# 🌍 Routes
# -------------------------------------------------------------
@app.route('/')
def home():
    """Render home page (upload form)."""
    return render_template('index.html')


@app.route('/result')
def result():
    """Render results page if needed."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image prediction requests."""
    if model is None:
        return jsonify({"error": "Model not loaded!"})

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded!"})

    file = request.files['file']
    if file.filename == "":
        return jsonify({"error": "Empty filename!"})

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        # 🔍 Preprocess the image
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # 🧠 Predict
        predictions = model.predict(img_array)
        confidence = float(np.max(predictions))
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        
        print("Predictions:", predictions)
        print("Predicted class index:", np.argmax(predictions))
        print("Predicted class:", predicted_class)
        print("Confidence:", confidence)

        print(f"Predicted: {predicted_class} | Confidence: {confidence:.2f}")

        # Low confidence safeguard
        if confidence < 0.10:
            return jsonify({
                "disease": "Unknown or Not a Tomato Leaf 🧐",
                "prevention": "Try uploading a clearer image of a tomato leaf."
            })

        # Return the result
        return render_template(
            "result.html",
            disease=predicted_class,
            prevention=PREVENTION.get(predicted_class, "No prevention info available.")
        )

    except Exception as e:
        return jsonify({"error": f"Prediction error: {e}"})


# -------------------------------------------------------------
# 🚀 App Runner
# -------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
