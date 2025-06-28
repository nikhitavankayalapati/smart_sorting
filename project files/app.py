from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = tf.keras.models.load_model("fruit_freshness_mobilenetv2.h5")
class_labels = ['fresh', 'rotten']
IMG_SIZE = (224, 224)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)[0]
    class_index = np.argmax(predictions)
    confidence = predictions[class_index]
    return class_labels[class_index], confidence

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    file_url = None

    if request.method == "POST":
        file = request.files["file"]
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            label, conf = predict_image(file_path)
            prediction = label
            confidence = f"{conf * 100:.2f}%"
            file_url = file_path

    return render_template("index.html", prediction=prediction, confidence=confidence, file_url=file_url)

if __name__ == "__main__":
    app.run(debug=True)
