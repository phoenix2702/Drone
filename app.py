import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the trained model
MODEL_PATH = "obstacle_detector.h5"  # Ensure this model exists in your directory
model = tf.keras.models.load_model(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            # Save the uploaded file
            file.save(filepath)

            # Read and preprocess the image
            img = cv2.imread(filepath)
            if img is None:
                return "Error: Image could not be loaded. Check file format."

            img = cv2.resize(img, (224, 224)) / 255.0
            img = np.expand_dims(img, axis=0)

            # Predict using the model
            prediction = model.predict(img)[0][0]
            result = "🚨 Obstacle detected! Avoid it!" if prediction > 0.5 else "✅ No obstacle detected."

            return redirect(url_for("result", filename=filename, result=result))

    return render_template("index.html")

@app.route("/result/<filename>/<result>")
def result(filename, result):
    return render_template("result.html", filename=filename, result=result)

if __name__ == "__main__":
    app.run(debug=True)
