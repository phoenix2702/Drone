import os
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import tflite_runtime.interpreter as tflite

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load the TensorFlow Lite model
MODEL_PATH = "model.tflite"  # Ensure this model exists in your directory
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def process_image(filepath):
    """Preprocess the image and run the model inference."""
    img = cv2.imread(filepath)
    if img is None:
        return None, "Error: Image could not be loaded. Check file format."

    img = cv2.resize(img, (224, 224))  # Resize to model's expected input
    img = np.expand_dims(img, axis=0).astype(np.float32)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = output_data[0][0]

    result_text = "🚨 Obstacle detected! Avoid it!" if prediction > 0.5 else "✅ No obstacle detected."
    return result_text, None  # Return result and no error


@app.route("/", methods=["GET", "POST"])
def upload_file():
    """Handles file upload and redirects to result."""
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Process image and get result
        result_text, error = process_image(filepath)
        if error:
            return error

        return redirect(url_for("result", filename=filename, result=result_text))

    return render_template("index.html")


@app.route("/result", methods=["GET", "POST"])
def result():
    """Displays result in HTML for browser and plain text for Kivy."""
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400

        file = request.files["file"]
        if file.filename == "":
            return "Empty filename", 400

        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Process image
        result_text, error = process_image(filepath)
        if error:
            return error, 400

        # ✅ If Kivy requests, return plain text
        if request.headers.get("User-Agent") == "Kivy":
            return result_text  # Kivy expects a plain text response

        # ✅ If browser requests, return HTML
        return render_template("result.html", filename=filename, result=result_text)

    # Handle GET request (browser viewing results)
    filename = request.args.get("filename", "")
    result_text = request.args.get("result", "")
    return render_template("result.html", filename=filename, result=result_text)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
