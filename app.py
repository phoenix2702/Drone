import os
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
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

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Read and preprocess the image
        img = cv2.imread(filepath)
        if img is None:
            return "Error: Image could not be loaded. Check file format."

        img = cv2.resize(img, (224, 224))  # Resize to match model's expected input
        img = np.expand_dims(img, axis=0).astype(np.float32)

        # Set the tensor to the input data
        interpreter.set_tensor(input_details[0]['index'], img)

        # Run inference
        interpreter.invoke()

        # Retrieve the output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = output_data[0][0]
        result = "🚨 Obstacle detected! Avoid it!" if prediction > 0.5 else "✅ No obstacle detected."

        # Redirect to the result route with filename and result as query parameters
        return redirect(url_for("result", filename=filename, result=result))

    return render_template("index.html")

@app.route("/result")
def result():
    filename = request.args.get("filename", "")
    result = request.args.get("result", "")
    return render_template("result.html", filename=filename, result=result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
