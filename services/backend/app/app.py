from flask import Flask, request, jsonify, send_file
from PIL import Image
import io
import os
import datetime as datatime

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "./static/images"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

def generate_image(input_image):
    generated_image = input_image  # Placeholder code
    return generated_image

# Define the route for image generation
@app.route("/generate", methods=["POST"])
def generate():
    if "input_image" not in request.files:
        return jsonify({"error": "No input image found."}), 400

    file = request.files["input_image"]
    if file and allowed_file(file.filename):
        input_image = Image.open(file)

        # input_image_path = os.path.join(app.config["UPLOAD_FOLDER"], "input_image.png")
        # input_image.save(input_image_path)

        generated_image = generate_image(input_image)

        # generated_image_path = os.path.join(app.config["UPLOAD_FOLDER"], "generated_image.png")
        # generated_image.save(generated_image_path)

        generated_image_io = io.BytesIO()
        generated_image.save(generated_image_io, format='PNG')
        generated_image_io.seek(0)

        return send_file(generated_image_io, mimetype='image/png', as_attachment=True, download_name='generated_image.png')

    return jsonify({"error": "Invalid file."}), 400

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

if __name__ == "__main__":
    app.run(debug=True)