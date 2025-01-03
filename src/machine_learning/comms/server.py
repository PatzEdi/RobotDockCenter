from flask import Flask, request
from PIL import Image
import io
import base64
# Let's use sys.path to import the inference script from the machine_learning directory
import sys
import os

current_script_path = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(current_script_path,"../model1"))
import inference  # Your inference script

inference.load_model(1)  # Load the model

changed_model = False

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return {"message": "Server is running"}
@app.route("/predict", methods=["POST"])
def predict():
    global changed_model
    base64_image_data = request.json["image"] # Get base64 image data from POST request
    image_data = base64.b64decode(base64_image_data) # Decode base64 string to bytes
    image = Image.open(io.BytesIO(image_data)) # Convert image data to PIL Image

    outputs = inference.predict_with_image_obj(image)  # Run your inference script
    predicted_values = inference.parse_outputs(outputs)

    # Here we decide what model to use based on the predicted distance_cline value.
    # If the distance_cline value is within the range of e.g. .5 from 0, we load model2
    if abs(predicted_values[1]) < 0.5 and not changed_model:
        inference.load_model(2)  # Load the model
        changed_model = True
    return {"predicted_values": predicted_values}
if __name__ == "__main__":
    app.run(debug=True)
