from flask import Flask, request
from PIL import Image
import io
import base64
# Let's use sys.path to import the inference script from the machine_learning directory
import sys
sys.path.append('src/machine_learning/model1')
import inference  # Your inference script

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return {'message': 'Server is running'}
@app.route('/predict', methods=['POST'])
def predict():
    base64_image_data = request.json['image']  # Get base64 image data from POST request
    image_data = base64.b64decode(base64_image_data)  # Decode base64 string back into bytes
    image = Image.open(io.BytesIO(image_data))  # Convert image data to PIL Image

    outputs = inference.predict_with_image_obj(image)  # Run your inference script
    predicted_values = inference.parse_outputs(outputs)
    return {'predicted_values': predicted_values}
if __name__ == "__main__":
    app.run(debug=True)