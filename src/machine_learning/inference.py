import train
from train import Predictor
from train import os

current_script_path = os.path.dirname(os.path.abspath(__file__))
model_save_path = os.path.join(current_script_path, '../../models/predictor_model.pth')

model = Predictor()

model.eval() # Set the model to evaluation mode

model.load_state_dict(train.torch.load(model_save_path))

# Let's create a function that will take in an image and return the predicted values:
def predict(image_path):
    # First, we need to preprocess the image:
    image = train.Image.open(image_path)
    image = train.transform(image)
    image = train.torch.unsqueeze(image, 0) # Add a batch dimension
    # Now we can pass the image to the model and get the predicted values:
    with train.torch.no_grad():
        outputs = model(image)
    return outputs

predicted = predict("/Users/edwardferrari/Documents/GitHub/RobotDockCenter/src/godot/data_images/image_151.png")
print(predicted)