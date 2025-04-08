import sys
import torch
import random
import time
import tqdm
import os
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F2
from train import Predictor
from PIL import Image
from torchvision import transforms

current_script_path = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_script_path, "../../godot/data_images")

# For the data
sys.path.append(os.path.join(os.path.dirname(__file__), "../data"))

from target_extraction import get_images_for_each_target

model = Predictor()
# We don't need data augmentation for inference, so we only resize and convert
# to tensor
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

class InferenceTools:

    def load_target_image_data(self, model_num, n_images=-1):

        print("\nLoading target image data...")
        target_1_data, target_2_data = get_images_for_each_target(
            data_dir,
            n_images_cls=n_images
        )

        target_image_data = target_1_data

        if model_num != 1:
            target_image_data = target_2_data

        return target_image_data


    def load_model(self, model_num):
        print("\nLoading model number " + str(model_num) + "...")
        model_save_path = os.path.join(
            current_script_path,
            "../../../models/model" + str(model_num) + ".pth"
        )

        model.load_state_dict(torch.load(model_save_path))
        model.eval() # Set the model to evaluation mode

    # Create a function that will take in an image and return the predicted values:
    def predict(self, image):
        # Let's check if image parameter is a string. If it is, we
        # decode it using PIL, as we assume it is a path.
        # If it isn't, we assume it is already decoded.
        if isinstance(image, str):
            # We need to preprocess the image:
            image = Image.open(image)
        # Transform & add batch dimension
        image = transform(image).unsqueeze(0)
        # Now we can pass the image to the model and get the predicted values:
        with torch.no_grad():
            outputs = model(image)

        return outputs

    # This function is used to extract the outputs from tensors to a normal list.
    def parse_outputs(self, outputs):
        # We access 0th index because outputs have a batch dimension
        predicted_values = outputs.tolist()[0]

        # Mutliply each element by 512 to get the pixel values
        predicted_values = [x * 512 for x in predicted_values]

        return predicted_values

    # A more complete function that returns the parsed predicted values and the
    # time spent on the prediction.
    def predict_single_image(self, image, return_time=False):
        time_start = time.time()
        predicted = self.predict(image)
        parsed = self.parse_outputs(predicted)
        time_spent = None

        if return_time:
            time_spent = time.time() - time_start

        return parsed, time_spent

    # Shuffle images around for more random viewing
    def shuffle_images(self, target_image_data):
        random.shuffle(target_image_data)

# Specifically for the view
class PltInferenceView(InferenceTools):
    def __init__(self, target_image_data, shuffle_images=False, plot_pred=True):
        self.current_index = 0
        self.target_image_data = target_image_data
        self.plot_pred = plot_pred

        if shuffle_images:
            self.shuffle_images(self.target_image_data)


    # Function to update the displayed image
    def update_image(self, ax, index):
        ax.clear()
        image_path = self.target_image_data[index][0]
        real_targets = self.target_image_data[index][1].tolist()
        # Convert to pixels from normalized 0-1 model output
        real_targets = [x * 512 for x in real_targets]
        predicted = self.parse_outputs(self.predict(image_path))
        # Convert using pillow
        image = Image.open(image_path)
        ax.imshow(image, cmap="gray")
        ax.set_title(f"Image {index + 1}/{len(self.target_image_data)}\nReal: {real_targets}\nPredicted: {predicted}")
        ax.axis("off")
        # Plot the predictions
        if self.plot_pred:
            ax.scatter(predicted[0], predicted[1], color="red")

        plt.draw()

    # Key press event handler
    def on_key(self, event):
        if event.key == "right":
            if self.current_index < len(self.target_image_data) - 1:
                self.current_index += 1
                self.update_image(self.ax, self.current_index)
        elif event.key == "left":
            if self.current_index > 0:
                self.current_index -= 1
                self.update_image(self.ax, self.current_index)

    # Main func
    def setup_plt(self):
        # Set up the plot
        self.fig, self.ax = plt.subplots(figsize=(8,8))
        self.update_image(self.ax, self.current_index)

        # Connect the event handler
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        # Show the plot
        plt.show()

if __name__ == "__main__":
    # Let's just manually set the model num for now
    model_num = 1
    inference_tools = InferenceTools()
    # Load some stuff based on model num
    inference_tools.load_model(model_num)
    target_image_data = inference_tools.load_target_image_data(
        model_num,
        n_images=50
    )

    # Instance the PltInferenceView class
    inference_image_viewer = PltInferenceView(
        target_image_data,
        shuffle_images=False
    )
    # Setup and launch
    inference_image_viewer.setup_plt()
