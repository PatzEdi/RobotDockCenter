import sys
import torch
import random
import time
import tqdm
import os
import matplotlib.pyplot as plt
from train import Predictor
from train import transform
from PIL import Image

current_script_path = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_script_path, "../../godot/data_images")

# For the data
sys.path.append(os.path.join(os.path.dirname(__file__), "../data"))

from target_extraction import get_images_for_each_target

model = Predictor()

# Let's just manually set the model num for now
model_num = 1

class InferenceTools:

    def load_target_image_data(self, model_num):

        print("\nLoading target image data...")
        target_1_data, target_2_data = get_images_for_each_target(data_dir)

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
    def predict(self, image_path):
        # First, we need to preprocess the image:
        image = Image.open(image_path)
        # Transform & add batch dimension
        image = transform(image).unsqueeze(0)
        # Now we can pass the image to the model and get the predicted values:
        with torch.no_grad():
            outputs = model(image)
        return outputs


    def predict_with_image_obj(self, image):
        image = train.transform(image)
        image = train.torch.unsqueeze(image, 0) # Add a batch dimension
        # Now we can pass the image to the model and get the predicted values:
        with train.torch.no_grad():
            outputs = model(image)
        return outputs

    # This function is used to extract the outputs from tensors to a normal list.
    def parse_outputs(self, outputs):
        # We access 0th index because outputs have a batch dimension
        predicted_values = outputs.tolist()[0]

        # Mutliply each element by 512
        predicted_values = [x * 512 for x in predicted_values]

        return predicted_values


    def predict_single_image(self, image_path, return_time=False):
        time_start = time.time()
        predicted = self.predict(image_path)
        parsed = self.parse_outputs(predicted)
        time_spent = None
        if return_time:
            time_spent = time.time() - time_start
        return parsed, time_spent

    # Lets create a few functions:
    def scan_all_images(self, print_output=False):
        start_time = time.time()

        for i,image_path in enumerate(image_paths):
            predicted = predict(image_path)
            if print_output:
                print(f"{(i+1)}. {parse_outputs(predicted)}")

        end_time = time.time()
        total_time = end_time - start_time  # Calculate the elapsed time
        loops_per_second = len(image_paths) / total_time
        print(f"\nInference Speed: {loops_per_second} FPS")


    def get_average_accuracy(self, print_output=False, max_rot_accuracy=30):
        total_rotation_accuracy = 0
        total_distance_accuracy = 0
        counter = 0
        for i,image_path in enumerate(tqdm(image_paths,desc="Processing images.")):
            preds = parse_outputs(predict(image_path))
            predicted_rotation = preds[0]
            predicted_distance = preds[1]

            # We get the accuracy for the rotation value:
            real_rotation = data_targets[i][0]
            rotation_accuracy = abs(real_rotation - predicted_rotation)

            # We get the accuracy for the distance value:
            real_distance = data_targets[i][1]
            distance_accuracy = abs(real_distance - predicted_distance)

            if rotation_accuracy < max_rot_accuracy:
                total_rotation_accuracy += rotation_accuracy
                counter += 1

            total_distance_accuracy += distance_accuracy

        if print_output:
            print(f"Average Model Accuracy: Deg: {total_rotation_accuracy / counter}, Distance: {total_distance_accuracy / len(image_paths)}")
            print("\nImages Processed (rotation wise): ", counter)

        return total_rotation_accuracy / counter, total_distance_accuracy / len(image_paths)

    # Let's create a function below that uses matplotlib to display the image and
    # the predicted values, along with the real values.
    def display_image(self, image_path, real_values, predicted_values, plot_pred=True):
        image = Image.open(image_path)
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        plt.title(f"Image: {os.path.basename(image_path)}\nReal Values: {real_values}\nPredicted Values: {predicted_values}")
        if plot_pred:
            plt.scatter(predicted_values[0], predicted_values[1], color="red")
        plt.show()


    def show_images_with_plt(self, target_image_data, shuffle=False):

        (image_paths_shuffled,
        data_targets_shuffled) = self.shuffle_images(target_image_data)
        # We can print the image path for reference if needed
        print(data_targets_shuffled[0])

        for i in range(len(image_paths_shuffled)):
            predicted = self.predict(image_paths_shuffled[i])
            self.display_image(
                image_paths_shuffled[i],
                # Access data targets and convert to list from tensor
                data_targets_shuffled[i].tolist(),
                self.parse_outputs(predicted)
            )


    def shuffle_images(self, target_image_data):
        indices = list(range(len(target_image_data)))
        random.shuffle(indices)
        # Same indices for both lists so they retain relation
        images_shuffled = [target_image_data[i][0] for i in indices]
        data_targets_shuffled = [target_image_data[i][1] for i in indices]

        return images_shuffled, data_targets_shuffled


    def get_num_images(self):
        n_images = len(
            os.listdir(
                os.path.join(
                    current_script_path,
                    "../../godot/data_images"
                )
            )
        )
        return n_images

# Specifically for the view
class PltInferenceView(InferenceTools):
    def __init__(self, target_image_data, shuffle_images=False, plot_pred=True):
        self.current_index = 0
        self.target_image_data = target_image_data
        self.plot_pred = plot_pred

        if shuffle_images:
            self.target_image_data = self.shuffle_images(target_image_data)


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
    inference_tools = InferenceTools()
    # Load some stuff based on model num
    inference_tools.load_model(model_num)
    target_image_data = inference_tools.load_target_image_data(model_num)

    # Used for testing
    test_image_path = os.path.join(
        current_script_path,
        "../../godot/data_images/image_999.png"
    )

    # Instance the PltInferenceView class
    inference_image_viewer = PltInferenceView(target_image_data)
    # Setup and launch
    inference_image_viewer.setup_plt()
