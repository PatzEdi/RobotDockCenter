import time
import train
from train import Predictor
from train import os
# These two below are just used for other purposes in this script. The real inferencing occurs in the predict() method found in here.
from train import image_paths
from train import data_targets

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
def predict_with_image_obj(image):
    image = train.transform(image)
    image = train.torch.unsqueeze(image, 0) # Add a batch dimension
    # Now we can pass the image to the model and get the predicted values:
    with train.torch.no_grad():
        outputs = model(image)
    return outputs
# This is the the function that parses the outputs from the function above and either ceils or floors the direction output, as well as removes the tensor form of each value
def parse_outputs(outputs):
    predicted_values = [value.item() for value in outputs]

    # Let's floor down the first value to zero if it is less than .5, and ceil it if it is above .5:
    if (predicted_values[0] < .5):
        predicted_values[0] = 0
    elif (predicted_values[0] >= .5):
        predicted_values[0] = 1

    return predicted_values

def predict_single_image(image_number):
    predicted = predict(image_paths[image_number-1])
    print(predicted)
    print(parse_outputs(predicted))
# Lets create a few functions:
def scan_all_images(print_output=False):
    start_time = time.time()

    for i,image_path in enumerate(image_paths):
        predicted = predict(image_path)
        if print_output:
            print(f"{(i+1)}. {parse_outputs(predicted)}")

    end_time = time.time()
    total_time = end_time - start_time  # Calculate the elapsed time
    loops_per_second = len(image_paths) / total_time
    print(f"\nInference Speed: {loops_per_second} FPS")

# Let's create a function below that uses matplotlib to display the image and the predicted values, along with the real values.
def display_image(image_path, real_values, predicted_values):
    image = train.Image.open(image_path)
    train.plt.figure(figsize=(10,10))
    train.plt.imshow(image)
    train.plt.title(f"Image: {os.path.basename(image_path)}\nReal Values: {real_values}\nPredicted Values: {predicted_values}")
    train.plt.show()
def show_images_with_plt():
    image_paths_shuffled, data_targets_shuffled = shuffle_images()
    for i in range(len(image_paths_shuffled)):
        predicted = predict(image_paths_shuffled[i])
        display_image(image_paths_shuffled[i],parse_outputs(data_targets_shuffled[i]), parse_outputs(predicted))
def shuffle_images():
    import random
    indices = list(range(len(image_paths)))
    random.shuffle(indices)
    return [image_paths[i] for i in indices], [data_targets[i] for i in indices]
#predict_single_image(1)
#scan_all_images(print_output=True) # This will print out the predicted values for all of the images in the data_images folder, realtime
#show_images_with_plt()