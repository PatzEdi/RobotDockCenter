import time
import train
from train import Predictor
from train import os
from train import tqdm
# These two below are just used for other purposes in this script. The real inferencing occurs in the predict() method found in here.
from train import image_paths
from train import data_targets
from train import model_num

current_script_path = os.path.dirname(os.path.abspath(__file__))
model_save_path = os.path.join(current_script_path, '../../../models/model' + str(model_num) + '.pth')

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
# This function is used to extract the outputs from tensors to a normal list.
def parse_outputs(outputs):
    predicted_values = [value.item() for value in outputs]

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
def get_average_accuracy(print_output=False, max_accuracy=30):
    total_accuracy = 0
    counter = 0
    for i,image_path in enumerate(tqdm(image_paths,desc="Processing images.")):
        predicted_rotation = parse_outputs(predict(image_path))[0]
        real_rotation = data_targets[i][0]
        accuracy = abs(real_rotation - predicted_rotation)

        if accuracy < max_accuracy:
            total_accuracy += accuracy
            counter += 1
        
    if print_output:
        print(f"Average Model Accuracy: {total_accuracy / len(image_paths)}")
        print("\nImages Processed: ", counter)
    return total_accuracy / counter
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
#scan_all_images(print_output=False) # This will print out the predicted values for all of the images in the data_images folder, realtime
#show_images_with_plt()
#print(predict("/Users/edwardferrari/Documents/GitHub/RobotDockCenter/src/godot/frames_testing/frame.png"))
#get_average_accuracy(print_output=True)