import os
import config_parser


model_num = config_parser.get_model_num()

data_images_path = "/Users/edwardferrari/Documents/GitHub/RobotDockCenter/src/godot/data_images"

data_images_path += "2" if model_num == 2 else ""

def clear_data_images():

    for file in os.listdir(data_images_path):
        if not ".gitkeep" in file:
            print(file)
            os.remove(os.path.join(data_images_path, file))

confirmation = input(f"Are you sure you want to delete all of the images in the data_images folder for model number {model_num}? (y/n): ")

if confirmation.lower() == "y":
    clear_data_images()
    print(f"All images have been deleted for model number {model_num}")