import os
import config_parser


current_script_path = os.path.dirname(os.path.abspath(__file__))
model_num = config_parser.get_model_num()

# Let's just override the model_num for now
model_num = 1

abs_data_images_path = os.path.join(
    current_script_path,
    "src/godot/data_images"
)

abs_data_images_path += "2" if model_num == 2 else ""
# Print out the path so that we are sure what we are deleting
print(f"\nData images path:\n{abs_data_images_path}")

def clear_data_images(data_images_path):
    for file in os.listdir(data_images_path):
        if ".gitkeep" not in file:
            print(file)
            os.remove(os.path.join(data_images_path, file))

confirmation = input(
    f"Are you sure you want to delete all of the images in the"
    f"data_images folder for model number {model_num}? (y/n): "
)

if confirmation.lower() == "y":
    clear_data_images(abs_data_images_path)
    print(f"All images have been deleted for model number {model_num}")
