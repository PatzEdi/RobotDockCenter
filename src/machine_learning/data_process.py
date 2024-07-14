# This script will be used to gather the necessary data for training the model
# General system libraries:
import os
# Data libraries:
import polars as pl

print("\nDone importing libraries.")

# First, let's define the paths of the data, using os so that we can use relative paths:
current_script_path = os.path.dirname(os.path.abspath(__file__))

def get_data_targets(torch, model_num, convert_targets_to_tensor=False):
    # These are the main paths we will be using:
    data_images_path = os.path.join(current_script_path, '../godot/data_images' if model_num == 1 else '../godot/data_images2')
    data_image_text_path = os.path.join(current_script_path, '../godot/data_text/image_info_data_model' + str(model_num) + '.csv')

    print("\nLoading data from: " + data_image_text_path + " using polars...")
    # Let's read and load the data from a csv file using polars (very fast)
    df = pl.read_csv(data_image_text_path, infer_schema_length=10000) # We technically have all of the data here to train the model. Thanks polars for making it so easy! :)

    # Let's print out the contents:
    print(df)
    # Let's print out the amount of images that are in the data, which is the amount of rows:
    print("Amount of images: " + str(df.shape[0]))

    image_paths = [os.path.join(data_images_path, image_name) for image_name in df['Image'].to_list()]

    # Here we load in all of the data tables from the polars data frame: 
    distances_cline = df['DistanceCline'].to_list()
    rotation_values = df['RotationValue'].to_list()

    data_targets = []

    for i in range(df.shape[0]):
        data_targets.append([rotation_values[i],distances_cline[i]])
    # We convert the data_targets to tensors if the parameter is set to True:
    if (convert_targets_to_tensor):
        for i in range(len(data_targets)):
            for i2 in range(len(data_targets[i])):
                data_targets[i][i2] = torch.tensor(data_targets[i][i2], dtype=torch.float64)

    return data_targets, image_paths