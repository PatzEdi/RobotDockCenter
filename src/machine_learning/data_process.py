# This script will be used to gather the necessary data for training the model

# General system libraries:
import os
# Data libraries:
import polars as pl
import csv

# First, let's define the paths of the data, using os so that we can use relative paths:
current_script_path = os.path.dirname(os.path.abspath(__file__))
# These are the main paths we will be using:
data_images_path = os.path.join(current_script_path, '../godot/data_images')
data_image_text_path = os.path.join(current_script_path, '../godot/data_text/image_info_data.txt')

# Let's read the data from a csv file using polars (very fast)
df = pl.read_csv(data_image_text_path)
# Let's print out the contents:
print(df)