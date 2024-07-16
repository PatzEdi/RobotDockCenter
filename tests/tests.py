import unittest
import polars as pl
import os
import sys
import math
from hashlib import md5
# Let's use sys.path to import the inference script from the machine_learning directory
# inference_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'machine_learning'))
# sys.path.append(inference_dir_path)
# import inference

config_parser_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(config_parser_dir_path)
import config_parser

model = 1

class TestDataProcess(unittest.TestCase):

    rotation_range = config_parser.get_rotation_range()

    def open_data(self):
        # Define a path to a test CSV file
        test_csv_path = os.path.join(os.path.dirname(__file__), '../src/godot/data_text/image_info_data_model' + str(model) + '.csv')

        # Read the CSV file using the function you're testing
        return pl.read_csv(test_csv_path, infer_schema_length=10000)

    def test_data_open(self):
        df = self.open_data()
        # Let's assert that there are no errors when loading the data
        self.assertIsNotNone(df)
        self.df = df
        
    def test_data_size(self):
        df = self.open_data()
        # Assert that the DataFrame has the correct shape
        self.assertEqual(df.shape, (config_parser.get_data_set_len_row(), config_parser.get_data_set_len_col()))

    def test_equal_data_distribution(self):
        df = self.open_data()

        # We check for the rotation values:
        rotations = df['RotationValue'].to_list()

        self.assertEqual(self.is_equally_distributed(rotations), True)

        distances_cline = df['DistanceCline'].to_list()
        self.assertEqual(self.is_equally_distributed(distances_cline), True)

        

        print("\nData is distrubuted well! :)\n")

    def is_equally_distributed(self,data):
        # Step 1: Round values to the nearest 0.5 to standardize them
        rounded_data = [round(value * 2) / 2 for value in data]
        
        # Step 2: Count occurrences of each unique value
        value_counts = [rounded_data.count(value) for value in set(rounded_data)]
        
        # Step 3: Check if all values have the same count
        return len(set(value_counts)) == 1  # True if all counts are the same, False otherwise
    # This test makes sure that all of the images in the data are different, and does so by  computing the hashes of each image.
    def test_all_images_different(self):
        df = self.open_data()
        image_paths_model1 = [os.path.join(os.path.dirname(__file__), '../src/godot/data_images', image_name) for image_name in df['Image'].to_list()]
        image_paths_model2 = [os.path.join(os.path.dirname(__file__), '../src/godot/data_images2', image_name) for image_name in df['Image'].to_list()]
        # We will store the hashes of the images in a set to check for duplicates
        image_hashes_model1 = set()
        for image_path in image_paths_model1:
            image_hash = self.get_image_hash(image_path)
            image_hashes_model1.add(image_hash)
        self.assertEqual(len(image_hashes_model1), len(image_paths_model1))

        image_hashes_model2 = set()
        for image_path in image_paths_model2:
            image_hash = self.get_image_hash(image_path)
            image_hashes_model2.add(image_hash)
        self.assertEqual(len(image_hashes_model2), len(image_paths_model2))
        
    def get_image_hash(self,image_path):
        # Open the image in binary mode
        with open(image_path, 'rb') as image_file:
            # Read the contents of the file
            image_data = image_file.read()
            # Use hashlib to compute the MD5 hash of the image data
            hash_md5 = md5(image_data)
            # Return the hexadecimal representation of the digest
            return hash_md5.hexdigest()

if __name__ == '__main__':
    unittest.main()