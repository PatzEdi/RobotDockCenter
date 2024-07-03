import unittest
import polars as pl
import os
import sys
import math

# Let's use sys.path to import the inference script from the machine_learning directory
# inference_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'machine_learning'))
# sys.path.append(inference_dir_path)
# import inference

config_parser_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(config_parser_dir_path)
import config_parser

class TestDataProcess(unittest.TestCase):
    rotation_range = config_parser.get_rotation_range()
    def open_data(self):
        # Define a path to a test CSV file
        test_csv_path = os.path.join(os.path.dirname(__file__), '../src/godot/data_text/image_info_data.csv')

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

        rotations = df['RotationValue'].to_list()
        # Now, we do the same for the rotations:
        rotations_int = [round(rotation) for rotation in rotations]
        rotation_amounts = [rotations_int.count(rotation) for rotation in list(set(rotations_int))]
        self.assertEqual(len(set(rotation_amounts)), 1)
        print("\nData is distrubuted well! :)\n")
# class TestModel(unittest.TestCase):
#     def open_data(self):
#         # Define a path to a test CSV file
#         test_csv_path = os.path.join(os.path.dirname(__file__), '../src/godot/data_text/image_info_data.csv')

#         # Read the CSV file using the function you're testing
#         return pl.read_csv(test_csv_path, infer_schema_length=10000)
#     def test_model_performance(self):
#         image_paths, targets = inference.shuffle_images() # These are always random, so we can just pick the first image and its targets. It will be different every time.
#         image_outputs = inference.parse_outputs(inference.predict(image_paths[0]))
#         image_targets = inference.parse_outputs(targets[0])
#         print("\nModel is being tested on: " + os.path.basename(image_paths[0]) + "\n")
#         # Direction:
#         self.assertEqual(image_outputs[0], image_targets[0])
#         # Distance_rline:
#         self.assertAlmostEqual(image_outputs[1], image_targets[1], delta=1)
#         # Rotation_value:
#         self.assertAlmostEqual(image_outputs[2], image_targets[2], delta=1.5)

#         print("Model is performing well!")
if __name__ == '__main__':
    unittest.main()