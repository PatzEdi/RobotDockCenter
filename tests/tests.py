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