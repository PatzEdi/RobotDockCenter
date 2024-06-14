import unittest
import polars as pl
import os
import sys
import math

# Let's use sys.path to import the inference script from the machine_learning directory
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'machine_learning'))
sys.path.append(dir_path)
import inference

class TestDataProcess(unittest.TestCase):

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
        self.assertEqual(df.shape, (1200, 5))

    def test_equal_data_distribution(self):
        df = self.open_data()
        # First, let's get all the column values in lists:
        directions = df['Direction'].to_list()
        distances = df['Distance_rline'].to_list()
        rotations = df['Rotation_value'].to_list()

        # Lets make sure there are an equal amount of ones and zeros in the 'Direction' column
        self.assertEqual(directions.count(0), directions.count(1))

        # Now for the distances, we need to figure out how many of each type of distance there are, and if they are distributed well.
        distances_int = [math.floor(distance) for distance in distances]
        distances_dict = {i: distances_int.count(i) for i in range(8)}

        # Let's assert that the distances are distributed well by checking if the values in the dict above are all equal:
        for i in range(len(distances_dict)):
            self.assertEqual(distances_dict[i], distances_dict[0])

        self.assertEqual(sum(distances_dict[i] for i in range(len(distances_dict))), df.shape[0]) # Let's make sure the counting was correct

        # Now, we do the same for the rotations:
        rotations_int = [math.floor(rotation) if rotation > 0 else math.ceil(rotation) for rotation in rotations]
        rotations_dict = {i: rotations_int.count(i) for i in range(-30,31,1)}
        # Let's assert that the rotations are distributed well by checking if the amount values in the dict above are all equal:
        for i in range(-30,31):
            self.assertEqual(rotations_dict[i], rotations_dict[0])
        
        self.assertEqual(sum(rotations_dict[i] for i in range(-30,31)), df.shape[0]) # Let's make sure the counting was correct

        print("\nData is distrubuted well! :)\n")
class TestModel(unittest.TestCase):
    def open_data(self):
        # Define a path to a test CSV file
        test_csv_path = os.path.join(os.path.dirname(__file__), '../src/godot/data_text/image_info_data.csv')

        # Read the CSV file using the function you're testing
        return pl.read_csv(test_csv_path, infer_schema_length=10000)
    def test_model_performance(self):
        image_paths, targets = inference.shuffle_images() # These are always random, so we can just pick the first image and its targets. It will be different every time.
        image_outputs = inference.parse_outputs(inference.predict(image_paths[0]))
        image_targets = inference.parse_outputs(targets[0])
        print("\nModel is being tested on: " + os.path.basename(image_paths[0]) + "\n")
        # Direction:
        self.assertEqual(image_outputs[0], image_targets[0])
        # Distance_rline:
        self.assertAlmostEqual(image_outputs[1], image_targets[1], delta=1)
        # Rotation_value:
        self.assertAlmostEqual(image_outputs[2], image_targets[2], delta=1.5)

        print("Model is performing well!")
if __name__ == '__main__':
    unittest.main()