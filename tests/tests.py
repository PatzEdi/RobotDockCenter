import unittest
import polars as pl
import os
import sys

# Add the path where data_process.py is located to the system path
current_script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_script_path, '../src/machine_learning/'))

# Now you can import your module
#import data_process

class TestDataProcess(unittest.TestCase):
    def test_data_size(self):
        # Define a path to a test CSV file
        test_csv_path = os.path.join(os.path.dirname(__file__), '../src/godot/data_text/image_info_data.txt')


        # Read the CSV file using the function you're testing
        df = pl.read_csv(test_csv_path)

        # Assert that the DataFrame has the correct shape
        self.assertEqual(df.shape, (300, 5))

if __name__ == '__main__':
    unittest.main()