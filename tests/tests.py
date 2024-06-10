import unittest
import polars as pl
import os
import sys

class TestDataProcess(unittest.TestCase):
    def test_data_size(self):
        # Define a path to a test CSV file
        test_csv_path = os.path.join(os.path.dirname(__file__), '../src/godot/data_text/image_info_data.txt')


        # Read the CSV file using the function you're testing
        df = pl.read_csv(test_csv_path, infer_schema_length=10000)

        # Assert that the DataFrame has the correct shape
        self.assertEqual(df.shape, (1800, 5))

if __name__ == '__main__':
    unittest.main()