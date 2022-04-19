# this file contains tests of io.py
from unittest import TestCase

import numpy as np

import ncafm.io as io

from pathlib import Path

class TestDataPathing(TestCase):
    def test_inputs_for_get_path(self):
        additional_path = 'example_data'
        filename = 'all_data_dataframe_237K.csv'
        
        #use function to define path
        test_path = io.get_example_data_file_path(filename, data_dir = additional_path)
        
        #use Path to define the path
        current_directory = Path.cwd()
        directory = Path(current_directory, additional_path, filename)
        
        self.assertEqual(directory, test_path)

        