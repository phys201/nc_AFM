# this file contains tests of io.py
from unittest import TestCase

import numpy as np

import ncafm.io as io

from pathlib import Path

#go 'up' a folder to ncafm'
package_path = Path(__file__).parents[1] 

#go into the data folder
data_folder_path = 'example_data'

#data file name
filename = 'all_data_dataframe_237K.csv'

full_data_directory = Path(package_path, data_folder_path, filename)

'''
I will have to edit this with a different path

test_path = io.get_example_data_file_path(filename, data_dir = additional_path)

class TestDataPathing(TestCase):
    def test_inputs_for_get_path(self):
        
        #use Path to define the path
        current_directory = Path.cwd()
        directory = Path(current_directory, additional_path, filename)
        
        self.assertEqual(directory, test_path)
'''        
class TestDataImport(TestCase):
    def test_get_data(self):
        self.testdata = io.load_data(full_data_directory)
        
    def test_first_column_is_z(self):
        test_data = io.load_data(full_data_directory)
        
        first_col = test_data.columns[0]
        self.assertEqual(first_col, 'z')
