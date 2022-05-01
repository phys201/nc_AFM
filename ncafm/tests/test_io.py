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

class TestDataImport(TestCase):
    def test_package(self):
        s = nc_AFM.test()
        self.assertTrue(isinstance(s, str))
        
    def test_get_data(self):
        self.testdata = io.load_data(full_data_directory)
        
    def test_first_column_is_z(self):
        test_data = io.load_data(full_data_directory)
        
        first_col = test_data.columns[0]
        self.assertEqual(first_col, 'z')
        
class TestGetFilePath(TestCase):
    def test_get_example_data_file_path(self):
        test_path = io.get_example_data_file_path(filename, data_dir = 'ncafm/example_data')
        self.assertTrue(isinstance(test_path,Path)) 

if __name__ == '__main__':
    unittest.main()