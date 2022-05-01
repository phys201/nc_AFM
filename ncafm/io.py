import pandas as pd
from pathlib import Path


def get_example_data_file_path(filename, data_dir='example_data'):
    '''
    function that makes a path name using the current location of the file. 
    It add the option to add additional folders to the current directory.
    The user must specify the file name.
    
    Input:
    ------
    filename: string. file name (with extension) of the file you wish to path to
    data_dir: string. additional path specifications. Must be folders inside the current directory.
    
    Returns:
    --------
    path to the specified file
    '''
    # Path.cwd() returns the location of the source file currently in use. 
    # We can use it as base path to construct other paths from that should end up correct on other machines or
    # when the package is installed
    
    current_directory = Path.cwd()
    data_path = Path(current_directory, data_dir, filename)
    
    #start = Path.abspath(__file__)
    #start_dir = Path.dirname(start)
    #data_dir = Path.join(start_dir, data_dir)
    #data_path = Path.join(start_dir, data_dir, filename)
    
    return data_path


def load_data(data_file):
    '''
    Creates a pandas array from a data file specifed by a pathname. 
    '''
    return pd.read_csv(data_file)