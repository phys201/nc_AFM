#this file contains tests of the force caluclation function
from unittest import TestCase

import numpy as np

import ncafm.functions as fcn

dummy_a = 1*10**-9
dummy_k = 2000
dummy_f0 = 20*10**3
dummy_z = np.arange(1, 21, 0.5)*10**-9

const1 = 10**-7
const2 = 1.1*10**-9

dummy_df = const1*((const2)**2/dummy_z**3 - (const2)**1/dummy_z**2) 

class Testdf2force(TestCase):
    def test_arraysize(self):
        dummy_force = fcn.df2force(dummy_z, dummy_df, dummy_a, dummy_k, dummy_f0)
        size_z = len(dummy_z)
        size_force = len(dummy_force)
        self.assertEqual(size_z-1, size_force)
        
    def test_force_magnitude(self):
        '''minimum force for these values should be ~ -10 nN ~= 10^-8)'''
        dummy_force = fcn.df2force(dummy_z, dummy_df, dummy_a, dummy_k, dummy_f0)
        min_force = np.abs(np.min(dummy_force))
        self.assertAlmostEqual(-8, np.log10(min_force),1)