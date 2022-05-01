#this file contains tests of the force caluclation function
from unittest import TestCase

import numpy as np
import pandas as pd

import ncafm.conversions as cnv
 
dummy_a = 1*10**-9
dummy_k = 2000
dummy_f0 = 20*10**3
dummy_z = np.arange(1, 21, 0.5)*10**-9

const1 = 10**-7
const2 = 1.1*10**-9

dummy_df = pd.DataFrame(const1*((const2)**2/dummy_z**3 - (const2)**1/dummy_z**2), columns = ['0'])
    
class Testdf2force(TestCase):
    def test_arraysize(self):
        dummy_force = cnv.df2force(dummy_z, dummy_df, '0', dummy_a, dummy_k, dummy_f0, cpd_v = '0')
        size_z = len(dummy_z)
        size_force = len(dummy_force)
        self.assertEqual(size_z, size_force)
        
    def test_force_magnitude(self):
        dummy_force = cnv.df2force(dummy_z, dummy_df, '0', dummy_a, dummy_k, dummy_f0, cpd_v = '0')
        min_force = np.abs(np.min(dummy_force))
        #use log10 to check order of magnitude. Should be ~10 nN so e-8.
        self.assertAlmostEqual(-8.4, np.log10(min_force),1)
        
class Testptp2variation(TestCase):
    def test_noise_magnitude(self):
        dummy_ptp = 1
        sampling_time = 0.0125
        test_noise = cnv.ptp2variation(dummy_ptp, sampling_time)
        self.assertAlmostEqual(0.04, test_noise, 1)

if __name__ == '__main__':
    unittest.main()