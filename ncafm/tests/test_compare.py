#this file contains tests of the comparison function in the compare module.
from unittest import TestCase

import numpy as np
import pandas as pd
import ncafm.model as ml
import ncafm.compare as compare
import pymc3 as pm

dummy_m = 1
dummy_b = 3
dummy_x = np.arange(1,5,0.1)
dummy_y = dummy_m*dummy_x + dummy_b

m1 = pm.Model()
with m1:
    slope = pm.Normal('slope m1', mu = 0, sigma = 1)
    intercept = pm.Normal('intercept m1', mu = 0, sigma = 5)    
    line = pm.Normal('line', mu = slope*dummy_x + dummy_b, sigma = 0.1, observed = dummy_y)
        
m2 = pm.Model()
with m2:
    slope = pm.Normal('slope m2', mu = 0, sigma = 1)
    intercept = pm.Normal('intercept m2', mu = 0, sigma = 5)
    quad = pm.Normal('quad', mu = slope*dummy_x**2 + dummy_b, sigma = 0.1, observed = dummy_y)
        
def TestCompareModels(TestCase):
    def test_output_float(self):
        o12 = compare.compare_models(m1,m2,num_samples = 800, num_tunes = 300)
        self.assertTrue(isinstance(o12, float))
        
    def test_m1_grt_m2(self):
        o12 = compare.compare_models(m1,m2,num_samples = 800, num_tunes = 300)
        self.asserGreater(o12, 1)
        
