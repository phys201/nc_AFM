#this file contains tests of the models defined in the model module.
from unittest import TestCase

import numpy as np
import ncafm.model as model
import ncafm.functions as fcn
import scipy.stats


dummy_a = 1*10**-9
dummy_k = 2000
dummy_f0 = 20*10**3
dummy_z = np.arange(1, 21, 0.5)*10**-9

const1 = 10**-7
const2 = 1.1*10**-9

dummy_df = const1*((const2)**2/dummy_z**3 - (const2)**1/dummy_z**2) 

noise = 0.1

dummy_force = fcn.df2force(dummy_df, dummy_z, dummy_a, dummy_k, dummy_f0) 

class TestLenJonModel(TestCase):
    def test_returns_model_class(self):
        dummy_noisy_force = dummy_force + scipy.stats.norm.rvs(loc=0, scale = noise, size = len(z)-1)
        lj_model = model.len_jon(dummy_noisy_force, noise)
        
        self.assertTrue(isinstance(lj_model, pm.Model()))