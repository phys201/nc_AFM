#this file contains tests of the models defined in the model module.
from unittest import TestCase

import numpy as np
import ncafm.model as ml
import ncafm.functions as fn
import scipy.stats
import pymc3 as pm


dummy_a = 1*10**-9
dummy_k = 2000
dummy_f0 = 20*10**3
dummy_z = np.arange(1, 21, 0.5)*10**-9

const1 = 10**-7
const2 = 1.1*10**-9

dummy_df = const1*((const2)**2/dummy_z**3 - (const2)**1/dummy_z**2) 

noise = 0.1

dummy_force = fn.df2force(dummy_z, dummy_df, dummy_a, dummy_k, dummy_f0) 

dummy_hamaker = 1

class TestLenJonModel(TestCase):
    def test_returns_model_class(self):
        dummy_noisy_force = dummy_force*10**9 + scipy.stats.norm.rvs(loc=0, scale = noise, size = len(dummy_z)-1)
        lj_model = ml.len_jon(dummy_z[:-1], dummy_noisy_force, noise)  
        self.assertTrue(isinstance(lj_model, pm.Model))
        
        
class TestvdWSphModel(TestCase):
    def test_returns_model_class(self):
        dummy_noisy_force = dummy_force*10**9 + scipy.stats.norm.rvs(loc=0, scale = noise, size = len(dummy_z)-1)
        vdw_sph_model = ml.vdw_sph(dummy_z[:-1], dummy_noisy_force, noise, dummy_hamaker)
        self.assertTrue(isinstance(vdw_sph_model, pm.Model))
        
class TestvdWConeModel(TestCase):
    def test_returns_model_class(self):
        dummy_noisy_force = dummy_force*10**9 + scipy.stats.norm.rvs(loc=0, scale = noise, size = len(dummy_z)-1)
        vdw_cone_model = ml.vdw_cone(dummy_z[:-1], dummy_noisy_force, noise, dummy_hamaker)
        self.assertTrue(isinstance(vdw_cone_model, pm.Model))
        
class TestvdWConeSphModel(TestCase):
    def test_returns_model_class(self):
        dummy_noisy_force = dummy_force*10**9 + scipy.stats.norm.rvs(loc=0, scale = noise, size = len(dummy_z)-1)
        vdw_cone_sph_model = ml.vdw_cone_sph(dummy_z[:-1], dummy_noisy_force, noise, dummy_hamaker)
        self.assertTrue(isinstance(vdw_cone_sph_model, pm.Model))