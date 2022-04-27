#this file contains tests of the force caluclation function
from unittest import TestCase

import numpy as np

import ncafm.functions as fn
 
dummy_a = 1*10**-9
dummy_k = 2000
dummy_f0 = 20*10**3
dummy_z = np.arange(1, 21, 0.5)*10**-9

const1 = 10**-7
const2 = 1.1*10**-9

dummy_df = const1*((const2)**2/dummy_z**3 - (const2)**1/dummy_z**2)

dummy_epsilon = 1
dummy_sigma = 1
dummy_xi = 1
dummy_hamaker = 1
dummy_radius = 1
dummy_theta = 45
    
class Testdf2force(TestCase):
    def test_arraysize(self):
        dummy_force = fn.df2force(dummy_z, dummy_df, dummy_a, dummy_k, dummy_f0)
        size_z = len(dummy_z)
        size_force = len(dummy_force)
        self.assertEqual(size_z-1, size_force)
        
    def test_force_magnitude(self):
        dummy_force = fn.df2force(dummy_z, dummy_df, dummy_a, dummy_k, dummy_f0)
        min_force = np.abs(np.min(dummy_force))
        #use log10 to check order of magnitude. Should be ~10 nN so e-8.
        self.assertAlmostEqual(-8, np.log10(min_force),1)
        
class Testptp2variation(TestCase):
    def test_noise_magnitude(self):
        dummy_ptp = 1
        sampling_time = 0.0125
        test_noise = fn.ptp2variation(dummy_ptp, sampling_time)
        self.assertAlmostEqual(0.04, test_noise, 1)
    
class TestSimulate_lj_data(TestCase):
    def test_force_magnitude(self):
        z_test = dummy_z[0]*10**9
        noise = 0
        
        dummy_lj = 4*dummy_epsilon*(12*dummy_sigma**12/z_test**13 - 6*dummy_sigma**6/z_test**7)
        lj_from_fn = fn.simulate_lj_data(dummy_epsilon, dummy_sigma, noise, z_test)
        self.assertEqual(dummy_lj, lj_from_fn)
    
    def test_datatype_array(self):
        z_test = dummy_z[0]*10**9
        noise1 = 1
        lj_noise1 = fn.simulate_lj_data(dummy_epsilon, dummy_sigma, noise1, z_test)
        self.assertTrue(isinstance(lj_noise1, np.ndarray))
        
    def test_adding_noise(self):
        z_test = dummy_z[0]*10**9
        noise0 = 0
        noise1 = 0.1
        
        lj_noise0 = fn.simulate_lj_data(dummy_epsilon, dummy_sigma, noise0, z_test)
        lj_noise1 = fn.simulate_lj_data(dummy_epsilon, dummy_sigma, noise1, z_test)
        diff = np.abs(lj_noise1-lj_noise0)
        #this still might fail 5% of the time ..
        self.assertLess(diff[0], noise1*2)
    
class TestSimulate_data_sph(TestCase):
    def test_force_magnitude(self):
        z_test = dummy_z[0]*10**9
        noise = 0
        
        dummy_sph = dummy_xi/z_test**3 - 2*dummy_hamaker*dummy_radius**3/(3*z_test**2*(z_test+2*dummy_radius)**2)
        sph_from_fn = fn.simulate_data_sph(dummy_xi, dummy_hamaker, dummy_radius, noise, z_test)
        self.assertEqual(dummy_sph, sph_from_fn) 
    
class TestSimulate_data_cone(TestCase):
    def test_force_magnitude(self):
        z_test = dummy_z[0]*10**9
        noise = 0
        
        dummy_cone = dummy_xi/z_test**3 - dummy_hamaker*np.tan(np.deg2rad(dummy_theta))**2/(6*z_test)
        cone_from_fn = fn.simulate_data_cone(dummy_xi, dummy_hamaker, dummy_theta, noise, z_test)
        self.assertEqual(dummy_cone, cone_from_fn) 
    
class TestSimulate_data_cone_sph(TestCase):
    def test_force_magnitude(self):
        z_test = dummy_z[0]*10**9
        noise = 0
        
        term1 = dummy_radius/z_test**2
        term2 = dummy_radius*(1-np.sin(np.deg2rad(dummy_theta)))/(z_test*(z_test+dummy_radius*(1-np.sin(np.deg2rad(dummy_theta))))) 
        term3 = np.tan(np.deg2rad(dummy_theta))**2/(z_test+dummy_radius*(1-np.sin(np.deg2rad(dummy_theta))))
        
        dummy_cone_sph = dummy_xi/z_test**3 - dummy_hamaker/6*(term1 + term2 + term3)
        cone_sph_from_fn = fn.simulate_data_cone_sph(dummy_xi, dummy_hamaker, dummy_radius, dummy_theta, noise, z_test)
        self.assertEqual(dummy_cone_sph, cone_sph_from_fn) 