#this file contains tests of the functions used to simulate force data
from unittest import TestCase

import numpy as np

import ncafm.simulatedata as sim

dummy_z = 1
dummy_epsilon = 1
dummy_sigma = 1
dummy_xi = 1
dummy_hamaker = 1
dummy_radius = 1
dummy_voltage = 1
dummy_theta = 45

class TestSimulate_lj_data(TestCase):
    def test_force_magnitude(self):
        noise = 0
        
        dummy_lj = 4*dummy_epsilon*(12*dummy_sigma**12/dummy_z**13 - 6*dummy_sigma**6/dummy_z**7)
        lj_from_fn = sim.len_jon(dummy_epsilon, dummy_sigma, noise, dummy_z)
        self.assertEqual(dummy_lj, lj_from_fn)
    
    def test_datatype_array(self):
        noise1 = 1
        lj_noise1 = sim.len_jon(dummy_epsilon, dummy_sigma, noise1, dummy_z)
        self.assertTrue(isinstance(lj_noise1, np.ndarray))
        
    def test_adding_noise(self):
        noise0 = 0
        noise1 = 0.1
        
        lj_noise0 = sim.len_jon(dummy_epsilon, dummy_sigma, noise0, dummy_z)
        lj_noise1 = sim.len_jon(dummy_epsilon, dummy_sigma, noise1, dummy_z)
        diff = np.abs(lj_noise1-lj_noise0)
        #this still might fail 5% of the time ..
        self.assertLess(diff[0], noise1*2)
    
class TestSimulate_data_sph(TestCase):
    def test_force_magnitude(self):
        noise = 0
        
        dummy_sph = dummy_xi/dummy_z**3 - 2*dummy_hamaker*dummy_radius**3/(3*dummy_z**2*(dummy_z+2*dummy_radius)**2)
        sph_from_fn = sim.sph(dummy_xi, dummy_hamaker, dummy_radius, noise, dummy_z)
        self.assertEqual(dummy_sph, sph_from_fn) 
    
class TestSimulate_data_cone(TestCase):
    def test_force_magnitude(self):
        noise = 0
        
        dummy_cone = dummy_xi/dummy_z**3 - dummy_hamaker*np.tan(np.deg2rad(dummy_theta))**2/(6*dummy_z)
        cone_from_fn = sim.cone(dummy_xi, dummy_hamaker, dummy_theta, noise, dummy_z)
        self.assertEqual(dummy_cone, cone_from_fn) 
    
class TestSimulate_data_cone_sph(TestCase):
    def test_force_magnitude(self):
        noise = 0
        
        term1 = dummy_radius/dummy_z**2
        term2 = dummy_radius*(1-np.sin(np.deg2rad(dummy_theta)))/(dummy_z*(dummy_z+dummy_radius*(1-np.sin(np.deg2rad(dummy_theta))))) 
        term3 = np.tan(np.deg2rad(dummy_theta))**2/(dummy_z+dummy_radius*(1-np.sin(np.deg2rad(dummy_theta))))
        
        dummy_cone_sph = dummy_xi/dummy_z**3 - dummy_hamaker/6*(term1 + term2 + term3)
        cone_sph_from_fn = sim.cone_sph(dummy_xi, dummy_hamaker, dummy_radius, dummy_theta, noise, dummy_z)
        self.assertEqual(dummy_cone_sph, cone_sph_from_fn)
        
class TestSimulate_data_ele(TestCase):
    def test_force_magnitude_sph(self):
        noise = 0
        vdw_force = - 2*dummy_hamaker*dummy_radius**3/(3*dummy_z**2*(dummy_z+2*dummy_radius)**2)
        
        dummy_ele = dummy_xi/dummy_z**3 + vdw_force - 9*10**36*dummy_voltage**2*(np.pi*dummy_radius**2)**2/dummy_z**4
        ele_from_fn = sim.ele(dummy_theta, dummy_radius, dummy_voltage, noise, dummy_z, dummy_hamaker, dummy_xi, vdw_type = 'sph')
        self.assertEqual(dummy_ele, ele_from_fn)
    
    
    def test_force_magnitude_cone(self):
        noise = 0
        vdw_force = - dummy_hamaker*np.tan(np.deg2rad(dummy_theta))**2/(6*dummy_z) 

        dummy_ele = dummy_xi/dummy_z**3 + vdw_force - 9*10**36*dummy_voltage**2*(np.pi*dummy_radius**2)**2/dummy_z**4
        ele_from_fn = sim.ele(dummy_theta, dummy_radius, dummy_voltage, noise, dummy_z, dummy_hamaker, dummy_xi, vdw_type = 'cone')
        self.assertEqual(dummy_ele, ele_from_fn)


    def test_force_magnitude_sph_cone(self):
        noise = 0
        term1 = dummy_radius/dummy_z**2
        term2 = dummy_radius*(1-np.sin(np.deg2rad(dummy_theta)))/(dummy_z*(dummy_z+dummy_radius*(1-np.sin(np.deg2rad(dummy_theta))))) 
        term3 = np.tan(np.deg2rad(dummy_theta))**2/(dummy_z+dummy_radius*(1-np.sin(np.deg2rad(dummy_theta))))
        vdw_force = - dummy_hamaker/6*(term1 + term2 + term3)        

        dummy_ele = dummy_xi/dummy_z**3 + vdw_force - 9*10**36*dummy_voltage**2*(np.pi*dummy_radius**2)**2/dummy_z**4
        ele_from_fn = sim.ele(dummy_theta, dummy_radius, dummy_voltage, noise, dummy_z, dummy_hamaker, dummy_xi, vdw_type = 'sph+cone')
        self.assertEqual(dummy_ele, ele_from_fn)