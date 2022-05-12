#this file contains tests of the functions used to in the visualize module
from unittest import TestCase

import ncafm.visualize as vis
import ncafm.model as ml
import numpy as np
import pandas as pd
import pymc3 as pm

dummy_z = np.arange(1, 21, 0.5)

dummy_c = 100
dummy_alpha = 1
dummy_hamaker = 1
dummy_radius = 50
dummy_noise = 0.1

dummy_force = dummy_c*np.exp(-dummy_alpha*dummy_z) - dummy_hamaker*dummy_radius**3/(dummy_z**2*(dummy_z + 2*dummy_radius)**2)

class TestPlotTraces(TestCase):
    def test_bad_string_sph_cone(self):
        sph_model = ml.vdw_sph(dummy_z, dummy_force, dummy_noise, dummy_hamaker)
        map_estimate = pm.find_MAP(model = sph_model)
        with sph_model:
            traces = pm.sample(1000, return_inferencedata=True)
        self.assertRaises(ValueError, vis.plot_traces, dummy_z, dummy_force, traces, map_estimate, dummy_hamaker, fit_type = 'test')

class TestPrintCI(TestCase):
    def test_input_trace(self):
        sph_model = ml.vdw_sph(dummy_z, dummy_force, dummy_noise, dummy_hamaker)
        map_estimate = pm.find_MAP(model = sph_model)
        with sph_model:
            traces = pm.sample(1000, return_inferencedata=True)
        traces_dataframe = traces.posterior.to_dataframe()
        self.assertEqual(vis.print_ci(traces, 'radius'), vis.print_ci(traces_dataframe, 'radius'))

if __name__ == '__main__':
    unittest.main()