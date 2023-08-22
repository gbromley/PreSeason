import unittest
import os
import numpy as np
import seasonality.seasonalityfunctions as sf

### Testing Section
class TestFourier(unittest.TestCase):
    def setUp(self):
        TestDir = os.getcwd()+'/tests/'
        with np.load(TestDir+'precip_dayofyear_testdata.npz') as data:
            self.annual_P_cycle = data['annual_precip_cycle']
        
        self.test_coefficients = np.load(TestDir+'test_fourier_coefficients.npz')
        self.test_smooth = np.load(TestDir+'5th_order_smooth_of_apc.npz')
        
    def test_fourier_coefficients(self):
        a_n, b_n, var_n = sf.fourier_coefficients(self.annual_P_cycle, num_harm = 2)
        test_a_n = np.array_equal(a_n,self.test_coefficients['a_n'])
        test_b_n = np.array_equal(b_n,self.test_coefficients['b_n'])
        test_var_n = np.array_equal(var_n,self.test_coefficients['var_n'])
        
        self.assertTrue(test_a_n)
        self.assertTrue(test_b_n)
        self.assertTrue(test_var_n)
        
    def TestSmoothing(self):
        smooth_output = sf.harmonic_smoothing(self.annual_P_cycle,num_harm=5)
        success = np.array_equal(self.test_smooth,smooth_output)
        self.assertTrue(success)
    