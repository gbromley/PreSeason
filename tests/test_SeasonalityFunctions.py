import unittest
import os
import numpy as np
import seasonality.seasonalityfunctions as sf
import time


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.process_time()
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.process_time()
        print(f"{func.__name__} ran in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

class BaseTesting(unittest.TestCase):
    
    @classmethod
    def loadData(cls):
        cls.TestDir = os.getcwd()+'/tests/'
        cls.ClimatologicalDailyP = np.load(cls.TestDir+'precip_dayofyear_testdata.npz')['annual_precip_cycle']
        
        cls.FourierCoefficients = np.load(cls.TestDir+'test_fourier_coefficients.npz')
        cls.SmoothFifthHarmonic = np.load(cls.TestDir+'5th_order_smooth_of_apc.npz')['arr_0']
        
    @classmethod
    def setUpClass(cls) -> None:
        cls.loadData()
        return super().setUpClass()

### Testing Section
class TestFourier(BaseTesting):
    def setUp(self):
        self.setUpClass()
        
    def test_fourier_coefficients(self):
        a_n, b_n, var_n = sf.fourier_coefficients(self.ClimatologicalDailyP, num_harm = 2)
        test_a_n = np.array_equal(a_n,self.FourierCoefficients['a_n'])
        test_b_n = np.array_equal(b_n,self.FourierCoefficients['b_n'])
        test_var_n = np.array_equal(var_n,self.FourierCoefficients['var_n'])
        
        self.assertTrue(test_a_n)
        self.assertTrue(test_b_n)
        self.assertTrue(test_var_n)
        
    

class TestSmoothing(BaseTesting):
        def setUp(self):
            self.input_data_B17 = np.array([1,2,3,4,5])
            #Below is expected smoothing output
            self.check_data_B17 = np.array([1,2,3,3,4])
            
            
        def test_smoothing(self):
            smooth_output = sf.smoothing_harmonic(self.ClimatologicalDailyP,num_harm=5)
            success = np.array_equal(self.SmoothFifthHarmonic,smooth_output)
            self.assertTrue(success)
        
        def test_B17_smoothing(self):
            test_output = sf.smooth_B17(self.input_data_B17, num_passes=1)
            np.testing.assert_array_equal(self.check_data_B17, test_output)
            
        def test_convolve_testing(self):

            test_output = sf.filter3(self.input_data_B17, num_passes=1)
            np.testing.assert_array_equal(self.check_data_B17, test_output)