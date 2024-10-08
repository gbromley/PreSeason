import unittest
import os
import numpy as np
import preseason.tools as sf
import time
import scipy.stats as stats
import pandas as pd
import pytest
#TODO Clean up the old unittest code.

### Generate synthetic data ###
@pytest.fixture
def create_synthetic_xr_dataarray():
    #TODO make sure this doesn't violate a test rule 
    data = sf.generate_p_da()
    return data

@pytest.fixture
def create_synthetic_pd_df():
    data = sf.generate_p_da()
    return data


# Testing functions that share similar inputs
functions_to_test = [sf.fourier_coefficients, sf.smoothing_harmonic, sf.min_first_harmonic, sf.smooth_B17, sf.filter3, sf.find_ddt_onset, sf.cumul_anom]

#Testing common input types to make sure functions don't fail silently
input_types = [5, 'five', 5.7, list([0,1,2]), create_synthetic_xr_dataarray, create_synthetic_pd_df]

@pytest.mark.parametrize('types', input_types) 
@pytest.mark.parametrize('func', functions_to_test)
def test_tseries_input(func, types):
    with pytest.raises(TypeError):
        func(types) 


class BaseTesting(unittest.TestCase):
    
    @classmethod
    def loadData(cls):
        cls.TestDir = os.getcwd()+'/tests/test_data/'
        cls.ClimatologicalDailyP = np.load(cls.TestDir+'precip_dayofyear_testdata.npz')['annual_precip_cycle']
        
        cls.FourierCoefficients = np.load(cls.TestDir+'test_fourier_coefficients.npz')
        cls.SmoothFifthHarmonic = np.load(cls.TestDir+'5th_order_smooth_of_apc.npz')['arr_0']
        
    @classmethod
    def setUpClass(cls) -> None:
        cls.loadData()
        return super().setUpClass()

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
            
        def test_inflection(self):
            x = np.linspace(-10, 10, 400)
            arr = x**3
            out = sf.find_ddt_onset(arr)
            answer = 195            
            
            np.testing.assert_equal(out, answer)

class TestStats(BaseTesting):
    def setUp(self):
        self.doy_mean_data = np.array([120,180,270,360])
        self.doy_outlier_data = np.array([350, 355, 364,1,3,4,220])
    
    def test_mean_doy(self):
        output = sf.mean_doy(self.doy_mean_data)
        check_data = stats.circmean(self.doy_mean_data, high=365)
        
        self.assertAlmostEqual(output,check_data)
        
    def test_median_doy(self):
        # Using days_in_year=360 to make checking correctness easier
        output = sf.median_doy(self.doy_mean_data, days_in_year=360)
        check_median = 180.0
        
        self.assertAlmostEqual(output,check_median)
        
    def test_outliers(self):
        output_outl = sf.check_outliers(self.doy_outlier_data, threshold=1.5)
        
        output_nooutl = sf.check_outliers(self.doy_outlier_data[:-1], threshold=1.5)
        self.assertEqual(int(output_outl),6)
        
        self.assertFalse(np.any(output_nooutl))

def test_calc_annual_cycle(create_synthetic_xr_dataarray):
    data = create_synthetic_xr_dataarray
    output = sf.calc_annual_cycle(data)
    assert len(output) == 365


def test_smooth_b17_data_with_nans():
    test_data = np.arange(0, 20.5, 1)
    
    index = np.array([0,7,20])
    test_data[index] = np.nan 
    
    output = sf.smooth_B17(test_data)
    
    check =  not np.all(np.isnan(output))
    assert(check == True)