import pytest
import os
import numpy as np
import preseason.tools as sf
import time
import scipy.stats as stats
import pandas as pd

### In case we need to adjust leap days, we can.
### Pandas date_range creates 366 days by default.
DAYS_IN_YEAR = 365


def test_doy():
    time_array = pd.date_range('2000-01-01', freq='D', periods = 5)
    expected = [1,2,3,4,5]
    actual = sf.toDOY(time_array)
    np.testing.assert_array_equal(actual, expected)
    



def test_len_fake_data():
    num_years = 5
    start_year = 2016 # This is a leap year, so makes a good test.
    fake_data = sf.generate_p_da(start_year, num_years)
    
    assert len(fake_data) == num_years * DAYS_IN_YEAR

def test_errors_fake_data_generation():
    with pytest.raises(TypeError):
        sf.generate_p_da('t', 2015)
        sf.generate_p_da(5, 2015.7)