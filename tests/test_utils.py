import pytest
import os
import numpy as np
import seasonality.seasonalityfunctions as sf
import time
import scipy.stats as stats
import pandas as pd

def test_doy():
    time_array = pd.date_range('2000-01-01', freq='D', periods = 5)
    expected = [1,2,3,4,5]
    actual = sf.toDOY(time_array)
    np.testing.assert_array_equal(actual, expected)