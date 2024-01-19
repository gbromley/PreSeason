import pytest
import os
import numpy as np
import seasonality.seasonalityfunctions as sf
import time
import scipy.stats as stats
import pandas as pd

DAYS_IN_YEAR = 365