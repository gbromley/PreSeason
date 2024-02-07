import numpy as np
import xarray as xr
import seasonality.seasonalityfunctions as sf
import pandas as pd

# Constants Used

DAYS_IN_YEAR = 365

#TODO Change the onset stuff to a class


#TODO create decorator function for using apply_ufunc
"""demise_LM01_test = xr.apply_ufunc(
    demise_LM01,
    anomalies,
    anomalies.time,
    start_wet2,
    input_core_dims=[["time"],["time"],[]],
    exclude_dims=set(["time"]),
    output_core_dims=[["year"]],
    vectorize=True,
    dask = 'parallelized',
    #output_dtypes = 'datetime64[D]',
    #output_sizes={"data_jday": 71},
)
"""

def B17_analysis_start(data, dim='dayofyear'):
    """
    Summary:
    --------
    Finds where to begin the wet season onset analysis period. Uses the minimum of the 1st harmonic.
    
    Input:
    ------
        data: Time series of the annual precip cycle, length should be 365.
    
    Output:
    -------
        analysis_doy: The day of year that the onset analysis should begin.
    
    """
    
        #TODO Remove vectorize=True
    output = xr.apply_ufunc(
        sf.min_first_harmonic,
        data.load(),
        input_core_dims=[["dayofyear"]],
        exclude_dims=set(["dayofyear"]),
        vectorize=True,
        dask = "parallelized",
        output_dtypes = [data.dtype]
        
    )
    return output

def onset_LM01(data, startWet):
    
    output = xr.apply_ufunc(
    _onset_LM01,
    data.load(),
    data.time,
    startWet,
    input_core_dims=[["time"],["time"],[]],
    exclude_dims=set(["time"]),
    output_core_dims=[["year"]],
    vectorize= True,
    dask = 'parallelized',
    #output_dtypes = 'datetime64[D]',
    #output_sizes={"data_jday": 71},
    )
    return output

def demise_LM01(data, startWet):
    
    output = xr.apply_ufunc(
    _demise_LM01,
    data.load(),
    data.time,
    startWet,
    input_core_dims=[["time"],["time"],[]],
    exclude_dims=set(["time"]),
    output_core_dims=[["year"]],
    vectorize= True,
    dask = 'parallelized',
    #output_dtypes = 'datetime64[D]',
    #output_sizes={"data_jday": 71},
    )
    return output   

#TODO Add check for all positive data. This requres anomalies, which should not all be positive.
def _onset_LM01(data, time, startWet):    
    """
    Summary:
    --------
    Calculates the start of a wet season using the Liebmann and Marengo, 2001 methods. 
    
    Input:
    ------
        data: Daily precipitation anomalies.
        days: Numpy array of day of year (1-365), no leap.
        years: Numpy array of year for each item in days.
        startWet: The doy for when to start the cumulative sum calcuation.
    
    Output:
    -------
        onsetDOY: Array of onset dates. Length is the same as the number of years on input data.

    
    """
    
    time = pd.to_datetime(time)
    years = np.array(time.year)
    days = np.array(time.dayofyear)
    
    if len(days) != len(years):
        raise ValueError('Length of days and years must be the same.')
        
    # Want to make sure we get all the input years before any trimming
    unique_years = np.unique(years)
    
    onsetDOY = np.empty((len(unique_years)))
    onsetDOY[:] = np.nan
    
    
    

    # Day of year is missing integer 60 which is February 29th.
    # Need to add zero because a tuple is returned from np.where
    start_index = np.where(days == startWet+1)[0]
    
    
    

    
    ### looping through start dates ###
    for start_day in start_index:
        
        analysis_begin = start_day
        analysis_end = start_day + DAYS_IN_YEAR
        

        
        if (analysis_end > len(data)):
            analysis_end = len(data)
        
        analysis_days = days[analysis_begin:analysis_end]
        analysis_years = years[analysis_begin:analysis_end] 
        
        cumsum_data = sf.cumul_anom(data, analysis_begin, analysis_end)
        
        
        # this returns the index of the data not the day
        onset_index = np.argmin(cumsum_data)
        onset_day = analysis_days[onset_index]
        onset_year = analysis_years[onset_index]
        
        where_to_place = np.argwhere(unique_years == onset_year)[0][0]
        onsetDOY[where_to_place] = onset_day
        
    return onsetDOY

def _demise_LM01(data, time, startWet):
    """
    Summary:
    --------
    Calculates the end of a wet season using the Liebmann and Marengo, 2001 methods. This is a 
    retrospective calculation. 
    
    Input:
    ------
        data: Daily precipitation anomalies.
        days: Numpy array of day of year (1-365), no leap.
        years: Numpy array of year for each item in days.
        startWet: The doy for when to start the cumulative sum calcuation.
    
    Output:
    -------
        demiseDOY: Array of onset dates. Length is the same as the number of years on input data.

    
    """
    # reverse input for retrospective calculation
    
    
    data = data[::-1]
    time = time[::-1]
    
    
    demiseDOY = _onset_LM01(data, time, startWet)
        
        
    return demiseDOY

def onset_B17(data, startWet):
    
    output = xr.apply_ufunc(
    _onset_B17,
    data.load(),
    data.time,
    startWet,
    input_core_dims=[["time"],["time"],[]],
    exclude_dims=set(["time"]),
    output_core_dims=[["year"]],
    vectorize= True,
    dask = 'parallelized',
    #output_dtypes = 'datetime64[D]',
    #output_sizes={"data_jday": 71},
    )
    return output

def demise_b17(data, startWet):
    
    output = xr.apply_ufunc(
    _demise_B17,
    data.load(),
    data.time,
    startWet,
    input_core_dims=[["time"],["time"],[]],
    exclude_dims=set(["time"]),
    output_core_dims=[["year"]],
    vectorize= True,
    dask = 'parallelized',
    #output_dtypes = 'datetime64[D]',
    #output_sizes={"data_jday": 71},
    )
    return output
    
def _onset_B17(data, time, startWet):
    
    """
    Summary:
    --------
    Calculates the start of a wet season using the Bombardi et al., 2017 methods. 
    
    Input:
    ------
        data: Daily precipitation anomalies.
        days: Numpy array of day of year (1-365), no leap.
        years: Numpy array of year for each item in days.
        startWet: The doy for when to start the cumulative sum calcuation.
    
    Output:
    -------
        onsetDOY: Array of onset dates. Length is the same as the number of years on input data.

    
    """
    #TODO Move all data pre-processing to function.
    time = pd.to_datetime(time)
    years = np.array(time.year)
    days = np.array(time.dayofyear)
    
    
    #TODO These tests can move to data 
    if len(days) != len(years):
        raise ValueError('Length of days and years must be the same.')
        
    # Want to make sure we get all the input years before any trimming
    unique_years = np.unique(years)
    
    doy_b17 = np.empty(len(unique_years))
    doy_b17[:] = np.nan
    
    onsetDOY = _onset_LM01(data, time, startWet)

    # Day of year is missing integer 60 which is February 29th. 
    # Need to add zero because a tuple is returned from np.where
    
    outliers = sf.check_outliers(onsetDOY)
    
    
    if np.any(outliers):
        
        onsetDOY[outliers] = np.nan
    
    #TODO Days and startwet are off by one
    start_day_index = np.where(days == startWet+1)[0]
    
    ### looping through start dates ###
    for start_day in start_day_index:
        
        # Period over which we take the cumulative sum
        analysis_begin = start_day
        analysis_end = start_day + DAYS_IN_YEAR
        
        # Handle if it's the last chunk of data
        if (analysis_end > len(data)):
            analysis_end = len(data)
        
        analysis_days = days[analysis_begin:analysis_end]
        analysis_years = years[analysis_begin:analysis_end] 
        
        cumsum_data = sf.cumul_anom(data, analysis_begin, analysis_end)
        # this returns the index of the data not the day
        #onset_index = np.argmin(cumsum_data)
        #onset_day = analysis_days[onset_index]
        #onset_year = analysis_years[onset_index]
        #if analysis_end == len(data):
        #     extend = cumsum_data[::-1]
        #    extend = 
            
        
        smoothed_cs_data = sf.smooth_B17(cumsum_data)
        
        onset_index = sf.find_ddt_onset(smoothed_cs_data)
        
        
        
        # this returns the index of the data not the day
        #TODO Clean up saving the data so it's less funky
        if np.isnan(onset_index):
            continue
            
        onset_day = analysis_days[onset_index]
        onset_year = analysis_years[onset_index]
        
        
        where_to_place = np.argwhere(unique_years == onset_year)[0][0]
        
        doy_b17[where_to_place] = onset_day
            
        
        
        
    
    onsetDOY = doy_b17
    
    outliers_3 = sf.check_outliers(onsetDOY, threshold=3.)
    if np.any(outliers_3):
        
        onsetDOY[outliers_3] = np.nan
    
    
    return onsetDOY
    
def _demise_B17(data, time,  startWet):
    
    """
    Summary:
    --------
    Calculates the start of a wet season using the Bombardi et al., 2017 methods. 
    
    Input:
    ------
        data: Daily precipitation anomalies.
        days: Numpy array of day of year (1-365), no leap.
        years: Numpy array of year for each item in days.
        startWet: The doy for when to start the cumulative sum calcuation.
    
    Output:
    -------
        demiseDOY: Array of onset dates. Length is the same as the number of years on input data.

    
    """
    # TODO move lines 206-250 to seperate class or function
    
    data = data[::-1]
    time = data.time[::-1]
    
    demiseDOY = _onset_B17(data, time, startWet)
        
    return demiseDOY
    
