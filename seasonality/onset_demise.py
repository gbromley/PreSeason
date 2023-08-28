import numpy as np
import xarray as xr
import seasonality.seasonalityfunctions as sf
#TODO Change the onset stuff to a class


# TODO create decorator function for using apply_ufunc
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

def B17_analysis_start(data):
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
    
    output = xr.apply_ufunc(
        sf.min_first_harmonic,
        data,
        input_core_dims=[["dayofyear"]],
        exclude_dims=set(["dayofyear"]),
        vectorize=True,
        dask = "parallelized",
        output_dtypes = [data.dtype]
        
    )
    return output

def onset_LM01(data, days, years, startWet):
    
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
    
    if len(days) != len(years):
        raise ValueError('Length of days and years must be the same.')
        
    # Want to make sure we get all the input years before any trimming
    unique_years = np.unique(years)
    
    onsetDOY = np.empty((len(unique_years)))
    onsetDOY[:] = np.nan
    


    # Day of year is missing integer 60 which is February 29th.
    # Need to add zero because a tuple is returned from np.where
    start_index = np.where(days == startWet)[0]
    
    
    ### looping through start dates ###
    for start_day in start_index:
        
        onset_day, onset_year = sf.cumul_anom(data, days, years, start_day)
        
        where_to_place = np.argwhere(unique_years == onset_year)[0][0]
        onsetDOY[where_to_place] = onset_day
        
    return onsetDOY

def demise_LM01(data, days, years, startWet):
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
    days = days[::-1]
    years = years[::-1]
    
    if len(days) != len(years):
        raise ValueError('Length of days and years must be the same.')
        
    # Want to make sure we get all the input years before any trimming
    unique_years = np.unique(years)
    
    demiseDOY = np.empty((len(unique_years)))
    demiseDOY[:] = np.nan
    


    # Day of year is missing integer 60 which is February 29th.
    # Need to add zero because a tuple is returned from np.where
    temp_start_index = np.where(days == startWet)[0]
    
    # double check we have enough data for last onset calculation
    if len(days[temp_start_index[-1]:]) < 180:
        
        # trim off data we can't use
        data = data[temp_start_index[0]:temp_start_index[-1]]
        days = days[temp_start_index[0]:temp_start_index[-1]]
        years = years[temp_start_index[0]:temp_start_index[-1]]
        
    # Reindex start days with trimmed days  
    # Need to add zero because a tuple is returned from np.where
    start_day_index = np.where(days == startWet)[0]
    
    ### looping through start dates ###
    for start_day in start_day_index:
        
        # Make analysis period 180 days in length. 
        #TODO #check that 180 days matters
        analysis_begin = start_day
        analysis_end = start_day + 180
        
        analysis_days = days[analysis_begin:analysis_end]
        analysis_years = years[analysis_begin:analysis_end]
        
        cumsum_data = np.cumsum(data[analysis_begin:analysis_end])
        
        # this returns the index of the data not the day
        demise_index = np.argmin(cumsum_data)
        demise_day = analysis_days[demise_index]
        demise_year = analysis_years[demise_index]
        
        where_to_place = np.argwhere(unique_years == demise_year)[0][0]
        demiseDOY[where_to_place] = demise_day
        
        
    return demiseDOY

"""
def test_onsetB17(data, days, years, startWet):
    output_onset_doy = onset_LM01(data, days, years, startWet)
    
    outlier_locs = sf.check_outliers(output_onset_doy)
"""   
    


def onset_B17(data, days, years, startWet):
    
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
    # TODO move lines 206-250 to seperate class or function
    if len(days) != len(years):
        raise ValueError('Length of days and years must be the same.')
        
    # Want to make sure we get all the input years before any trimming
    unique_years = np.unique(years)
    
    onsetDOY = np.empty((len(unique_years)))
    onsetDOY[:] = np.nan
    


    # Day of year is missing integer 60 which is February 29th.
    # Need to add zero because a tuple is returned from np.where
    temp_start_index = np.where(days == startWet)[0]
    
    # double check we have enough data for last onset calculation
    if len(days[temp_start_index[-1]:]) < 180:
        
        # trim off data we can't use
        data_trimmed = data[temp_start_index[0]:temp_start_index[-1]]
        days_trimmed = days[temp_start_index[0]:temp_start_index[-1]]
        years_trimmed = years[temp_start_index[0]:temp_start_index[-1]]
    else:
        data_trimmed = data
        days_trimmed = days
        years_trimmed = years
            
    # Reindex start days with trimmed days  
    # Need to add zero because a tuple is returned from np.where
    start_day_index = np.where(days_trimmed == startWet)[0]
    
    ### looping through start dates ###
    for start_day in start_day_index:
        
        # Make analysis period 180 days in length. 
        #TODO #check that 180 days matters
        analysis_begin = start_day
        analysis_end = start_day + 180
        
        analysis_days = days_trimmed[analysis_begin:analysis_end]
        analysis_years = years_trimmed[analysis_begin:analysis_end]
        
        cumsum_data = np.cumsum(data_trimmed[analysis_begin:analysis_end])
        
        smoothed_cs_data = sf.smooth_B17(cumsum_data)
        
        onset_index = sf.find_ddt_onset(smoothed_cs_data)
        
        
        # this returns the index of the data not the day
        onset_day = analysis_days[onset_index]
        onset_year = analysis_years[onset_index]
        
        where_to_place = np.argwhere(unique_years == onset_year)[0][0]
        onsetDOY[where_to_place] = onset_day
        
    return onsetDOY
    
def demise_B17(data, days, years, startWet):
    
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
    days = days[::-1]
    years = years[::-1] 
    
    if len(days) != len(years):
        raise ValueError('Length of days and years must be the same.')
        
    # Want to make sure we get all the input years before any trimming
    unique_years = np.unique(years)
    
    onsetDOY = np.empty((len(unique_years)))
    onsetDOY[:] = np.nan
    


    # Day of year is missing integer 60 which is February 29th.
    # Need to add zero because a tuple is returned from np.where
    temp_start_index = np.where(days == startWet)[0]
    
    # double check we have enough data for last onset calculation
    if len(days[temp_start_index[-1]:]) < 180:
        
        # trim off data we can't use
        data_trimmed = data[temp_start_index[0]:temp_start_index[-1]]
        days_trimmed = days[temp_start_index[0]:temp_start_index[-1]]
        years_trimmed = years[temp_start_index[0]:temp_start_index[-1]]
    else:
        data_trimmed = data
        days_trimmed = days
        years_trimmed = years
            
    # Reindex start days with trimmed days  
    # Need to add zero because a tuple is returned from np.where
    start_day_index = np.where(days_trimmed == startWet)[0]
    
    ### looping through start dates ###
    for start_day in start_day_index:
        
        # Make analysis period 180 days in length. 
        #TODO #check that 180 days matters
        analysis_begin = start_day
        analysis_end = start_day + 180
        
        analysis_days = days_trimmed[analysis_begin:analysis_end]
        analysis_years = years_trimmed[analysis_begin:analysis_end]
        
        cumsum_data = np.cumsum(data_trimmed[analysis_begin:analysis_end])
        
        smoothed_cs_data = sf.smooth_B17(cumsum_data)
        
        demise_index = sf.find_ddt_demise(smoothed_cs_data)
        
        
        # this returns the index of the data not the day
        onset_day = analysis_days[demise_index]
        onset_year = analysis_years[demise_index]
        
        where_to_place = np.argwhere(unique_years == onset_year)[0][0]
        onsetDOY[where_to_place] = onset_day
        
    return onsetDOY
    
