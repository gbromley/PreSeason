import numpy as np
import xarray as xr
import seasonality.seasonalityfunctions as sf

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
        sf.first_harmonic_min,
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
    temp_start_index = np.where(days == startWet)[0]
    
    # double check we have enough data for last onset calculation
    if len(days[temp_start_index[-1]:]) < 180:
        
        # trim off data we can't use
        data_trimmed = data[temp_start_index[0]:temp_start_index[-1]]
        days_trimmed = days[temp_start_index[0]:temp_start_index[-1]]
        years_trimmed = years[temp_start_index[0]:temp_start_index[-1]]
        
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
        
        # this returns the index of the data not the day
        onset_index = np.argmin(cumsum_data)
        onset_day = analysis_days[onset_index]
        onset_year = analysis_years[onset_index]
        
        where_to_place = np.argwhere(unique_years == onset_year)[0][0]
        onsetDOY[where_to_place] = onset_day
        
    return onsetDOY
