
import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal

DAYS_IN_YEAR = 365


def find_sequence(arr, seq):
    """_summary_

    Args:
        arr (_type_): _description_
        seq (_type_): _description_

    Returns:
        _type_: _description_
    """    
    if len(seq) == 0:
        return -1

    # Masks for each element in the sequence
    masks = [arr == val for val in seq]

    # Find indices where the first element of the sequence occurs
    first_elem_indices = np.where(masks[0])[0]

    for idx in first_elem_indices:
        #TODO clean up the following if statement
        if all((idx + i < len(arr) and masks[i][idx + i]) for i in range(len(seq))):
            return idx

    return -1

def fourier_coefficients(tseries, num_harm, time=None):
       
    """
    Summary:
    --------
    
    Calculates fourier series for data using:
    a_0 + \sum_{n=1}^{\infty} \left[ a_n \cos\left(\frac{2\pi nt}{T}\right) + b_n \sin\left(\frac{2\pi nt}{T}\right) \right]
        
    Input:
    ------ 
    
        tseries: Mean annual daily precipitation data. Should be of length 365.
        n: Number of harmonics to calculate
        time: Integer array of len(tseries)
        
    Output:
    -------
        a_n: Array of A_n fourier coefficients for Nth harmonics
        b_n: Array of B_n fourier coefficients for Nth harmonics
        var: Ratio of explained variance of nth harmonics
        
    """
    
    length_data = len(tseries)
    # This function only works for data of length 365.
    #assert(length_data) == 365.0
    
    # Putting this here for future expansion if needed
    dt = 1.
    
    if tseries.ndim > 2:
        return(None)
    elif tseries.ndim == 2:
        time = tseries[0,:]
        data = tseries[1,:]
    elif (tseries.ndim == 1 and time == None):
        time = np.arange(1,length_data+1,1)
        data = tseries
        
    else:
        data = tseries
    
    # 0th harmonic is just the mean, not used currently
    a_0 = np.sum(data) / length_data
    
    a_n = np.zeros(num_harm)
    b_n = np.zeros(num_harm)
    var = np.zeros(num_harm)
    
    for n in np.arange(1,num_harm+1,1):
        # Already calculated 0th harmonic (a_0).
        cos_vals = np.cos(2.0 * np.pi * n * time * (1/length_data))
        sin_vals = np.sin(2.0 * np.pi * n * time * (1/length_data))
        
        a_n[n-1] = (2/length_data) * np.sum(data * cos_vals) * dt
        b_n[n-1] = (2/length_data) * np.sum(data * sin_vals) * dt
    
    #Calculating explained variance
    P_harm = [0.5 * (a**2 + b**2) for a,b in zip(a_n,b_n)]
    P_total = a_0**2 + np.sum(P_harm)
        
    var = [P/P_total for P in P_harm]    
       
    return a_n, b_n, var

def smoothing_harmonic(tseries, num_harm, time=None):
    """
    Summary:
    --------
    Smooths data using n harmonics.
    
    Input:
    ------
        tseries: Time series of data. It should be daily precipitation data of length 365
        n: Number of harmonics to calculate
        time: Index integer array of len(tseries)
    
    Output:
    -------
        smoothed: Numpy array of smoothed data using n harmonics. Should be same length as tseries.
    
    """
    data_len = len(tseries)
    
    # Handling different combos
    if (time is not None and tseries.ndim ==1):
        
        a_n, b_n, var = fourier_coefficients(tseries, num_harm, time)
    
    else:
        time = np.arange(1,data_len+1,1)
        a_n, b_n, var = fourier_coefficients(tseries, num_harm)
    
    
    # This is calculating 0th harmonic, i.e. the mean
    smoothed = np.zeros(data_len)
    smoothed[:] = np.mean(tseries)
    
    for n in np.arange(1,num_harm+1,1):
        
        smoothed = smoothed + a_n[n-1]*np.cos(2.0 * np.pi * n * time * (1/data_len)) + b_n[n-1]*np.sin(2.0 * np.pi * n * time * (1/data_len))
    
    return smoothed     

def min_first_harmonic(tseries):
    """
    Summary:
    --------
    Calculates the start day for determining wet season onset/demise. 
    Uses the minimum of the first harmonic.
    
    Input:
    ------
        tseries: Annual precip cycle, length 356.
    
    Output:
    -------
        start_day: Calculation start day in julian days.
    
    """
    ### Adding the assert so we make sure we use the correct data ###
    
    #assert(len(tseries) == 366)
    harmonic = smoothing_harmonic(tseries,num_harm=1)
    min_harmonic = np.argmin(harmonic)
    return min_harmonic

def smooth_B17(tseries, num_passes=50):
    """
    Summary:
    --------
    Smooths a time series using a 1-2-1 filter. This filter is essentially
    a moving average of window size 3.
    
    Input:
    ------
        tseries: Time series data, not limited to a particular length.
        num_passes: Number of times to apply the smoothing to the time series.
        Default is 50 times.
    
    Output:
    -------
        smoothie: Smoothed time series    
    """
    smoothie = tseries
    temp = tseries
    
    for n in np.arange(0,num_passes):
        temp[0] = 0.5*(smoothie[0]+smoothie[1])
        temp[-1] = 0.5*(smoothie[-1]+smoothie[-2])
        temp[1:-1] = 0.25*smoothie[0:-2] + 0.5*smoothie[1:-1]+0.25*smoothie[2:]
        smoothie=temp
    return smoothie


def filter3(tseries,num_passes=1):
    # Swap out convolve with the code below
    # cumsum_vec = numpy.cumsum(numpy.insert(data, 0, 0)) 
    # ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    output = tseries
    inflection_array = output
    for n in np.arange(0,num_passes):
        inflection_array[0] = np.mean(output[0:2])
        inflection_array[-1] = np.mean(output[-2:])
        inflection_array[1:-1] = signal.convolve(output, np.array([1,1,1]), "valid")/3
        output = inflection_array
    return output

def test_smooth(tseries,num_passes=1):
    output = np.zeros(len(tseries))
    inflection_array = output
    coef= np.array([1,2,1])
    output[0] = np.mean(tseries[0:2])
    output[-1] = np.mean(tseries[-2:])
    for i in np.arange(1,len(tseries) - 1,1):
        
        output[i] = np.sum(tseries[i-1:i+2] * coef / 4)
    return output

def find_ddt_onset(tseries, window=5):
    """
    Summary:
    --------
    Returns the first occurrence of an inflection point in the derivative time series.  
    
    Input:
    ------
        tseries: Time series data, not limited to a particular length.
        
    
    Output:
    -------
        output: First derivative of input series.
    
    """
    # Find 1st derivative
    deriv = np.gradient(tseries)
    
    #find the differences in the derivative
    diffs = np.diff(deriv)
    
    #Convert diffs to positve or negative
    sign_array = np.sign(diffs)
    
    test_seq = np.ones(window)
    center = int(np.ceil(window/2))
    test_seq[0:center+1] = -1
    
    index = find_sequence(sign_array, test_seq)
    
    
    
    return index



def mean_doy(input_array, days_in_year=365):
    
    """
    Summary:
    --------
    Calculates the average of days in a given year using circular statistics. Would work for 
    an average across years, if the dates can be converted to DOY. 
    
    Input:
    ------
        input_array: Numpy array of DOY. Needs to be less than 365. 

    Output:
    -------
        mean_doy: Returns the mean of an array of integer days of year.
    """
    
    #days_in_year = 365
    # Circular mean
    # Need to normalize to radians
    sin_samp = np.sin((input_array)*2.*np.pi / (days_in_year))
    cos_samp = np.cos((input_array)*2.*np.pi / (days_in_year))
    sin_sum = np.sum(sin_samp)
    cos_sum = np.sum(cos_samp)
    
    result = np.arctan2(sin_sum, cos_sum)
    
    doy_mean = result*(days_in_year)/2.0/np.pi
    
    if doy_mean < 0:
        doy_mean = doy_mean + days_in_year
    
    return doy_mean

def median_doy(input_array, days_in_year=365):
       
    """
    Summary:
    --------
    Calculates the median of days in a given year using circular statistics. Would work for 
    a median across years, if the dates can be converted to DOY. 
    
    Input:
    ------
        input_array: Numpy array of DOY. Needs to be less than 365. 

    Output:
    -------
        median_doy: Returns the median of an array of integer days of year.
    """
    
    #days_in_year = 365
    # Circular median
    # Need to normalize to radians
    sin_samp = np.sin((input_array)*2.*np.pi / (days_in_year))
    cos_samp = np.cos((input_array)*2.*np.pi / (days_in_year))
    sin_med = np.nanmedian(sin_samp)
    cos_med = np.nanmedian(cos_samp)
    
    result = np.arctan2(sin_med, cos_med)
    
    doy_med = result*(days_in_year)/2.0/np.pi
    
    if doy_med < 0:
        doy_med = doy_med + days_in_year
    
    return doy_med

def check_outliers(input_array, threshold=1.5, days_in_year=365.):
    """
    Summary:
    --------
    Calculates the outliers of input array based on IQR * threshold.
    
    Input:
    ------
        input_array: Positive array of days of year.
        threshold: Value to apply to IAR. Default is 1.5.
        days_in_year: Default set to 365, but users might want to use leap days?
    
    Output:
    -------
        outl: Array of index locations of outliers
    
    """
    
    #if np.any(np.isnan(input_array)):
        
        #input_array[np.isnan(input_array)] = np.ma.masked
    
    if np.any(input_array < 0):
        raise ValueError('Array should be entirely positive.')
        
    
    
    medians = median_doy(input_array)

    # Need to subtract the medians so the percentile calculations are correct.
    data_medians = input_array - medians
    
    # Converting dates close to the beginning and end of year
    # Idea from Bombardi
    pos=np.where(data_medians[:] > days_in_year * 0.5)[0]
    if len(pos) > 0:
        
        data_medians[pos]=data_medians[pos]-days_in_year
    
    neg=np.where(data_medians[:] < days_in_year * (-0.5))[0]
    if len(neg) > 0:
        
        data_medians[neg]=data_medians[neg]+days_in_year
    
    iqr=np.nanpercentile(data_medians,75)-np.nanpercentile(data_medians,25)
    
    outl=np.where(np.abs(data_medians) > iqr*threshold)[0]
    
    return outl

def cumul_anom(data, analysis_begin, analysis_end):
    
    # Day of year is missing integer 60 which is February 29th.
    # Need to add zero because a tuple is returned from np.where
    #start_index = np.where(days == startWet)[0]
    
    # double check we have enough data for last onset calculation
    #if len(days[temp_start_index[-1]:]) < 180:
        
        # trim off data we can't use
        #data_trimmed = data[:temp_start_index[-1]]
        #days_trimmed = days[:temp_start_index[-1]]
        #years_trimmed = years[:temp_start_index[-1]]
            
    # Reindex start days with trimmed days  
    # Need to add zero because a tuple is returned from np.where
    #start_day_index = np.where(days_trimmed == startWet)[0]
    
    
    cumsum_data = np.cumsum(data[analysis_begin:analysis_end])
    

    return cumsum_data #,onset_index

def calc_annual_cycle(data):
    annual_precip_cycle= data.groupby('time.dayofyear').mean(dim='time')
    mask = (annual_precip_cycle['dayofyear'] != 60)
    annual_precip_cycle = annual_precip_cycle.where(mask, drop=True)
    return annual_precip_cycle

def toDOY(time):
    """
    Summary:
    --------
        Returns the DOY from a date string
    
    Input:
    ------
        time: Array of time values
        
    
    Output:
    -------
        output: Numpy array of values, 1-366, corresponding to the days in the input array.
    
    """
    if not isinstance(time, pd.DatetimeIndex):
        raise TypeError('')
    DOY = np.array(pd.to_datetime(time).dayofyear)
    return DOY   


def generate_p_da(years = 5, start = 2015):
    
    """
    Summary:
        Generates synthetic precipitation data for the testing of functions, and learning.

    Raises:
        TypeError: Years needs to be an int.
        TypeError: start needs to be an int.

    Returns:
        DataArray: Xarray data array is returned.
    """
    
    if not isinstance(years, int):
        raise TypeError('Year argument needs to be an integer!')
    
    if not isinstance(start, int):
        raise TypeError('Start argument needs to be an integer')
    
    # Function to generate synthetic precipitation data
    def generate_precipitation_data(years=5, start_year=2020):
    

    # Create a date range for the specified number of years
        dates = pd.date_range(start=f'{years}-01-01', periods=years*DAYS_IN_YEAR)

        # Create a baseline seasonal cycle of precipitation (sinusoidal)
        # This simulates higher precipitation in certain months
        seasonal_cycle = np.sin(2 * np.pi * dates.dayofyear / DAYS_IN_YEAR)

        # Add a stochastic component using Gaussian noise
        stochastic_component = np.random.normal(0, 1, size=len(dates))

        # Combine the seasonal and stochastic components to get the final precipitation data
        # Adjust the amplitude and mean to realistic values (e.g., mean=3, amplitude=2)
        mean_precipitation = 3
        amplitude = 2
        precipitation_data = mean_precipitation + amplitude * seasonal_cycle + stochastic_component

        # Ensure all precipitation values are non-negative
        precipitation_data = np.maximum(precipitation_data, 0)

        # Create a DataFrame
        da = xr.DataArray(
        data=precipitation_data,
        dims=["time"],
        coords=dict(
            time=dates,
        ),
        attrs=dict(
            description="Synthetically generated precipitation data (fake).",
            units="mm",
        ),
    )
        
        
        df = pd.DataFrame({'Date': dates, 'Precipitation': precipitation_data})
        df.set_index('Date', inplace=True)

        return df





