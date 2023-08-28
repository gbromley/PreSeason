
import numpy as np

from scipy import signal

# TODO move some of these functions to a utils file

def findFirst_numpy(a, b):
    #TODO fill out function data
    """
    Summary:
    --------
    Calculates fourier series for data using:
    
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
    inflection_array = np.lib.stride_tricks.sliding_window_view(a,len(b))
    result = np.where(np.all(inflection_array == b, axis=1))
    return result[0][0] if result else np.nan

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
    assert(length_data) == 365.0
    
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
        tseries: Harmonic approximation to use
    
    Output:
    -------
        start_day: Calculation start day in julian days.
    
    """
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

#these all output the same as smooth-1-2-1 but might be faster on big arrays.
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
    Returns the 1st derivative of an array
    
    Input:
    ------
        tseries: Time series data, not limited to a particular length.
        
    
    Output:
    -------
        output: First derivative of input series.
    
    """
    deriv = np.gradient(tseries)
    
    # Numpy function that returns the sign of each element
    # -1 if negative, 0 if 0, 1 if positive
    sign_array = np.sign(deriv)
    
    
    
    
    # build up the inflection window
    inflection_array = np.ones(window)
    
    # Finding the middle of the window, rounded down if odd.
    split = int(np.floor(window/2))
    
    # Make first half -1
    inflection_array[:split] = -1
    
    index = findFirst_numpy(sign_array, inflection_array)
    
    
    
    return index

def find_ddt_demise(tseries, window=5):
    """
    Summary:
    --------
    Returns the 1st derivative of an array
    
    Input:
    ------
        tseries: Time series data, not limited to a particular length.
        
    
    Output:
    -------
        output: First derivative of input series.
    
    """
    deriv = np.gradient(tseries)
    
    # Numpy function that returns the sign of each element
    # -1 if negative, 0 if 0, 1 if positive
    sign_array = np.sign(deriv)
    
    # build up the inflection window
    split = int(np.floor(window/2))
    #mid = int(np.floor(window/2))
    # build up the inflection window
    inflection_array = np.ones(window)
    inflection_array[split:] = -1
    
    index = findFirst_numpy(sign_array, inflection_array)
    
    
    
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
    
    if np.any(np.isnan(input_array)):
        
        input_array[np.isnan(input_array)] = np.ma.masked
    
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

def cumul_anom(data, days, years, start_day, days_in_year=365):
    
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
    
    
    analysis_begin = start_day
    analysis_end = start_day + days_in_year
    if (analysis_end > len(data)):
        analysis_end = len(data)

    analysis_days = days[analysis_begin:analysis_end]
    analysis_years = years[analysis_begin:analysis_end]
    
    cumsum_data = np.cumsum(data[analysis_begin:analysis_end])
    
    # this returns the index of the data not the day
    onset_index = np.argmin(cumsum_data)
    onset_day = analysis_days[onset_index]
    onset_year = analysis_years[onset_index]
    return onset_day, onset_year #,onset_index




def process_data_avail(data, days, years, startWet):
    
    # Day of year is missing integer 60 which is February 29th.
    # Need to add zero because a tuple is returned from np.where
    temp_start_index = np.where(days == startWet)[0]
    
    # double check we have enough data for last onset calculation
    if len(days[temp_start_index[-1]:]) < 180:
        
        # trim off data we can't use
        data_trimmed = data[:temp_start_index[-1]]
        days_trimmed = days[:temp_start_index[-1]]
        years_trimmed = years[:temp_start_index[-1]]
            
    # Reindex start days with trimmed days  
    # Need to add zero because a tuple is returned from np.where
    start_day_index = np.where(days_trimmed == startWet)[0]
    
    