

### Import for functions ###

### Imports for Relative Entropy Function ###
import scipy.stats as ss
import numpy as np
import scipy.optimize as optimize

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

def harmonic_smoothing(tseries, num_harm, time=None):
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

