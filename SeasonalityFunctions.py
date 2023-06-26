

### Import for functions ###

### Imports for Relative Entropy Function ###
import scipy.stats as ss
import numpy as np



### Start of functions ###

'''
### Computes the relative entropy ###

Input should be daily precipitation. This function then compares the data 
to a constant seasonal cycle. 
'''
def relEntropy(var1):
    constantP = np.empty(365)
    constantP[:] = 1.0/365.0
    if np.isnan(var1.any()):
        return np.nan
    RelEnt = ss.entropy(pk=var1, qk=constantP)
    return RelEnt

# region
'''
###
Name:

Description:

Input:

Output:

###
'''

def giniCoeff(var1):
    return
# endregion

### Function used in curve fitting for fitting 1st harmonic ###
def curve_2harmonic(x, a0, a1, b1):
    output = (a0 +
    a1*np.cos(x/len(x)*2*np.pi) + b1*np.sin(x/len(x)*2*np.pi))
    return output
'''
Name: fit_curve2

Description: Funtion that fits the 1st harmonic curve and 
the explained variance of the first harmonics of a time series

Input:
    xdata: Independent variable for calculation, often time.
    ydata: Values of the time series

Output:
   params: Array of coefficients of the first harmonics
   params_covariance: Covariance of the harmonic coefficients

'''
def fit_curve2(xdata, ydata):
    guess = [np.mean(ydata), 0, 0]
    params, params_covariance = optimize.curve_fit(curve_2harmonic, xdata, ydata, guess)
    return params, params_covariance

'''
Name: fourier1

Description: Returns the minimum of the 1st harmonic of a time series

Input: 
    tseries: Time series of data that we are calculating the minimum of the 1st harmonic with.


Output:
   min: the index of the minimum of the 1st harmonic curve

'''
def fourier1(tseries):
    mtot=len(tseries)
    time=np.arange(1,mtot+1,1.)
    params, _ = fit_curve2(time,tseries)
    return np.argmin(curve_2harmonic(time,*params))

'''
###

Name: harmSeasonality

Description: Returns the ratio of the 1st and 2nd harmonic of the annual cycle of daily precipitation. This determines if the distribution of precip is uni or bi-modal.

Input:
    tseries: Time series of annual, daily precipitation. Should be length of 365 or 366 (if you want to include leap days...)

Output:
    seasonality: Float that is the ratio of the 1st and 2nd harmonic.

###
'''
def harmSeasonality(tseries):
    mtot = len(tseries)
    time=np.arange(1,mtot+1,1.)
    params, _ = fit_curve2(time,tseries)
    return (params[2]/params[1])

#region
'''
###
Name: Harmonics

Description: Mostly original function used by Bombardi et al., 2019 for calculating the harmonics of annual daily precipitation.

Input:
    tseries: Mean daily precipitation for a year. Should be length 365.
    nmodes: How many harmonics to calculate. Default is two. 
    missval: Set what the missing value is. Default is np.nan.

Output:
    harmonic1: The harmonic coefficients for the 1st harmonic.

###
'''
def Harmonics(tseries, nmodes=2,missval=np.nan):
    tot = 366 ### Dealing with only 1 year of data here
    mtot=len(tseries)
    time=np.arange(1,mtot+1,1.)
    newdim=len(tseries)  # removing missing data
    harmonic1 =np.zeros((tot))
    tdata=tseries
    svar=np.sum((tdata[:]-np.mean(tdata))**2)//(newdim-1)
    nm=nmodes
    if 2*nm > newdim:
        nm=newdim/2
    coefa=np.zeros((nm))
    coefb=np.zeros((nm))
    hvar=np.zeros((nm))
    harmonic1[:] = np.mean(tseries)
    for tt in range(0,nm):
        Ak=np.sum(tdata[:]*xu.cos(2.*np.pi*(tt+1)*time[:]/float(newdim)))
        Bk=np.sum(tdata[:]*xu.sin(2.*np.pi*(tt+1)*time[:]/float(newdim)))
        coefa[tt]=Ak*2./float(newdim)
        coefb[tt]=Bk*2./float(newdim)
        hvar[tt]=newdim*(coefa[tt]**2+coefb[tt]**2)/(2.*(newdim-1)*svar)
        harmonic1=harmonic1+coefa[tt]*np.cos(2.*np.pi*time[:]/float(tot))+coefb[tt]*np.sin(2.*np.pi*time[:]/float(tot))
        
    # if hvar[1] >= hvar[0]:
    #     return np.nan
    # elif hvar[2] >= hvar[0]:
    #     return np.nan
    # else:
    return harmonic1 
#endregion 
def calcStartWet(data):
    output = xr.apply_ufunc(
    fourier1,
    annual_precip_cycle.load(),
    input_core_dims=[["dayofyear"]],
    exclude_dims=set(["dayofyear"]),
    vectorize=True,)
    return output
  
