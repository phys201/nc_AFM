import numpy as np
import scipy.stats

def df2force(z, df_data, a, k, f_0):
    
    '''
    calculates force from given frequency shift data and corresponding z positions.
    Notice, the absolute value of the force does not depend on the absolute value of the z position,
    just the difference between sequential z positions. Therefore the z input can be the 'piezo' z position;
    it doens't have to be relative to the sample.
    
    Inputs:
    -------
    z: ndarray of the height of the tip. It does not need to be height with respect to sample,
        but the differences between heights need to be accurate. In units of [m]
    df_data: frequency shift data at corresponding z positions. In units of [Hz]. 
    a: float. amplitude of oscillation in units of [m]
    k: spring constant of the AFM tip. in untis of [N/m]
    f_0: reasonant frequency of the AFM tip. in [Hz].
    
    Returns:
    --------
    force_array: ndarray (of size z - 1) of the corresponding force in units of [N], calculated using the method outlined
        by Sader and Jarvis (2004) DOI: 10.1063/1.1667267 Eg (9).
        
        Notice: the resulting force array is smaller than the input z and df arrays by 1 due to taking a derivative. The final data point should be removed from z. 
    
    '''
    
    root_amplitude = np.sqrt(a)

    const_2 = root_amplitude / (8*np.sqrt(np.pi)) #units sqrt m
    const_3 = root_amplitude**3 / np.sqrt(2)      #units m * sqrt m

    xdata = z
    ydata = df_data
    
    new_size = len(z) -1

    #creat empty array to store the calculation in
    force_array = np.zeros([new_size])
    
    Omega = ydata / f_0
    Omegadz = np.gradient(Omega) / np.gradient(xdata)

    first_term = np.zeros([new_size])
    second_term = np.zeros([new_size])
    third_term = np.zeros([new_size])
    correction_term = np.zeros([new_size])

    for zi in range(0, new_size):
        t = zi+1
        root_t_z = np.sqrt((xdata[t:] - xdata[zi]))
    
        integral_1 = np.trapz((Omega[t:]), x = xdata[t:]) 
        integral_2 = np.trapz(((const_2 / root_t_z) * Omega[t:]), x = xdata[t:])     
        integral_3 = np.trapz(- (const_3 / root_t_z * Omegadz[t:]), x = xdata[t:])
        
        first_term[zi] = 2 * k * (integral_1)
        second_term[zi] = 2 * k * (integral_2)
        third_term[zi] = np.real(2 * k * (integral_3))
    
        # correction terms for t=z from [2] mathematica notebook SJ
        corr1 = Omega[zi] * (xdata[t]-xdata[zi])
        corr2 = 2 * const_2 * Omega[zi] * np.sqrt(xdata[t] - xdata[zi])
        corr3 = -2 * const_3 * Omegadz[zi] * np.sqrt(xdata[t] - xdata[zi])
        correction_term[zi] = 2 * k * (corr1 + corr2 + corr3)
    
    force_array = first_term+second_term+third_term+correction_term
    
    return force_array


def ptp2variation(ptp, averaging_time, sampling_rate = 100000, plot = False):
    '''
    A function that converts the peak to peak measurement to a time-averaged Gaussian noise variation.
    
    The easiest way to do this is to convolve the high frequency noise array with an array of ones of the lenth I want to take a rolling average of. I use 'valid' so that np.convolve only starts once the 1-array fully inside the noise array.
    
    Inputs:
    -------
    ptp: float. In Hz. Peak-to-peak value of the signal when the sample is out of range.
    averaging_time: float. In s. Time to average over the high frequency noise - ie the time of the measurement,
    sampling_rate: float. sampling rate of the signal. Default on our oscilloscope is 10000 /s. 
        This may be the resonant frequency (~20 000) for some cases.
        
    plot: boolean (optional). Make a plot of the artificially generated noise.
        
    Returns:
    -------
    averaged_noise: float. In s. The high-frequnencu noise averaged over the length of the measurement.
    
    '''
    
    half_ptp = ptp/2
    f_sampling = 100000

    test_time = np.arange(0,1, 1/f_sampling)

    test_noise = scipy.stats.norm.rvs(loc=0, scale = half_ptp, size = len(test_time))
    
    average_over_n_indices = int(len(test_noise)/len(np.arange(0,1,averaging_time)))
    
    rolling_ave_noise = np.convolve(test_noise, np.ones([average_over_n_indices]), 'valid')/average_over_n_indices
    
    averaged_noise = np.abs(np.mean(rolling_ave_noise))
    
    return averaged_noise