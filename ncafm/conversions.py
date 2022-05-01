import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

def df2force(z, df_data, v, a, k, f_0, cpd_v= '0.0'):
    
    '''
    calculates force from given frequency shift data and corresponding z positions.
    Notice, the absolute value of the force does not depend on the absolute value of the z position,
    just the difference between sequential z positions. Therefore the z input can be the 'piezo' z position;
    it doens't have to be relative to the sample.
    
    Inputs:
    -------
    z: ndarray of the height of the tip. It does not need to be height with respect to sample,
        but the differences between heights need to be accurate. In units of [m]
    df_data: frequency shift dataframe at corresponding z positions at all V. In units of [Hz].
    V: string of the voltage you wish to convert
    CPD_V: strin gof the CPD voltage - used for a small shift correction to the data
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
    ydata = df_data[v]
    
    size = len(z)

    #creat empty array to store the calculation in
    force_array = np.zeros([size])
    
    #the df data need sto be shifted up based on the CPD sweep
    #At CPD df should be 0 Hz far from the sample (i.e the last 65 data points), 
    #however, it has a constant shift of ~ 0.5 Hz. 
    #Take the average of these points (to avoid noise) and shift the whole curve by that amount
    correction = np.mean(df_data[cpd_v].iloc[-65:])*0.8
    
    Omega = (ydata - correction) / f_0
    Omegadz = np.gradient(Omega) / np.gradient(xdata)

    first_term = np.zeros([size])
    second_term = np.zeros([size])
    third_term = np.zeros([size])
    correction_term = np.zeros([size])
    
    t_minus_z = xdata[1] - xdata[0]
    root_t_z = np.sqrt(t_minus_z)

    for zi in range(0, size):
        if zi < size - 1:
            t = zi+1
            integral_1 = np.trapz((Omega[t:]), x = xdata[t:]) 
            integral_2 = np.trapz(((const_2 / root_t_z) * Omega[t:]), x = xdata[t:])     
            integral_3 = np.trapz(- (const_3 / root_t_z * Omegadz[t:]), x = xdata[t:])
        
            first_term[zi] = 2 * k * (integral_1)
            second_term[zi] = 2 * k * (integral_2)
            third_term[zi] = np.real(2 * k * (integral_3))
       
        else:
        #for the last data point we can't perform the integral, 
        #so set = 0, and just have the correction term
            first_term[zi] = 0
            second_term[zi] = 0
            third_term[zi] = 0
            
        # correction terms for t=z from [2] mathematica notebook SJ
        corr1 = Omega[zi] * t_minus_z
        corr2 = 2 * const_2 * Omega[zi] * root_t_z
        corr3 = -2 * const_3 * Omegadz[zi] * root_t_z
        correction_term[zi] = 2 * k * (corr1 + corr2 + corr3)
    
    force_array = first_term+second_term+third_term+correction_term
    
    return force_array


def ptp2variation(ptp, averaging_time, sampling_rate = 10000, plot_high_freq = False):
    '''
    A function that converts the peak to peak measurement to a time-averaged Gaussian noise variation.
    
    The easiest way to do this is to convolve the high frequency noise array with an array of ones of the lenth I want to take a rolling average of. I use 'valid' so that np.convolve only starts once the 1-array fully inside the noise array.
    
    Inputs:
    -------
    ptp: float. In Hz. Peak-to-peak value of the signal when the sample is out of range.
    averaging_time: float. In s. Time to average over the high frequency noise - ie the time of the measurement,
    sampling_rate: float. sampling rate of the signal. Default on our oscilloscope is 10000 /s (100000 per 10 s). 
        This may be the resonant frequency (~20 000) for some cases.
        
    plot: boolean (optional). Make a plot of the artificially generated noise.
        
    Returns:
    -------
    averaged_noise: float. In s. The high-frequnencu noise averaged over the length of the measurement.
    
    '''
    
    half_ptp = ptp/2

    test_time = np.arange(0, 1, 1/sampling_rate)

    test_noise = scipy.stats.norm.rvs(loc=0, scale = half_ptp, size = len(test_time))
    
    if plot_high_freq == True:
        plt.plot(test_time, test_noise)
        plt.xlabel('time (s)')
        plt.ylabel('measurement (Hz)');
        plt.show()
    
    average_over_n_indices = int(len(test_noise)/len(np.arange(0,1,averaging_time)))
    
    rolling_ave_noise = np.convolve(test_noise, np.ones([average_over_n_indices]), 'valid')/average_over_n_indices
    
    averaged_noise = np.sqrt(np.mean(rolling_ave_noise**2))
    
    return averaged_noise
