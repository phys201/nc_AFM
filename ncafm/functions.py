import numpy as np
import matplotlib.pyplot as plt
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


def simulate_lj_data(epsilon, sigma, noise, z_input, z_0=0):
    
    '''
    generates noisy Lennard Jones force data
    
    Inputs:
    -------
    epsilon: float. The depth of the well in the L-J theory
    sigma: float. The distance to 0 potential in the L-J theory
    noise: float or ndarray (of size z) of noise to be added at each point.
    z: ndarray. the range over which the function will generate the data
    z_0: float. In nm. A z offset. Optional, default is no offset.
    
    Returns:
    --------
    data: ndarray (of size z) of the corresponding Lennard Jones force, 
        assuming a normal distribution of the noise
    
    '''
    z = z_input - z_0
    
    perfect_data = 4*epsilon*(12*sigma**12/z**13 - 6*sigma**6/z**7)
    
    if isinstance(z_input, np.ndarray) == True:
        length = len(z_input)
    else:
        length = 1
    
    #add some noise
    noisyLJ_data = perfect_data + scipy.stats.norm.rvs(loc=0, scale = noise, size = length)
    
    #I know this will create something ~ nN
    return noisyLJ_data

def simulate_data_sph(factor, hamaker, radius, noise, z_input, z_0=0):
    
    '''
    generates noisy force data which includes the new repulsive term from the Lennard Jones force (~z^-3) 
    and a physically motivated vdW force based on a simple sphere above a plane.
    
    Inputs:
    -------
    factor: float. In aJ/nm^2. The repulsive term factor. 
        Equal to epsilon*sigma^2 in the LJ model, but even less physically motivated. 
    hamaker: float. In aJ. Hamaker's constant for the specific tip and sample materials.
    radius: float. In nm. radius of the sphere of the tip.
    noise: float In nN. or ndarray (of size z) of noise to be added at each point.
    z: ndarray. In nm. the range over which the function will generate the data
    z_0: float. In nm. A z offset. Optional, default is no offset.
    
    Returns:
    --------
    data: ndarray (of size z) of the corresponding new Lennard Jones + vdW M1, 
        with noise, assuming a normal distribution of the noise
    
    '''
    
    z = z_input - z_0
    
    perfect_data = factor/z**3 - 2*hamaker*radius**3/(3*z**2*(z+2*radius)**2)
    
    #allows us to calculate one value
    if isinstance(z_input, np.ndarray) == True:
        length = len(z_input)
    else:
        length = 1
    
    #add some noise
    noisy_m1_data = perfect_data + scipy.stats.norm.rvs(loc=0, scale = noise, size = length)
    
    #I know this will create something ~ nN
    return noisy_m1_data

def simulate_data_cone(factor, hamaker, theta, noise, z_input, z_0 = 0):
    
    '''
    generates noisy force data which includes the new repulsive term from the Lennard Jones force (~z^-3) 
    and a physically motivated vdW force based on a simple sphere above a plane.
    
    Inputs:
    -------
    factor: float. In aJ/nm^2. The repulsive term factor. 
        Equal to epsilon*sigma^2 in the LJ model, but even less physically motivated. 
    hamaker: float. In aJ. Hamaker's constant for the specific tip and sample materials.
    theta: float. In degrees. half-angle opening of the tip.
    noise: float In nN. or ndarray (of size z) of noise to be added at each point.
    z: ndarray. In nm. the range over which the function will generate the data
    z_0: float. In nm. A z offset. Optional, default is no offset.
    
    Returns:
    --------
    data: ndarray (of size z) of the corresponding new Lennard Jones + vdW M1, 
        with noise, assuming a normal distribution of the noise
    
    '''
    theta_rad = np.deg2rad(theta)
    
    z = z_input - z_0
    
    perfect_data = factor/z**3 - hamaker*np.tan(theta_rad)**2/(6*z)
    
    #allows us to calculate one value
    if isinstance(z_input, np.ndarray) == True:
        length = len(z_input)
    else:
        length = 1
    
    #add some noise
    noisy_m2_data = perfect_data + scipy.stats.norm.rvs(loc=0, scale = noise, size = length)
    
    #I know this will create something ~ nN
    return noisy_m2_data


def simulate_data_cone_sph(factor, hamaker, radius, theta, noise, z_input, z_0=0):
    
    '''
    generates noisy force data which includes the new repulsive term from the Lennard Jones force (~z^-3) 
    and a physically motivated vdW force.
    
    Inputs:
    -------
    epsilon: float. In nV. The depth of the well in the L-J theory. 
    sigma: float. In nm. The distance to 0 potential in the L-J theory.
    hamaker: float. In nV. Hamaker's constant for the specific tip and sample materials.
    radius: float. In nm. radius of the sphere of the tip.
    theta: float. In degrees. half-angle opening of the conical part of the tip. 
    noise: float In nN. or ndarray (of size z) of noise to be added at each point.
    z: ndarray. In nm. the range over which the function will generate the data
    z_0: float. In nm. A z offset. Optional, default is no offset.
    
    Returns:
    --------
    data: ndarray (of size z) of the corresponding new Lennard Jones + vdW M3, 
        with noise, assuming a normal distribution of the noise
    
    '''
    theta_rad = np.deg2rad(theta)
    
    z = z_input - z_0
    
    perfect_data = factor/z**3 - hamaker/6*(radius/z**2 
                                    + radius*(1-np.sin(theta_rad))/(z*(z+radius*(1-np.sin(theta_rad)))) 
                                    + np.tan(theta_rad)**2/(z+radius*(1-np.sin(theta_rad))))
    
    #allows us to calculate one value
    if isinstance(z_input, np.ndarray) == True:
        length = len(z_input)
    else:
        length = 1
    
    #add some noise
    noisy_m3_rep_data = perfect_data + scipy.stats.norm.rvs(loc=0, scale = noise, size = length)
    
    #I know this will create something ~ nN
    return noisy_m3_rep_data


def simulate_data_ele(theta, radius, voltage, noise, z_input, hamaker, factor, z_0=0, vdw_type = 'sph'):
    
    '''
    generates noisy force data which includes physically motivated vdw term and electrostatics term.
    
    Inputs:
    -------
    factor: float. In aJ/nm^2. The repulsive term factor. 
    radius: float. In nm. radius of the sphere of the tip.
    voltage: float. the voltage that minimizes the electrostatics forces
    noise: float In nN. or ndarray (of size z) of noise to be added at each point.
    z: ndarray. In nm. the range over which the function will generate the data.
    z_0: float. In nm. A z offset. Optional, default is no offset.
    vdw_type: string. Specifies the vdW force geometry as either sphere, cone, or sphere+cone
    hamaker: float. In nV. Hamaker's constant for the specific tip and sample materials.
    theta: float. [in degrees]. Only required if vdw_type = 'cone' or 'sph+cone'.
    
    Returns:
    --------
    data: ndarray (of size z) of the corresponding new vdw term + electrostatics term, 
        with noise, assuming a normal distribution of the noise
    
    '''
    theta_rad = np.deg2rad(theta)
    
    z = z_input - z_0
    
    if vdw_type == 'sph':
        vdw_force = - 2*hamaker*radius**3/(3*z**2*(z+2*radius)**2)
            
    elif vdw_type == 'cone':
        vdw_force = - hamaker*np.tan(theta_rad)**2/(6*z)
            
    elif vdw_type == 'sph+cone':
        vdw_force = - hamaker/6*( radius/z**2 
                                        + radius*(1-np.sin(theta_rad))/(z*(z+ radius*(1-np.sin(theta_rad)) )) 
                                        + (np.tan(theta_rad))**2/(z+ radius*(1-np.sin(theta_rad)) ) 
                                        )  
    else:
        return ValueError('vdw_type does not correspond to a defined model type. Options: sph, cone, sph+cone')
    
    perfect_data = factor/z**3 + vdw_force - 9*10**36*voltage**2*(np.pi*radius**2)**2/z**4

    #allows us to calculate one value
    if isinstance(z_input, np.ndarray) == True:
        length = len(z_input)
    else:
        length = 1
    
    #add some noise
    noisy_m4_rep_data = perfect_data + scipy.stats.norm.rvs(loc=0, scale = noise, size = length)
    
    #I know this will create something ~ nN
    return noisy_m4_rep_data