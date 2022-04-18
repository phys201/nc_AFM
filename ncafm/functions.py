import numpy as np

def calculate_force(a,k,f_0, z, df_data):
    
    '''
    calculates force from given frequency shift data and corresponding z positions.
    Notice, the absolute value of the force does not depend on the absolute value of the z position,
    just the difference between sequential z positions. Therefore the z input can be the 'piezo' z position;
    it doens't have to be relative to the sample.
    
    Inputs:
    -------
    a: float. amplitude of oscillation in units of m
    k: spring constant of the AFM tip. in untis of N/m
    f_0: reasonant frequency of the AFM tip. in Hz.
    z: ndarray of the height of the tip. It does not need to be height with respect to sample,
        but the differences between heights need to be accurate.
    df_data: frequency shift data at corresponding z positions. In units of Hz. 
    
    Returns:
    --------
    force_array: ndarray (of size z) of the corresponding force, calculated using the method outlined
        by Sader and Jarvis (2004) DOI: 10.1063/1.1667267 Eg (9)
    
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