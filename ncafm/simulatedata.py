import numpy as np
import scipy.stats

def len_jon(z_input, epsilon, sigma, noise, z_0=0):
    
    '''
    generates noisy Lennard Jones force data
    
    Inputs:
    -------
    z_input: ndarray. the range over which the function will generate the data
    epsilon: float. The depth of the well in the L-J theory
    sigma: float. The distance to 0 potential in the L-J theory
    noise: float or ndarray (of size z) of noise to be added at each point.
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
    
    noisy_data = perfect_data + scipy.stats.norm.rvs(loc=0, scale = noise, size = length)
    return noisy_data
        

def sph(z_input, factor, alpha, hamaker, radius, noise, z_0=0):
    
    '''
    generates noisy force data which includes the new repulsive term from the Lennard Jones force (~z^-3) 
    and a physically motivated vdW force based on a simple sphere above a plane.
    
    Inputs:
    -------
    z_input: ndarray. In nm. the range over which the function will generate the data
    factor: float. In aJ/nm. The repulsive term factor. 
    alpha: float. In nm^-1. The repulsive term length scale.
    hamaker: float. In aJ. Hamaker's constant for the specific tip and sample materials.
    radius: float. In nm. radius of the sphere of the tip.
    noise: float In nN. or ndarray (of size z) of noise to be added at each point.
    z_0: float. In nm. A z offset. Optional, default is no offset.
    
    Returns:
    --------
    data: ndarray (of size z) of the corresponding new Lennard Jones + vdW M1, 
        with noise, assuming a normal distribution of the noise
    
    '''
    
    z = z_input - z_0
    
    perfect_data = factor*np.exp(-alpha*z) - 2*hamaker*radius**3/(3*z**2*(z+2*radius)**2)
    
    #allows us to calculate one value
    if isinstance(z_input, np.ndarray) == True:
        length = len(z_input)
    else:
        length = 1
    
    noisy_data = perfect_data + scipy.stats.norm.rvs(loc=0, scale = noise, size = length)
    return noisy_data

def cone(z_input, factor, alpha, hamaker, theta, noise, z_0 = 0):
    
    '''
    generates noisy force data which includes the new repulsive term from the Lennard Jones force (~z^-3) 
    and a physically motivated vdW force based on a simple sphere above a plane.
    
    Inputs:
    -------
    z_input: ndarray. In nm. the range over which the function will generate the data
    factor: float. In aJ/nm. The repulsive term factor. 
    alpha: float. In nm^-1. The repulsive term length scale. 
    hamaker: float. In aJ. Hamaker's constant for the specific tip and sample materials.
    theta: float. In degrees. half-angle opening of the tip.
    noise: float In nN. or ndarray (of size z) of noise to be added at each point.
    z_0: float. In nm. A z offset. Optional, default is no offset.
    
    Returns:
    --------
    data: ndarray (of size z) of the corresponding new Lennard Jones + vdW M1, 
        with noise, assuming a normal distribution of the noise
    
    '''
    theta_rad = np.deg2rad(theta)
    
    z = z_input - z_0
    
    perfect_data = factor*np.exp(-alpha*z) - hamaker*np.tan(theta_rad)**2/(6*z)
    
    #allows us to calculate one value
    if isinstance(z_input, np.ndarray) == True:
        length = len(z_input)
    else:
        length = 1
    
    
    noisy_data = perfect_data + scipy.stats.norm.rvs(loc=0, scale = noise, size = length)
    return noisy_data


def cone_sph(z_input, factor, alpha, hamaker, radius, theta, noise, z_0=0):
    
    '''
    generates noisy force data which includes the new repulsive term from the Lennard Jones force (~z^-3) 
    and a physically motivated vdW force.
    
    Inputs:
    -------
    z_input: ndarray. In nm. the range over which the function will generate the data
    factor: float. In aJ/nm. The repulsive term factor. 
    alpha: float. In nm^-1. The repulsive term length scale.
    hamaker: float. In nV. Hamaker's constant for the specific tip and sample materials.
    radius: float. In nm. radius of the sphere of the tip.
    theta: float. In degrees. half-angle opening of the conical part of the tip. 
    noise: float In nN. or ndarray (of size z) of noise to be added at each point.
    z_0: float. In nm. A z offset. Optional, default is no offset.
    
    Returns:
    --------
    data: ndarray (of size z) of the corresponding new Lennard Jones + vdW M3, 
        with noise, assuming a normal distribution of the noise
    
    '''
    theta_rad = np.deg2rad(theta)
    
    z = z_input - z_0
    
    perfect_data = factor*np.exp(-alpha*z) - hamaker/6*(radius/z**2 
                                    + radius*(1-np.sin(theta_rad))/(z*(z+radius*(1-np.sin(theta_rad)))) 
                                    + np.tan(theta_rad)**2/(z+radius*(1-np.sin(theta_rad))))
    
    #allows us to calculate one value
    if isinstance(z_input, np.ndarray) == True:
        length = len(z_input)
    else:
        length = 1
    
    noisy_data = perfect_data + scipy.stats.norm.rvs(loc=0, scale = noise, size = length)
    return noisy_data


def ele(z_input, factor, alpha, hamaker, radius, voltage, noise, theta = 30, z_0=0, vdw_type = 'sph'):
    
    '''
    generates noisy force data which includes physically motivated vdw term and electrostatics term.
    
    Inputs:
    -------
    z_input: ndarray. In nm. the range over which the function will generate the data.
    factor: float. In aJ/nm. The repulsive term factor. 
    alpha: float. In nm^-1. The repulsive term length scale.
    hamaker: float. In nV. Hamaker's constant for the specific tip and sample materials.
    radius: float. In nm. radius of the sphere of the tip. 
    voltage: float. In V. the voltage applied minus the votlage that minimizes the electrostatics forces
    noise: float In nN. or ndarray (of size z) of noise to be added at each point.
    
    theta: float. [in degrees]. Only required if vdw_type = 'cone' or 'sph+cone'.
    z_0: float. In nm. A z offset. Optional, default is no offset.
    vdw_type: string. Specifies the vdW force geometry as either sphere, cone, or sphere+cone
    
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
    
    
    epsilon_0 = 8.854*10**-3 #nN/V^2
    perfect_data = factor*np.exp(-alpha*z) + vdw_force - np.pi*epsilon_0*voltage**2*(np.pi*radius**2)**2/z**4
    
    #allows us to calculate one value
    if isinstance(z_input, np.ndarray) == True:
        length = len(z_input)
    else:
        length = 1
    
    noisy_data = perfect_data + scipy.stats.norm.rvs(loc=0, scale = noise, size = length)
    return noisy_data