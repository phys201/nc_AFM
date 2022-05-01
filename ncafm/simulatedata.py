import numpy as np
import scipy.stats

def len_jon(epsilon, sigma, noise, z_input, z_0=0):
    
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

def sph(factor, hamaker, radius, noise, z_input, z_0=0):
    
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

def cone(factor, hamaker, theta, noise, z_input, z_0 = 0):
    
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


def cone_sph(factor, hamaker, radius, theta, noise, z_input, z_0=0):
    
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


def ele(theta, radius, voltage, noise, z_input, hamaker, factor, z_0=0, vdw_type = 'sph'):
    
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