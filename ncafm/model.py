#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pymc3 as pm

def len_jon(z, force_data, noise, prior_type = 'Jeffreys', epsilon_init = 1, sigma_init = 1, epsilon_lower = 0.01, epsilon_upper = 100, sigma_lower = 0.01, sigma_upper = 100):
    
    '''
    generates Lennard Jones force model
    
    Inputs:
    -------
    z: ndarray. [in nm]. x data of the observed force data. 
    force_data: ndarray. [in nN] Observed (noisy) force data to fit to a purely Lennard Jones model
    noise: float or ndarray (of size force_data) [in nN] 
    
    Optional inputs:
    ---------------
    prior_type: string. 'Jefferys' to use a Jeffreys prior, 'uniform' to use a uniform prior. Prior defines the prior on both epsilon and sigma. Default is Jeffreys.
    
    epsilon_init: float. [in aJ] starting point for epsilon. default = 1.
    sigma_init: float. [in nm] starting point for sigma. default = 1. 
    
    epsilon_lower: float. [in aJ] The minimum value for epsilon in the prior (the depth of the well). default = e-2.
    epsilon_upper: float. [in aJ] The maximum value for epsilon in the prior (the depth of the well). default = e2.
    
    sigma_lower: float. [in nm] The minimum value for sigma in the prior (the size of the well). default = e-2.
    sigma_upper: float. [in nm] The maximum value for sigma in the prior (the size of the well). default = e2.
    
    Returns:
    --------
    lj_model: generated model
    
    '''
    lj_model = pm.Model()
    with lj_model:

        if prior_type == 'Jeffreys':
            #Jefferys prior
            logepsilon = pm.Uniform('logepsilon', np.log10(epsilon_lower), np.log10(epsilon_upper), testval = np.log10(epsilon_init))
            logsigma = pm.Uniform('logsigma', np.log10(sigma_lower), np.log10(sigma_upper), testval = np.log10(sigma_init))
        
            #convert to reg parameters:
            epsilon = pm.Deterministic('epsilon', 10**(logepsilon))
            sigma = pm.Deterministic('sigma', 10**(logsigma))
        
        elif prior_type == 'uniform' or 'Uniform':
            epsilon = pm.Uniform('epsilon', epsilon_lower, epsilon_upper, testval = epsilon_init)
            sigma = pm.Uniform('epsilon', sigma_lower, sigma_upper, testval = sigma_init)
            
        else:
            return ValueError('prior_type does not correspond to a defined prior type. Options: Jeffreys, uniform')

        #model
        force_model = 4*epsilon*(12*sigma**12/z**13 - 6*sigma**6/z**7)

        # Likelihood of observations (i.e. noise around model)
        measurements = pm.Normal('force', mu=force_model, sigma=noise, observed=force_data)
    
    return lj_model

def vdw_sph(z_input, force_data, noise, hamaker, fit_z0 = False, prior_type = 'Jeffreys', rep_factor_init = 100, radius_init = 25, rep_factor_lower = 0.1, rep_factor_upper = 10000, radius_var = 20):
    
    '''
    Generates a force model which includes a repulsive (~1/z^3) and an attracrive vdW force derived from a spherical tip.
    
    Inputs:
    -------
    z_input: ndarray. [in nm]. x data of the observed force data. 
    force_data: ndarray. [in nN] Observed (noisy) force data to fit to a purely Lennard Jones model
    noise: float or ndarray (of size force_data) [in nN] 
    hamaker: float. Hamaker's Constant. [in aJ] The constant is defined for each pair of materials. User must calculate the value. 
    
    Optional inputs:
    ---------------
    prior_type: string. 'Jefferys' to use a Jeffreys prior, 'uniform' to use a uniform prior. Prior defines the prior the repulsive factor 'rep_factor'. Default is Jeffreys.
    
    rep_factor_init: float. [in aJ*nm^2] starting point for epsilon. default = 100.
    radius_init: float. [in nm] starting point for radius. default = 25.
    theta_init: float. [in degrees] starting point for hanf-angle opening of the conical tip. default = 35.
    
    rep_factor_lower: float. [in aJ*nm^2] The minimum value for repulsive factor in the prior (the depth of the well). 
                    default = e-1.
    rep_factor_upper: float. [in aJ*nm^2] The maximum value for repuslive in the prior (the depth of the well). 
                    default = e5.
    
    radius_var: float. [in nm] The variance for the radius. Prior defined by a gamma function. 
                    The mean and the variance will be converted to alpha and beta for a gamma dist. 
                    Default = 20
    
    Returns:
    --------
    m1_z3_rep: generated force model for vdW forces from a spherical tip and planar sample.
    
    '''
    m1_z3_rep = pm.Model()

    with m1_z3_rep:
        
        if prior_type == 'Jeffreys':
            #Jefferys prior
            log_rep_factor = pm.Uniform('log_rep_factor', np.log10(rep_factor_lower), np.log10(rep_factor_upper), testval = np.log10(rep_factor_init))
        
            #convert to reg parameters:
            rep_factor = pm.Deterministic('rep_factor', 10**(log_rep_factor))
        
        elif prior_type == 'uniform' or 'Uniform':
            rep_factor = pm.Uniform('rep_factor', rep_factor_lower, rep_factor_upper, testval = rep_factor_init)
            
        else:
            return ValueError('prior_type does not correspond to a defined prior type. Options: Jeffreys, uniform')

        #Gamma prior on radius
        radius = pm.Gamma('radius', mu = radius_init, sigma = radius_var, testval = radius_init)
        
        if fit_z0 == True:
            #Should make this a truncated normal with the min being where z starts to go negative.
            z_0 = pm.Normal('z offset', mu=0, sigma = 1, testval = 0)
            z = z_input - z_0
        else:
            z = z_input

        #model
        force_model = rep_factor/z**3 - 2*hamaker*radius**3/(3*z**2*(z+2*radius)**2)

        # Likelihood of observations (i.e. noise around model)
        measurements = pm.Normal('force', mu=force_model, sigma=noise, observed=force_data)
    
    return m1_z3_rep

def vdw_cone(z_input, force_data, noise, hamaker, fit_z0 = False, prior_type = 'Jeffreys', rep_factor_init = 100, radius_init = 25, rep_factor_lower = 0.1, rep_factor_upper = 10000, radius_var = 20):
    
    '''
    Generates a force model which includes a modified repulsive term from the Lennard Jones force (~1/z^3) derived from squaring the highest power of z in the vdw potential and an attracrive vdW force derived from by a conical tip.
    Notice: this could maybe be 1/z^2, but 1/z^3 seems to work 
    
    Inputs:
    -------
    z_input: ndarray. [in nm]. x data of the observed force data. 
    force_data: ndarray. [in nN] Observed (noisy) force data to fit to a purely Lennard Jones model
    noise: float or ndarray (of size force_data) [in nN] 
    hamaker: float. Hamaker's Constant. [in aJ] The constant is defined for each pair of materials. User must calculate the value. 
    
    Optional inputs:
    ---------------
    prior_type: string. 'Jefferys' to use a Jeffreys prior, 'uniform' to use a uniform prior. Prior defines the prior the Lennard Jones repulsive factor 'lj_rep'. Default is Jeffreys.
    
    rep_factor_init: float. [in aJ*nm^2] starting point for epsilon. default = 100.
    radius_init: float. [in nm] starting point for radius. default = 25.
    theta_init: float. [in degrees] starting point for hanf-angle opening of the conical tip. default = 35.
    
    rep_factor_lower: float. [in aJ*nm^2] The minimum value for repulsive factor in the prior (the depth of the well). 
                    default = e-1.
    rep_factor_upper: float. [in aJ*nm^2] The maximum value for repuslive in the prior (the depth of the well). 
                    default = e5.
    
    radius_var: float. [in nm] The variance for the radius. Prior defined by a gamma function. 
                    The mean and the variance will be converted to alpha and beta for a gamma dist. 
                    Default = 20
    
    Returns:
    --------
    m2_z3_rep: generated model
    
    '''
    m2_z3_rep = pm.Model()

    with m2_z3_rep:
        
        if prior_type == 'Jeffreys':
            #Jefferys prior
            log_rep_factor = pm.Uniform('log_rep_factor', np.log10(rep_factor_lower), np.log10(rep_factor_upper), testval = np.log10(rep_factor_init))
        
            #convert to reg parameters:
            rep_factor = pm.Deterministic('rep_factor', 10**(log_rep_factor))
        
        elif prior_type == 'uniform' or 'Uniform':
            rep_factor = pm.Uniform('rep_factor', rep_factor_lower, rep_factor_upper, testval = rep_factor_init)
            
        else:
            return ValueError('prior_type does not correspond to a defined prior type. Options: Jeffreys, uniform')

        #truncated normal on theta
        theta = pm.TruncatedNormal('theta', mu=theta_init, sigma=theta_var, lower=0, upper=90, testval=theta_init)

        theta_rad = np.deg2rad(theta)
        
        if fit_z0 == True:
            #Should make this a truncated normal with the min being where z starts to go negative.
            z_0 = pm.Normal('z offset', mu=0, sigma = 1, testval = 0)
            z = z_input - z_0
        else:
            z = z_input

        #model
        force_model = rep_factor/z**3 - hamaker*np.tan(theta_rad)**2/(6*z)

        # Likelihood of observations (i.e. noise around model)
        measurements = pm.Normal('force', mu=force_model, sigma=noise, observed=force_data)
    
    return m2_z3_rep

def vdw_sph_cone(z_input, force_data, noise, hamaker, fit_z0 = False, prior_type = 'Jeffreys', rep_factor_init = 100, radius_init = 25, theta_init = 35, rep_lower = 0.1, rep_upper = 10**5, radius_var = 10, theta_var = 15):
    
    '''
    Generates a force model which includes the repulsive term from the Lennard Jones force that goes as ~ 1/z^3 and vdW force for the attractive term physically motivated by a sphere at the end of a cone.
    
    Inputs:
    -------
    z: ndarray. [in nm]. x data of the observed force data. 
    force_data: ndarray. [in nN] Observed (noisy) force data to fit to a purely Lennard Jones model
    noise: float or ndarray (of size force_data) [in nN] 
    hamaker: float. Hamaker's Constant. [in aJ] The constant is defined for each pair of materials. User must calculate the value. 
    
    Optional inputs:
    ---------------
    prior_type: string. 'Jefferys' to use a Jeffreys prior, 'uniform' to use a uniform prior. Prior defines the prior the Lennard Jones repulsive factor 'lj_rep'. Default is Jeffreys.
    
    rep_factor_init: float. [in aJ*nm^12] starting point for th repulsive factor. default = 100.
    radius_init: float. [in nm] starting point for radius. default = 25.
    theta_init: float. [in degrees] starting point for hanf-angle opening of the conical tip. default = 35.
    
    rep_factor_lower: float. [in aJ*nm^2] The minimum value for repulsive factor in the prior (the depth of the well). 
                    default = e-1.
    rep_factor_upper: float. [in aJ*nm^2] The maximum value for repuslive in the prior (the depth of the well). 
                    default = e5.
    
    radius_var: float. [in nm] The variance for the radius. Prior defined by a gamma function. 
                    The mean and the variance will be converted to alpha and beta for a gamma dist. 
                    Default = 20
    
    theta_var: float. [in degrees]. Standard deviation of the prior on of the half-angle opening. 
                    The distribition is centered at theta_init. Default variation = 15.
    
    Returns:
    --------
    m3_z3_model: generated model
    
    '''
    m3_z3_model = pm.Model()

    with m3_z3_model:
        
        if prior_type == 'Jeffreys':
            #Jefferys prior
            log_rep = pm.Uniform('log_lj_rep', np.log10(rep_lower), np.log10(rep_upper), testval = np.log10(rep_init))
        
            #convert to reg parameters:
            rep = pm.Deterministic('lj_rep', 10**(log_rep))
        
        elif prior_type == 'uniform' or 'Uniform':
            rep = pm.Uniform('lj_rep', rep_lower, rep_upper, testval = rep_init)
            
        else:
            return ValueError('prior_type does not correspond to a defined prior type. Options: Jeffreys, uniform')

        #Gamma prior on radius
        radius = pm.Gamma('radius', mu = radius_init, sigma = radius_var, testval = radius_init)

        #truncated normal on theta
        theta = pm.TruncatedNormal('theta', mu=theta_init, sigma=theta_var, lower=0, upper=90, testval=theta_init)

        theta_rad = np.deg2rad(theta)
        
        if fit_z0 == True:
            #Should make this a truncated normal with the min being where z starts to go negative.
            z_0 = pm.Normal('z offset', mu=0, sigma = 1, testval = 0)
            z = z_input - z_0
        else:
            z = z_input

        #model
        force_model = rep/z**3 - hamaker/6*( radius/z**2 
                                        + radius*(1-np.sin(theta_rad))/(z*(z+ radius*(1-np.sin(theta_rad)) )) 
                                        + (np.tan(theta_rad))**2/(z+ radius*(1-np.sin(theta_rad)) ) 
                                        )

        # Likelihood of observations (i.e. noise around model)
        measurements = pm.Normal('force', mu=force_model, sigma=noise, observed=force_data)
    
    return m3_z3_model

    
def vdw_ele(z_input, force_data, noise, voltage, rep_factor, radius_init, radius_var, theta=40, fit_z0 = False, vdw_type = 'sph', effective_area_lower=0.001 , effective_area_upper=1e3):
    
    '''
    generates model for the force that includes a physically motivated vdw term and electrostatics term. 
    The paramters in the vdW force should be fit at the CPD voltage (the votlage that minimizes the electrostatics forces) and be input. Only radius should be left as a parameter from the vdW forces. 
    
   Inputs:
    -------
    z: ndarray. [in nm]. x data of the observed force data. 
    force_data: ndarray. [in nN] Observed (noisy) force data to fit to a purely Lennard Jones model
    noise: float or ndarray (of size force_data) [in nN] 
    hamaker: float. Hamaker's Constant. [in aJ] The constant is defined for each pair of materials. User must calculate the value. 
    rep_factor: float. [in aJ/nm^2] determined by fitting to the appropriate model at CPD voltage.
    theta: float. [in degrees]. Onlt required if vdw_type = 'cone' or 'sph+cone'. 
    
    radius_init: best fit value of the radius from the vdw fitting at CPD voltage
    radius_var: best fit variation of the radius from the vdw fitting at CPD voltage. 
    
    Optional inputs:
    ---------------
    
    Returns:
    --------
    m3_z3_model: generated model
    
    '''
    ele_model = pm.Model()

    with ele_model:

        #uniform on radius
        radius = pm.Gamma('radius', mu=radius_init, sigma = radius_var, testval = radius_init)
        
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
        
        if fit_z0 == True:
            #Should make this a truncated normal with the min being where z starts to go negative.
            z_0 = pm.Normal('z offset', mu=0, sigma = 1, testval = 0)
            z = z_input - z_0
        else:
            z = z_input
        
        #model
        force_model = rep/z**3 + vdw_force - 9*10**36*voltage**2*effective_area**2/z**4

        # Likelihood of observations (i.e. noise around model)
        measurements = pm.Normal('force', mu=force_model, sigma=noise, observed=force_data)
    
    return ele_model


