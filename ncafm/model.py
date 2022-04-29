#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pymc3 as pm

def len_jon(z_input, force_data, noise, fit_z0 = False, epsilon_init = 1, sigma_init = 1, epsilon_mean = 25, epsilon_var = 25, sigma_mean = 25, sigma_var = 25):
    
    '''
    generates Lennard Jones force model
    
    Required Inputs (3):
    -------
    z: ndarray. [in nm]. x data of the observed force data. 
    force_data: ndarray. [in nN] Observed (noisy) force data to fit to a purely Lennard Jones model
    noise: float or ndarray (of size force_data) [in nN] 
    
    Optional inputs:
    ---------------
    prior_type: string. 'Jefferys' to use a Jeffreys prior, 'uniform' to use a uniform prior. Prior defines the prior on both epsilon and sigma. Default is Jeffreys.
    
    epsilon_init: float. [in aJ] starting point for epsilon. default = 1.
    sigma_init: float. [in nm] starting point for sigma. default = 1. 
    
    epsilon_mean: float. [in aJ] The mean value for epsilon in the prior (the depth of the well). default = 25.
    epsilon_var: float. [in aJ] The normal variation for epsilon in the prior (the depth of the well). default = 25.
                    The normal mean and variation are converted to alpha and beta for a Gamma distribution.
    
    sigma_mean: float. [in nm] The mean value for sigma in the prior (the size of the well). default = 25.
    sigma_var: float. [in nm] The normal variation for sigma in the prior (the size of the well). default = 25.
                    The normal mean and variance are converted to alpha and beta for a Gamma distribution.
    
    Returns:
    --------
    lj_model: generated model
    
    '''
    lj_model = pm.Model()
    with lj_model:

        if (epsilon_mean >= epsilon_var) and (sigma_mean >= sigma_var):
            epsilon = pm.Gamma('epsilon', mu = epsilon_mean, sigma = epsilon_var, testval = epsilon_init)
            sigma = pm.Gamma('sigma', mu = sigma_mean, sigma = sigma_var, testval = sigma_init)
            
        else:
            raise ValueError('variation is larger than mean for either epsilon or sigma.')
        
        if fit_z0 == True:
            #normal centered so that the last data point is 1nm above the surface
            #truncated so that there is 0 probability z can go negative.
            z_0 = pm.TruncatedNormal('z offset', mu=z_input[0]-1, sigma = 1, upper = z_input[0], testval = z_input[0]-2)
            z = z_input - z_0
        else:
            z = z_input
        
        #model
        force_model = 4*epsilon*(12*sigma**12/z**13 - 6*sigma**6/z**7)

        # Likelihood of observations (i.e. noise around model)
        measurements = pm.Normal('force', mu=force_model, sigma=noise, observed=force_data)
    
    return lj_model

def vdw_sph(z_input, force_data, noise, hamaker, fit_z0 = False, rep_factor_init = 50, radius_init = 30, rep_factor_mean = 25, rep_factor_var = 25, radius_var = 10):
    
    '''
    Generates a force model which includes a repulsive (~1/z^3) and an attracrive vdW force derived from a spherical tip.
    
    Required Inputs (4):
    -------
    z_input: ndarray. [in nm]. x data of the observed force data. 
    force_data: ndarray. [in nN] Observed (noisy) force data to fit to a purely Lennard Jones model
    noise: float or ndarray (of size force_data) [in nN] 
    hamaker: float. Hamaker's Constant. [in aJ] The constant is defined for each pair of materials. User must calculate the value. 
    
    Optional inputs:
    ---------------
    prior_type: string. 'Jefferys' to use a Jeffreys prior, 'uniform' to use a uniform prior. Prior defines the prior the repulsive factor 'rep_factor'. Default is Jeffreys.
    
    rep_factor_init: float. [in aJ*nm^2] starting point for epsilon. default = 50.
    radius_init: float. [in nm] starting point for radius. default = 25.
    theta_init: float. [in degrees] starting point for hanf-angle opening of the conical tip. default = 35.
    
    rep_factor_mean: float. [in aJ*nm^2] The mean value for repulsive factor in the prior. 
                    default = 25.
    rep_factor_var: float. [in aJ*nm^2] The normal variation for repuslive in the prior.  
                    default = 25.
                    The normal mean and variance will be converted to alpha and beta for a gamma distribution prior.
    
    radius_var: float. [in nm] The variance for the radius. Prior defined by a gamma function. 
                    The mean and the variance will be converted to alpha and beta for a gamma dist. 
                    Default = 20
    
    Returns:
    --------
    m1_z3_rep: generated force model for vdW forces from a spherical tip and planar sample.
    
    '''
    m1_z3_rep = pm.Model()

    with m1_z3_rep:
        
        if (rep_factor_mean >= rep_factor_var):
            rep_factor = pm.Gamma('rep factor', mu = rep_factor_mean, sigma = rep_factor_var, testval = rep_factor_init)
            
        else:
            raise ValueError('variation is larger than mean for the repulsive factor.')

        #Gamma prior on radius
        radius = pm.Gamma('radius', mu = radius_init, sigma = radius_var, testval = radius_init)
        
        if fit_z0 == True:
            #normal centered so that the last data point is 1nm above the surface
            #truncated so that there is 0 probability z can go negative.
            z_0 = pm.TruncatedNormal('z offset', mu=z_input[0]-1, sigma = 1, upper = z_input[0], testval = z_input[0]-2)
            z = z_input - z_0
        else:
            z = z_input

        #model
        force_model = rep_factor/z**3 - 2*hamaker*radius**3/(3*z**2*(z+2*radius)**2)

        # Likelihood of observations (i.e. noise around model)
        measurements = pm.Normal('force', mu=force_model, sigma=noise, observed=force_data)
    
    return m1_z3_rep

def vdw_cone(z_input, force_data, noise, hamaker, fit_z0 = False, rep_factor_init = 50, theta_init = 40, rep_factor_mean = 25, rep_factor_var = 25, theta_var = 15):
    
    '''
    Generates a force model which includes a modified repulsive term from the Lennard Jones force (~1/z^3) derived from squaring the highest power of z in the vdw potential and an attracrive vdW force derived from by a conical tip.
    Notice: this could maybe be 1/z^2, but 1/z^3 seems to work 
    
    Required Inputs (4):
    -------
    z_input: ndarray. [in nm]. x data of the observed force data. 
    force_data: ndarray. [in nN] Observed (noisy) force data to fit to a purely Lennard Jones model
    noise: float or ndarray (of size force_data) [in nN] 
    hamaker: float. Hamaker's Constant. [in aJ] The constant is defined for each pair of materials. User must calculate the value. 
    
    Optional inputs:
    ---------------
    prior_type: string. 'Jefferys' to use a Jeffreys prior, 'uniform' to use a uniform prior. Prior defines the prior the Lennard Jones repulsive factor 'lj_rep'. Default is Jeffreys.
    
    rep_factor_init: float. [in aJ*nm^2] starting point for epsilon. default = 100.
    theta_init: float. [in degrees] starting point for hanf-angle opening of the conical tip. default = 35.
    
    rep_factor_mean: float. [in aJ*nm^2] The mean value for repulsive factor in the prior. 
                    default = 25.
    rep_factor_var: float. [in aJ*nm^2] The normal variation for repuslive in the prior. 
                    default = 25.
                    The normal mean and variance are converted to alpha and beta for a gamma distribution prior.
    
    theta_var: float. [in degrees] The variance for the half-angle opening. Prior defined by a normal distribution.
                    default = 15
    
    Returns:
    --------
    m2_z3_rep: generated model
    
    '''
    m2_z3_rep = pm.Model()

    with m2_z3_rep:
        
        if (rep_factor_mean >= rep_factor_var):
            rep_factor = pm.Gamma('rep factor', mu = rep_factor_mean, sigma = rep_factor_var, testval = rep_factor_init)
            
        else:
            raise ValueError('variation is larger than mean for the repulsive factor.')
        
        #truncated normal on theta
        theta = pm.TruncatedNormal('theta', mu=theta_init, sigma=theta_var, lower=0, upper=90, testval=theta_init)

        theta_rad = np.deg2rad(theta)
        
        if fit_z0 == True:
            #normal centered so that the last data point is 1nm above the surface
            #truncated so that there is 0 probability z can go negative.
            z_0 = pm.TruncatedNormal('z offset', mu=z_input[0]-1, sigma = 1, upper = z_input[0], testval = z_input[0]-2)
            z = z_input - z_0
        else:
            z = z_input

        #model
        force_model = rep_factor/z**3 - hamaker*np.tan(theta_rad)**2/(6*z)

        # Likelihood of observations (i.e. noise around model)
        measurements = pm.Normal('force', mu=force_model, sigma=noise, observed=force_data)
    
    return m2_z3_rep

def vdw_cone_sph(z_input, force_data, noise, hamaker, fit_z0 = False, rep_factor_init = 50, radius_init = 30, theta_init = 40, rep_factor_mean = 25, rep_factor_var = 25, radius_var = 10, theta_var = 15):
    
    '''
    Generates a force model which includes the repulsive term from the Lennard Jones force that goes as ~ 1/z^3 and vdW force for the attractive term physically motivated by a sphere at the end of a cone.
    
    Required Inputs (4):
    -------
    z: ndarray. [in nm]. x data of the observed force data. 
    force_data: ndarray. [in nN] Observed (noisy) force data to fit to a purely Lennard Jones model
    noise: float or ndarray (of size force_data) [in nN] 
    hamaker: float. Hamaker's Constant. [in aJ] The constant is defined for each pair of materials. User must calculate the value. 
    
    Optional inputs:
    ---------------
    prior_type: string. 'Jefferys' to use a Jeffreys prior, 'uniform' to use a uniform prior. Prior defines the prior the Lennard Jones repulsive factor 'lj_rep'. Default is Jeffreys.
    
    rep_factor_init: float. [in aJ*nm^12] starting point for th repulsive factor. default = 50.
    radius_init: float. [in nm] starting point for radius. default = 30.
    theta_init: float. [in degrees] starting point for hanf-angle opening of the conical tip. default = 40.
    
    rep_factor_mean: float. [in aJ*nm^2] The mean value for repulsive factor in the prior. 
                    default = 25.
    rep_factor_var: float. [in aJ*nm^2] The normal variation for repuslive in the prior. 
                    default = 25.
                    The mean and variance will be converted to alpha and beta for a gamma distribution prior.
    
    radius_var: float. [in nm] The variance for the radius. Prior defined by a gamma function. 
                    The mean and the variance will be converted to alpha and beta for a gamma distribution prior. 
                    Default = 10
    
    theta_var: float. [in degrees]. Standard deviation of the prior on of the half-angle opening. 
                    The distribition is centered at theta_init. Default variation = 15.
    
    Returns:
    --------
    m3_z3_model: generated model
    
    '''
    m3_z3_model = pm.Model()

    with m3_z3_model:
        
        if (rep_factor_mean >= rep_factor_var):
            rep_factor = pm.Gamma('rep factor', mu = rep_factor_mean, sigma = rep_factor_var, testval = rep_factor_init)
            
        else:
            raise ValueError('variation is larger than mean for the repulsive factor.')

        #Gamma prior on radius
        radius = pm.Gamma('radius', mu = radius_init, sigma = radius_var, testval = radius_init)

        #truncated normal on theta
        theta = pm.TruncatedNormal('theta', mu=theta_init, sigma=theta_var, lower=0, upper=90, testval=theta_init)

        theta_rad = np.deg2rad(theta)
        
        if fit_z0 == True:
            #normal centered so that the last data point is 1nm above the surface
            #truncated so that there is 0 probability z can go negative.
            z_0 = pm.TruncatedNormal('z offset', mu=z_input[0]-1, sigma = 1, upper = z_input[0], testval = z_input[0]-2)
            z = z_input - z_0
        else:
            z = z_input

        #model
        force_model = rep_factor/z**3 - hamaker/6*( radius/z**2 
                                        + radius*(1-np.sin(theta_rad))/(z*(z+ radius*(1-np.sin(theta_rad)) )) 
                                        + (np.tan(theta_rad))**2/(z+ radius*(1-np.sin(theta_rad)) ) 
                                        )

        # Likelihood of observations (i.e. noise around model)
        measurements = pm.Normal('force', mu=force_model, sigma=noise, observed=force_data)
    
    return m3_z3_model

    
def vdw_ele(z_input, force_data, noise, voltage, rep_factor, radius_init, radius_var, theta=40, fit_z0 = False, vdw_type = 'sph'):
    
    '''
    generates model for the force that includes a physically motivated vdw term and electrostatics term. 
    The paramters in the vdW force should be fit at the CPD voltage (the voltage that minimizes the electrostatics forces) and be input. Only radius should be left as a parameter from the vdW forces. 
    
   Inputs:
    -------
    z: ndarray. [in nm]. x data of the observed force data. 
    force_data: ndarray. [in nN] Observed (noisy) force data to fit to a purely Lennard Jones model
    noise: float or ndarray (of size force_data) [in nN] 
    hamaker: float. Hamaker's Constant. [in aJ] The constant is defined for each pair of materials. User must calculate the value. 
    rep_factor: float. [in aJ/nm^2] determined by fitting to the appropriate model at CPD voltage.
    theta: float. [in degrees]. Only required if vdw_type = 'cone' or 'sph+cone'. 
    
    radius_init: best fit value of the radius from the vdw fitting at CPD voltage
    radius_var: best fit variation of the radius from the vdw fitting at CPD voltage. 
    
    Optional inputs:
    ---------------
    
    Returns:
    --------
    ele_model: generated model including electrostatic force
    
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
        force_model = rep/z**3 + vdw_force - 9*10**36*voltage**2*(pi*radius**2)**2/z**4

        # Likelihood of observations (i.e. noise around model)
        measurements = pm.Normal('force', mu=force_model, sigma=noise, observed=force_data)
    
    return ele_model


