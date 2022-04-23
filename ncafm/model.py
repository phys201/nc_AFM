#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 02:39:33 2022
@author: Talha
"""

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az
import seaborn as sns

def len_jon_model(force_data, noise, epsilon_initial_guess, sigma_initial_guess, epsilon_lower = 0.0001, epsilon_upper = 1000, sigma_lower = 0.0001, sigma_upper = 1000):
    
    '''
    generates Lennard Jones force model
    
    Inputs:
    -------
    
    force_data: ndarray. Observed (noisy) Lennard Jones force data
    noise: float or ndarray (of size force_data)
    
    epsilon_lower: float. The minimum value for epsilon in the prior (the depth of the well)
    epsilon_upper: float. The maximum value for epsilon in the prior (the depth of the well)
    
    sigma_lower: float. The minimum value for sigma in the prior (the sizw of the well)
    sigma_upper: float. The maximum value for sigma in the prior (the sizw of the well)
    
    Returns:
    --------
    lj_model: generated model
    
    '''
    lj_model = pm.Model()
    with lj_model:

        #Jefferys prior from 0.0001 (10e-4) to 1000 e3
        logepsilon = pm.Uniform('logepsilon', np.log10(epsilon_lower), np.log10(epsilon_upper), testval = np.log10(epsilon_initial_guess))
        logsigma = pm.Uniform('logsigma', np.log10(sigma_lower), np.log10(sigma_upper), testval = np.log10(sigma_initial_guess))
        
        #convert to reg parameters:
        epsilon = pm.Deterministic('epsilon', 10**(logepsilon))
        sigma = pm.Deterministic('sigma', 10**(logsigma))

        #model
        force_model = 4*epsilon*(12*sigma**12/z**13 - 6*sigma**6/z**7)

        # Likelihood of observations (i.e. noise around model)
        measurements = pm.Normal('force', mu=force_model, sigma=noise, observed=force_data)
    
    return lj_model
    

def vdwm3_model(epsilon, sigma, radius, z, f_testdata_m3):
    
    '''
    generates force model which includes the repulsive term from the Lennard Jones force and a physically motivated 
    vdW force.
    
    Inputs:
    -------
    epsilon: float. In nV. The depth of the well in the L-J theory. 
    sigma: float. In nm. The distance to 0 potential in the L-J theory.
    radius: float. In nm. radius of the sphere of the tip.
    z: ndarray. In nm. the range over which the function will generate the data
    f_testdata_m3: ndarray. Simulated noisy Lennard Jones + vdW force data
    
    Returns:
    --------
    m3_model: Lennard Jones + vdW  model
    
    '''
    m3_model = pm.Model()

    with m3_model:

        #Jefferys prior from 0.0001 (10e-4) to 1000 e3
        logepsilon = pm.Uniform('logepsilon', -1, 2, testval = np.log10(2.5))
        logsigma = pm.Uniform('logsigma',-1, 2, testval = np.log10(2))

        #Jeffreys prior on H
        log_h = pm.Uniform('log_h', -2, 3, testval = np.log10(50))

        #convert to reg parameters:
        epsilon = pm.Deterministic('epsilon', 10**(logepsilon))
        sigma = pm.Deterministic('sigma', 10**(logsigma))

        hamaker = pm.Deterministic('hamaker', 10**(log_h))

        #uniform on radius
        radius = pm.Uniform('radius', 10, 50, testval = 25)

        #truncated normal on theta
        theta = pm.TruncatedNormal('theta', mu=35, sigma=15, lower=0, upper=90, testval=40)

        theta_rad = np.deg2rad(theta)


        #model
        force = 4*epsilon*(12*sigma**12/z**13) - hamaker/6*( radius/z**2 
                                                        + radius*(1-np.sin(theta_rad))/(z*(z+ radius*(1-np.sin(theta_rad)) )) 
                                                        + (np.tan(theta_rad))**2/(z+ radius*(1-np.sin(theta_rad)) ) 
                                                           )

        # Likelihood of observations (i.e. noise around model)
        measurements = pm.Normal('force', mu=force, sigma=noise, observed=f_testdata_m3)
    return m3_model


def vdwm3_newrep(repulsive_factor, sigma, radius, z, hamaker_fixed, f_testdata_m3_rep):
    
    '''
    generates noisy force data which includes the new repulsive term from the Lennard Jones force (~z^-3) 
    and a physically motivated vdW force.
    
    Inputs:
    -------
    repulsive_factor: float. In nV. Modified repulsive term for the depth of the well in the L-J theory. 
    sigma: float. In nm. The distance to 0 potential in the L-J theory.
    hamaker_fixed: float. In nV. Hamaker's constant for the specific tip and sample materials.
    radius: float. In nm. radius of the sphere of the tip.
    z: ndarray. In nm. the range over which the function will generate the data
    f_testdata_m3: ndarray. Simulated noisy Lennard Jones + vdW force data with modified repulsive term
    Returns:
    --------
    m3_rep_model: new Lennard Jones + vdW M3 model 
    '''

    m3_rep_model = pm.Model()

    with m3_rep_model:

        #Jefferys prior from 0.0001 (10e-4) to 1000 e3
        #logepsilon = pm.Uniform('logepsilon', -1, 1, testval = np.log10(2.5))
        #logsigma = pm.Uniform('logsigma',-1, 1, testval = np.log10(6.4))

        #Jeffreys prior on H
        #log_h = pm.Uniform('log_h', -2, 3, testval = np.log10(50))

        #convert to reg parameters:
        #epsilon = pm.Deterministic('epsilon', 10**(logepsilon))
        #sigma = pm.Deterministic('sigma', 10**(logsigma))

        #uniform on epsilon and sigma
        #epsilon = pm.Uniform('epsilon', 0.1,10)
        #sigma = pm.Uniform('sigma', 0.1 ,10)
        repulsive_factor = pm.Uniform('factor', 0.1, 100, testval = 50)

        #hamaker = pm.Deterministic('hamaker', 10**(log_h))

        #uniform on Hamaker
        #hamaker = pm.Normal('hamaker', mu=hamaker_fixed, sigma=hamaker_fixed/10, testval = hamaker_fixed)
        hamaker = hamaker_fixed

        #uniform on radius
        radius = pm.Uniform('radius', 10, 50, testval = 20)

        #truncated normal on theta
        theta = pm.TruncatedNormal('theta', mu=35, sigma=10, lower=0, upper=90, testval=40)

        theta_rad = np.deg2rad(theta)


        #model
        force = repulsive_factor/z**3 - hamaker/6*( radius/z**2 
                                                    + radius*(1-np.sin(theta_rad))/(z*(z+ radius*(1-np.sin(theta_rad)) )) 
                                                    + (np.tan(theta_rad))**2/(z+ radius*(1-np.sin(theta_rad)) ) 
                                                    )

        # Likelihood of observations (i.e. noise around model)
        measurements = pm.Normal('force', mu=force, sigma=noise, observed=f_testdata_m3_rep)
    return m3_rep_model