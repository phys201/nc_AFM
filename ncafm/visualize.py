import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ncafm.simulatedata as sim

def plot_traces(x, data, fit_traces, map_dict, hamaker, fit_type = 'sph', y_min = -0.7, y_max = 5, num_traces = 8):
    
    '''
    Plots the data, MAP fit and 8 traces (optional) from the MC sampling.
    
    Inputs:
    -------
    
    x: ndarray the x data of the fitting (z in nm)
    data: the y data of the fitting (force in nN)
    fit_traces: pandas DataFrame or pymc3 traces object. The results of MC sampling.
    map_dict: dictionary (or Dataframe) with the MAP values from the MC sampling
    hamaker: Hamaker's constant. in aJ.
    fit_type: string. denoting the model used. Default is 'sph' for spherical model (M1). 
        Other options: 'cone' for conical model (M2), 'cone+sph' for the conical plus spherical model (M3)
        
    y_min, y_max: float, the min and max of the plot to display (for a closer look). 
    num_traces: number of traces to disply on the plot. Default = 8 . Input should be an integer.
    '''
    
    plt.plot(x, data, label = 'data')

    if isinstance(fit_traces, pd.DataFrame) == False:
        fit_dataframe = fit_traces.posterior.to_dataframe()
    else:
        fit_dataframe = fit_traces

    if fit_type == 'sph':
        plt.plot(x, sim.sph(x, map_dict['rep factor'], map_dict['alpha'], hamaker, map_dict['radius'], 0, z_0 = map_dict['z offset']), c='k', label = 'MAP')
    elif fit_type == 'cone':
        plt.plot(x, sim.cone(x, map_dict['rep factor'], map_dict['alpha'], hamaker, map_dict['theta'], 0, z_0 = map_dict['z offset']), c='k', label = 'MAP')
    elif fit_type == 'cone+sph':
        plt.plot(x, sim.cone_sph(x, map_dict['rep factor'], map_dict['alpha'], hamaker, map_dict['radius'], map_dict['theta'], 0, z_0 = map_dict['z offset']), c='k', label = 'MAP')
    else: 
        raise ValueError('fit_type does not correspond to a defined model type. Options: sph, cone, cone+sph')
                         
    #choose 8 lines to show at random
    indices = np.random.randint(len(fit_dataframe), size=int(num_traces))
    for ind in indices:
        rep_factor = fit_dataframe['rep factor'].iloc[ind]
        alpha = fit_dataframe['alpha'].iloc[ind]
        z_0 = fit_dataframe['z offset'].iloc[ind]
        
        if fit_type == 'sph':
            radius = fit_dataframe['radius'].iloc[ind]
            plt.scatter(x, sim.sph(x, rep_factor, alpha, hamaker, radius, 0, z_0 = z_0), c="C1", s=4, alpha=0.4, label = 'trace')

        elif fit_type == 'cone':
            theta = fit_dataframe['theta'].iloc[ind]
            plt.scatter(x, sim.cone(x, rep_factor, alpha, hamaker, theta, 0, z_0 = z_0), c="C1", s=4, alpha=0.4, label = 'trace')

        elif fit_type == 'cone+sph':
            radius = fit_dataframe['radius'].iloc[ind]
            theta = fit_dataframe['theta'].iloc[ind]
            plt.scatter(x, sim.cone_sph(x, rep_factor, alpha, hamaker, radius, theta, 0, z_0 = z_0), c="C1", s=4, alpha=0.4, label = 'trace')
        
        else: 
            raise ValueError('fit_type does not correspond to a defined model type. Options: sph, cone, cone+sph')
            
    plt.ylim([y_min, y_max])
    plt.legend(loc = 1)
    plt.xlabel('tip height (nm)', fontsize = 14)
    plt.ylabel('force (nN)', fontsize = 14);
    
def print_ci(traces, variable, ci_range = 67):
    '''
    Function that prints the 67% credibility interval for the parameter for a given set of traces (or
    dataframe built from traces).
    
    Inputs:
    -------
    traces: pandas DataFrame or pymc3 traces object. The results of MC sampling.
    variable: string. the variable or parameter you wish to caluclation the credibility interval for.
    ci_range: float (or integer). between 1 and 99 representing the percentage of credibility interval you wish to caluclate.
        default = 67% (2 sigma)
    
    '''
    
    if isinstance(traces, pd.DataFrame) == False:
        traces_dataframe = traces.posterior.to_dataframe()
    else:
        traces_dataframe = traces

    if isinstance(variable, str) == False:
        raise ValueError("Input the variable name as a string")
        
    if ci_range < 1 or ci_range > 100:
        raise ValueError("ci_range should be between 1 and 99 representing the percentage of credibility interval you wish to caluclate.")
        
    lower_bound = np.round(0.5 - ci_range/2/100,3)
    upper_bound = np.round(0.5 + ci_range/2/100,3)
    
    
    
    MAP = traces_dataframe.quantile([lower_bound,0.50,upper_bound], axis=0)
    print("The {:.1f} % credibility interval for".format((upper_bound-lower_bound)*100), 
          variable, "= {:.2f} + {:.2f} - {:.2f}".format(MAP[variable][0.50],
                                            MAP[variable][upper_bound]-MAP[variable][0.50],
                                            MAP[variable][0.50]-MAP[variable][lower_bound]))