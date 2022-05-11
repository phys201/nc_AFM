import pymc3 as pm
import numpy as np
import logging

#use this to suppress the output when running this function
logger = logging.getLogger('pymc3')
logger.setLevel(logging.ERROR)

def compare_models(model1, model2, prior_m1 = 0.5, prior_m2 = 0.5, num_samples = 8000, num_tunes = 3000):
    
    '''
    A function that computes the odds ratio for 2 models. Default settings assumes equal probability to the models.
    
    Inputs:
    -------
    model1, model2: pm.Model objects. Can be created through the `model` module.
        It is up to the user to edit the priors on the model if they are different from the default.
        However, the default values are physically motivated before the knowledge of the data. 
        
    prior_m1, prior_m2: float. The prior probability for model1 and model2 repsectively. 
        If the prior for each model is the same the value can stay as the default of 0.5. 
        If there is a different prior probability for one model over another input relative values here.
        Notice: the values do not have to sum to 1, but cannot be negative. 
        The odds ratio will take into account the prior by taking the ratio.
        
    num_samples, num_tunes: integer. The number of samples and tuning steps to perform respectively. 
        
    Returns:
    --------
    odds_ratio: float. The odds ratio of model1 to model2. 
    '''
    
    seed = np.random.randint(1,10**6)
    
    with model1:
        samples1 = pm.sample_smc(num_samples, tune_steps= num_tunes, random_seed=seed)
        
    with model2:
        samples2 = pm.sample_smc(num_samples, tune_steps= num_tunes, random_seed=seed)
        
    #take the mean since it calculates the two chains separately. 
    odds_ratio = np.mean(np.exp(np.log(prior_m1)+samples1.report.log_marginal_likelihood - 
                                np.log(prior_m2)-samples2.report.log_marginal_likelihood))
    
    if odds_ratio > 1:
        print('The first input model is {:2.2e} times more probable than the second input model.'.format(odds_ratio))
    
    elif odds_ratio < 1:
        print('The second input model is {:2.2e} times more probable than the first input model.'.format(1/odds_ratio))
    else:
        print('The odds ratio is exactly 1! Was the same model input for both?')
        
    return odds_ratio