import numpy as np
import pymc3 as pm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

data_df = pd.read_excel('data_df.xlsx')

def get_col_vals(df, col1, col2):
    """
    Prepare data for t-test 
    """
    y1 = np.array(df[col1])
    y2 = np.array(df[col2])
    return y1, y2

def prep_data(df, col1, col2):
    """
    Prepare data for pymc3 and return mean mu and sigma
    """
    y1 = np.array(df[col1])
    y2 = np.array(df[col2])

    y = pd.DataFrame(dict(value=np.r_[y1, y2], 
                          group=np.r_[[col1]*len(y1), 
                            [col2]*len(y2)]))
    mu = y.value.mean()
    sigma = y.value.std() * 2
    
    return y, mu, sigma

#Set two groups for T-Test
y1, y2 = get_col_vals(data_df, 'group1', 'group2')

#Print T-Test results to the console
print(ttest_ind(y1,y2, equal_var=False))

#Set the paramaters for the Bayesian Analysis
y, mu, sigma = prep_data(data_df, 'group1', 'group2')
μ_m = y.value.mean()
μ_s = y.value.std() * 2

with pm.Model() as model:
    """
    The priors for each group.
    """
    group1_mean = pm.Normal('group1_mean', μ_m, sd=μ_s)
    group2_mean = pm.Normal('group2_mean', μ_m, sd=μ_s)


#Set Uniform distribution for the standard deviation guesses.
σ_low = 1
σ_high = 20

with model:
    group1_std = pm.Uniform('group1_std', lower=σ_low, upper=σ_high)
    group2_std = pm.Uniform('group2_std', lower=σ_low, upper=σ_high)

with model:
    """
    Prior for ν is an exponential (lambda=29) shifted +1.
    """
    ν = pm.Exponential('ν_min_one', 1/29.) + 1

with model:
    """
    Transforming standard deviations to precisions (1/variance) to calculate the likelihoods
    """
    λ1 = group1_std**-2
    λ2 = group2_std**-2

    group1 = pm.StudentT('group1', nu=ν, mu=group1_mean, lam=λ1, observed=y1)
    group2 = pm.StudentT('group2', nu=ν, mu=group2_mean, lam=λ2, observed=y2)

with model:
    """
    The effect size is the difference in means/pooled estimates of the standard deviation.
    The Deterministic class represents variables whose values are completely determined
    by the values of their parents.
    """
    diff_of_means = pm.Deterministic('difference of means',  group2_mean - group1_mean)
    diff_of_stds = pm.Deterministic('difference of stds',  group2_std - group1_std)
    effect_size = pm.Deterministic('effect size',
                                   diff_of_means / np.sqrt((group2_std**2 + group1_std**2) / 2))

with model:
    '''
    generate a model that takes sample from 2000 values stepped via the No U-Turn Sampling algorithm
    NUTS
    '''
    step = pm.NUTS()
    trace = pm.sample(2000, tune=500, init=None, step=step, cores=1)

#Plotting the posterior using the sample trace with a 50% burn in to remove transient effects of MCMC.
pm.plot_posterior(trace[1000:],
                  varnames=['difference of means', 'difference of stds', 'effect size'],
                  ref_val=0,
                  color='#87ceeb');

plt.show()