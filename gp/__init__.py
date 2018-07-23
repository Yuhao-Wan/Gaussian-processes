
# Implementation of Gaussian Process for utility inference using Pairwise Comparison 
# Details see:
#   1. "A Tutorial on Bayesian Optimization of Expensive Cost Functions"
#   2. "Preference learning with Gaussian processes" (ICML 2005)

import numpy as np
from numpy.random import RandomState
from scipy.stats import norm
import copy
import dataset
import gp
import acquisitionFunction

ds = Dataset(2)
gp = GP_pairwise(theta=20, seed=1) # build a new GP process
af = AcquisitionFunction(input_domain, seed=10) # build a new acquisition function

