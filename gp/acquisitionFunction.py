
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


class AcquisitionFunction:
    def __init__(self, input_domain, seed):
        """ An acquirer for a discrete set of points.
        
        Attributes:
            input_domain: the datapoints on which the discrete acquirer is defined.
            random_state: based on the random seed.
        """
        self.input_domain = copy.deepcopy(input_domain)
        self.random_state = np.random.RandomState(seed)
        self.history = np.empty((0, self.input_domain.shape[1]))


    def get_next_point(self, dataset, gaussian_process):
        """Selects a single next datapoint to query.
        
        Args: 
            gaussian_process
            dataset
        """
        next_point_1, next_point_2 = self.get_next_point_VAR(dataset, gaussian_process)
        self.history = np.vstack((self.history, next_point_1, next_point_2))
        return next_point_1, next_point_2
    
    def get_next_point_VAR(self, dataset, gaussian_process):
        """Selects the next two datapoint to query using maximum uncertainty
        
        Args:
            dataset
            gaussian_process
            
        Returns:
            The optimal next pair based on maximum variance.
        """
        var = np.zeros(self.input_domain.shape[0])
        batch_size = 16
        for curr_idx in range(0, self.input_domain.shape[0]+batch_size, batch_size):
            var[curr_idx:curr_idx+batch_size] = self._get_variance(self.input_domain[curr_idx:curr_idx+batch_size], gaussian_process)
        # find the two points with the highest variance, and which can be queried next
        next_point1 = self.input_domain[np.argsort(var)[-1]]
        next_point2 = self.input_domain[np.argsort(var)[-2]]
        next_point_idx1, next_point_idx2 = 1, 2
        while self._check_duplicate(dataset, next_point1, next_point2):
            next_point1 = self.input_domain[np.argsort(-var)[next_point_idx1]]
            next_point2 = self.input_domain[np.argsort(-var)[next_point_idx2]]
            next_point_idx1 += 1
            next_point_idx2 += 1
        return next_point1, next_point2

    def _get_variance(self, datapoints, gaussian_process):
        """Obtains the predictive pointwise variance."""
        pred_var = gaussian_process.get_Gaussian_params(datapoints, pointwise=True)[1]
        return pred_var    
    
    def _check_duplicate(self, dataset, point1, point2):
        """Checks if there are duplicate pairs being sampled."""
        if point1 not in dataset.datapoints or point2 not in dataset.datapoints:
            return False 
        idx1, idx2 = np.where(dataset.datapoints==point1), np.where(dataset.datapoints==point2)
        return True if [idx1, idx2] or [idx2, idx1] in dataset.comparisons else False
