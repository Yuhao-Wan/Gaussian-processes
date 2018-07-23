
# Implementation of Gaussian Process for utility inference using Pairwise Comparison 
# Details see:
#   1. "A Tutorial on Bayesian Optimization of Expensive Cost Functions"
#   2. "Preference learning with Gaussian processes" (ICML 2005)

import numpy as np
from numpy.random import RandomState
from scipy.stats import norm
import copy


class Dataset:
    """Dataset class
    
    Attributes:
        datapoints: matrix for datapoints (dimension t x n)
        comparisons: matrix for comparisons (dimension m x 2)
    """
    
    def __init__(self, num_features):
        """Inits Dataset with the number of features"""
        self.datapoints = np.empty((0, num_features))
        self.comparisons = np.empty((0, 2), dtype=np.int)

    def add_single_comparison(self, sup, inf):
        """Adds a single comparison to the dataset.
        
        Args:
            sup: The datapoint index that is superior in the comparison.
            inf: The datapoint index that is inferior in the comparison.
        """
        # add superior and inferior to our datapoints and get the indices in dataset
        sup_idx = self._add_single_datapoint(sup)
        inf_idx = self._add_single_datapoint(inf)
        self.comparisons = np.vstack((self.comparisons, [sup_idx, inf_idx]))
        
        return True

    def _add_single_datapoint(self, new_datapoint):
        """Adds a single datapoint to the existing dataset and returns index.
        
        Args:
            new_datapoint: The new datapoint to be added.        
            
        Returns:
            x_new_idx: The index of the new datapoint in the existing dataset.
        """
        self.datapoints = np.vstack((self.datapoints, new_datapoint))
        new_datapoint_index = self.datapoints.shape[0] - 1
        return new_datapoint_index

    def get_index(self, datapoint):
        """Gets the index of a datapoint in the dataset.
        
        Args:
            datapoint: A single datapoint.
        
        Returns:            
            The index of datapoint in the dataset.
        """
        return np.argmax(np.sum(datapoint != self.datapoints, axis=1) == 0)
