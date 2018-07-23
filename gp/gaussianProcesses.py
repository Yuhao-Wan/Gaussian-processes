
# Implementation of Gaussian Process for utility inference using Pairwise Comparison 
# Details see:
#   1. "A Tutorial on Bayesian Optimization of Expensive Cost Functions"
#   2. "Preference learning with Gaussian processes" (ICML 2005)

import numpy as np
from numpy.random import RandomState
from scipy.stats import norm
import copy


class GP_pairwise:
    """Gaussian process with a discrete-choice probit model (latent utility function).
    
    Attributes:
        sigma: Hyperparameter for std of the normal distributed noise for the utility function.
        theta: Hyperparameter for kernel width.
        seed: Seed for random state.
        datapoints: Matrix for the observed data.
        comparisons: Matrix for comparisons of the observed data.
        f_map: Approximate utility values of the datapoints.
        K: Covariance matrix of datapoints.
        K_inv: Inverse of the covariance matrix of datapoints.
        C: The matrix C for observed data.
        C_inv: The inverse of matrix C for observed data.
    """
    
    def __init__(self, sigma=0.01, theta=50, seed=None):
        """Inits GP class with all attributes."""
        self.sigma = sigma
        self.theta = theta
        self.random_state = RandomState(seed)
        self.datapoints = None
        self.comparisons = None
        self.f_map = None
        self.K = None
        self.K_inv = None
        self.C = None
        self.C_inv = None

    def update(self, dataset):
        """Update the Gaussian process using the given data.
        
        Args:
            dataset: Dataset consists of datapoints and comparison matrix data 
        """
        self.datapoints = dataset.datapoints
        self.comparisons = dataset.comparisons

        # compute the covariance matrix and its inverse
        self.K = self._get_K(self.datapoints)
        self.K_inv = self._get_inv(self.K)

        # compute the MAP estimate of f
        self.f_map = self._get_f_map()

        # compute C matrix given f_MAP and its inverse (psudo-inverse)
        self.C = self._get_C()
        self.C_inv = self._get_inv(self.C)

        return True 
    
    def sample(self, sample_points):
        """Get a sample from the current GP at the given points.
        
        Args:
            sample_points: The points at which we want to take the sample.
            
        Returns: 
            The values of the GP sample at the input points.
        """
        # get the mean and the variance of the predictive (multivariate gaussian) distribution at the sample points
        mean, var = self.get_Gaussian_params(sample_points, pointwise=False)

        # sample from the multivariate gaussian with the given parameters
        f_sample = self.random_state.multivariate_normal(mean, var, 1)[0]

        return f_sample
    
    def predict(self, sample_point):
        """Predicts value function for a single datapoint"""
        mean, var = self.get_Gaussian_params(np.array([sample_point]), pointwise=False)
        f_sample = self.random_state.multivariate_normal(mean, var, 1)[0]
        return mean

    def get_Gaussian_params(self, x_new, pointwise):
        """Gets the Gaussian parameters.
        
        Args:
            x_new: the points for which we want the predictive parameters.
            pointwise: whether we want pointwise variance or the entire covariance matrix.
            
        Returns:
            The predictive parameters of the Gaussian distribution at the given datapoints.
        """
        # if we don't have any data yet, use prior GP to make predictions
        if self.datapoints is None or self.f_map is None:
            pred_mean, pred_var = self._evaluate_prior(x_new)

        # otherwise compute predictive mean and covariance
        else:
            k_T = self._get_K(x_new, self.datapoints, noise=False)
            k = self._get_K(self.datapoints, x_new, noise=False)
            k_plus = self._get_K(x_new, noise=False)
            pred_mean = self._prior_mean(k_plus) + np.dot(np.dot(k_T, self.K_inv),
                                                         (self.f_map - self._prior_mean(self.datapoints)))            
            pred_var = k_plus - np.dot(np.dot(k_T, self._get_inv(self.K + self.C_inv)), k)
        if pointwise:
            pred_var = pred_var.diagonal()

        return pred_mean, pred_var

    
    def _get_K(self, x1, x2=None, noise=True):
        """Computes covariance matrix for preference data using the kernel function.
        
        Args:
            x1: The datapoints for which to compute covariance matrix.
            x2: If None, covariance matrix will be square for the input x1,
                If not None, covariance will be between x1 (rows) and x2 (cols)
            noise: Whether to add noise to the diagonal of the covariance matrix.
            
        Returns:
            The covariance matrix K.            
        """
        if x2 is None:
            x2 = x1
        else: 
            noise = False
        K = self._k(np.repeat(x1, x2.shape[0], axis=0), 
                               np.tile(x2, (x1.shape[0], 1)))
        K = K.reshape((x1.shape[0], x2.shape[0]))
        if noise:
            K += self.sigma ** 2 * np.eye(K.shape[0])
        return K
    
    def _k(self, x1, x2):
        """Exponentiated quadratic kernel function"""
        k = 0.8**2 * np.exp(-(1. / (2. * (self.theta ** 2))) * np.linalg.norm(x1 - x2, axis=1) ** 2)
        return k
        
    def _get_f_map(self):
        """Computes maximum a posterior (MAP) evaluation of f given the data using Newton's method
        
        Returns: 
            MAP of the Gassian processes values at current datapoints
        """
        converged = False
        try_no = 0

        f_map = None

        # Newton's method to approximate f_MAP
        while not converged and try_no < 1:

            # randomly initialise f_map
            f_map = self.random_state.uniform(0., 1., self.datapoints.shape[0])

            for m in range(100):
                # compute Z
                f_sup = np.array([f_map[self.comparisons[i, 0]] for i in range(self.comparisons.shape[0])])
                f_inf = np.array([f_map[self.comparisons[i, 1]] for i in range(self.comparisons.shape[0])])
                Z = self._get_Z(f_sup, f_inf)
                Z_logpdf = norm.logpdf(Z)
                Z_logcdf = norm.logcdf(Z)

                # compute b
                b = self._get_b(Z_logpdf, Z_logcdf)

                # compute gradient g
                g = self._get_g(f_map, b)

                # compute hessian H
                C = self._get_C(Z)
                H = - self.K_inv + C
                H_inv = self._get_inv(H)

                # perform update
                update = np.dot(H_inv, g)
                f_map -= update

                # stop criterion
                if np.linalg.norm(update) < 0.0001:
                    converged = True
                    break
                                   
            if not converged:
                print("Did not converge.")
                try_no += 1

        return f_map

    def _get_Z(self, f_sup, f_inf):
        """Gets the random variable Z based on given sup and inf pair."""
        return (f_sup - f_inf) / (np.sqrt(2) * self.sigma)
    
    def _get_b(self, Z_logpdf, Z_logcdf):
        """Gets the N-dimensional vector b"""
        h_j = np.array([np.array(self.comparisons[:, 0] == j, dtype=int) - np.array(self.comparisons[:, 1] == j, dtype=int) 
                        for j in range(self.datapoints.shape[0])])
        
        b = np.sum(h_j * np.exp(Z_logpdf - Z_logcdf), axis=1) / (np.sqrt(2) * self.sigma)
        return b
    
    def _get_g(self, f_map, b):
        """Gets the gradient g"""
        return -np.dot(self.K_inv, (f_map - self._prior_mean(self.datapoints))) + b

    def _get_C(self, Z=None):
        """Gets the matrix C"""
        if Z is None:
            # compute z
            f_sup = np.array([self.f_map[self.comparisons[i, 0]] for i in range(self.comparisons.shape[0])])
            f_inf = np.array([self.f_map[self.comparisons[i, 1]] for i in range(self.comparisons.shape[0])])
            Z = (f_sup - f_inf) / (np.sqrt(2) * self.sigma)

        Z_logpdf = norm.logpdf(Z)
        Z_logcdf = norm.logcdf(Z)

        # init with zeros
        C = np.zeros((self.datapoints.shape[0], self.datapoints.shape[0]))

        # build up diagonal for pairs (x, x)
        diag_arr = np.array([self._get_C_entry(m, m, Z, Z_logpdf, Z_logcdf) for m in
                             range(self.datapoints.shape[0])])
        np.fill_diagonal(C, diag_arr)  # happens in-place

        # go through the existing list of comparisons and update C
        for k in range(self.comparisons.shape[0]):
            m = self.comparisons[k, 0]  # superior
            n = self.comparisons[k, 1]  # inferior
            C[m, n] = self._get_C_entry(m, n, Z, Z_logpdf, Z_logcdf)
            C[n, m] = self._get_C_entry(n, m, Z, Z_logpdf, Z_logcdf)

        # add jitter terms to make matrix C positive semidefinite for stable computation
        C += np.eye(self.datapoints.shape[0]) * 0.01

        return C

    def _get_C_entry(self, m, n, Z, Z_logpdf, Z_logcdf):
        """Gets a single entry for the Hessian matrix at indices (m,n)"""
        h_x_m = np.array(self.comparisons[:, 0] == m, dtype=int) - np.array(self.comparisons[:, 1] == m, dtype=int)
        h_x_n = np.array(self.comparisons[:, 0] == n, dtype=int) - np.array(self.comparisons[:, 1] == n, dtype=int)
        p = h_x_m * h_x_n * (np.exp(2.*Z_logpdf - 2.*Z_logcdf) + Z * np.exp(Z_logpdf - Z_logcdf))
        c = - np.sum(p) / (2 * self.sigma**2)
        return c 
    
    
    def _evaluate_prior(self, input_points):
        """Evaluates the prior distribution given some datapoints.
        
        Args:
            input_points: input datapoints at which to evaluate prior distribution.
            
        Returns:
            The predictive mean and covariance at the given inputs.
        """
        pred_mean = self._prior_mean(input_points)
        num_inputs = input_points.shape[0]
        pred_cov = self._k(np.repeat(input_points, num_inputs, axis=0),
                           np.tile(input_points, (num_inputs, 1))).reshape((num_inputs, num_inputs))
        return pred_mean, pred_cov
                                             
    def _get_inv(self, M):
        """Computes the inverse of the given matrix or the psudoinverse."""
        try:
            M_inv = np.linalg.inv(M)
        except:
            M_inv = np.linalg.pinv(M)
        return M_inv
    
    def _prior_mean(self, x):
        """Returns the prior mean of zeros"""
        m = np.zeros(x.shape[0])
        return m



