"""
Evaluation and benchmarking framework for factor models (Simplified)
--------------------------------------------------------------------
This version of factor_eval.py only includes the SyntheticData class,
which generates synthetic datasets for latent factor models. All other
classes, functions, and code not directly related to synthetic data 
generation have been removed.

Original code by: Greg Ver Steeg (gregv@isi.edu), 2017.
Modified by: [Your Name], [Date]
"""

import sys
import numpy as np
from numpy.linalg import eigvalsh
from scipy.linalg import pinvh
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


class SyntheticData(object):
    """
    Generate synthetic data from a Non-overlapping Gaussian latent factor model.
    
    Parameters
    ----------
    n_samples : int, default=1000
        Number of samples in the training set.
    n_test : int, default=1000
        Number of samples in the test set.
    n_sources : int, default=100
        Number of latent factor sources (latent factors).
    k : int, default=10
        Number of observed variables per latent factor. Total observed variables = k * n_sources.
    snr : float, default=0.5
        Signal-to-noise ratio. Controls the relative amount of noise.
        Higher snr => stronger latent signal compared to noise.
    correlate_sources : bool or float, default=False
        If False, sources are independent. 
        If a float (e.g. eps), then sources are correlated based on eps.
    get_covariance : bool, default=False
        If True, also generate and store the true covariance matrix.
    random_scale : bool, default=False
        If True, randomly scale the observed variables.
    nuisance : int, default=0
        Number of nuisance variables to add that do not belong to any factor.
    seed : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    data : ndarray of shape (n_samples + n_test, n_variables)
        The full dataset including both train and test samples.
    train : ndarray of shape (n_samples, n_variables)
        The training portion of the dataset.
    test : ndarray of shape (n_test, n_variables)
        The test portion of the dataset.
    clusters : list
        Ground truth cluster assignments for each observed variable.
        Each factor corresponds to one cluster, so variables are grouped by factor.
        Nuisance variables are assigned cluster = -1.
    sources : ndarray of shape (n_samples + n_test, n_sources)
        The latent factor values (ground truth sources).
    sources_train : ndarray of shape (n_samples, n_sources)
        Latent factor values for training set.
    sources_test : ndarray of shape (n_test, n_sources)
        Latent factor values for test set.
    true_cov : ndarray of shape (n_variables, n_variables), optional
        The true covariance matrix of the data, if get_covariance=True.
    description : dict
        A dictionary of parameters describing the generated data.
    description_string : str
        A string representation of the parameters.
    """

    def __init__(self, n_samples=1000, n_test=1000, n_sources=100, k=10, snr=0.5,
                 correlate_sources=False, get_covariance=False,
                 random_scale=False, nuisance=0, seed=None):
        np.random.seed(seed)
        self.n_sources = n_sources
        self.n_variables = k * n_sources
        self.n_samples = n_samples

        # Define cluster assignments
        self.clusters = [i / k for i in range(n_sources * k)]
        self.clusters += [-1] * nuisance  # Nuisance variables have cluster = -1

        self.description = {
            "p": self.n_variables,
            "n": n_samples,
            "m": n_sources,
            "SNR": snr,
            "correlate_sources": correlate_sources,
            "nuisance": nuisance,
            "random_scale": random_scale
        }
        self.description_string = ','.join(["{}={}".format(key, v) for key, v in self.description.items()])

        # Generate data
        if get_covariance:
            self.data, self.sources, self.true_cov = self.gen_data_cap(n_sources * k, n_samples + n_test, snr,
                                                                       n_sources, correlate_sources, get_covariance,
                                                                       random_scale, nuisance)
        else:
            self.data, self.sources = self.gen_data_cap(n_sources * k, n_samples + n_test, snr,
                                                        n_sources, correlate_sources, get_covariance,
                                                        random_scale, nuisance)

        # Split into train/test
        self.train = self.data[:n_samples]
        self.test = self.data[n_samples:]
        self.sources_train = self.sources[:n_samples]
        self.sources_test = self.sources[n_samples:]

    def gen_data_cap(self, p=500, n_samples=500, snr=0.5, n_sources=16,
                     correlate_sources=False, get_covariance=False,
                     random_scale=False, nuisance=0):
        """
        Generate synthetic data from a latent factor model:
        Z_j ~ N(0,1) are latent factors. 
        For each factor, we have k observed variables:
        X_i = (Z_j + noise) / sqrt(1 + noise_variance), with noise_variance = 1/snr.
        
        If correlate_sources is a float (eps), sources are correlated according to eps.
        Otherwise, they are independent.

        If get_covariance=True, also compute the true covariance matrix.

        If random_scale=True, each variable is scaled by a random factor drawn from a lognormal distribution.
        """
        if correlate_sources:
            eps = correlate_sources
            sources = (eps * np.random.randn(n_samples, 1) + np.random.randn(n_samples, n_sources)) / np.sqrt(1 + eps**2)
        else:
            sources = np.random.randn(n_samples, n_sources)

        k = int(p / n_sources)
        assert p % n_sources == 0, 'For simplicity, we force k variables per source.'

        noises = [[1. / snr for _ in range(k)] for _ in range(n_sources)]
        observed = np.vstack([
            (source + np.sqrt(noises[j][i]) * np.random.randn(n_samples)) / np.sqrt(1 + noises[j][i])
            for j, source in enumerate(sources.T) for i in range(k)
        ]).T

        # Add nuisance variables (pure noise)
        observed = np.hstack([observed, np.random.standard_normal((n_samples, nuisance))])

        # Random scaling of variables if required
        if random_scale:
            scales = 2 ** np.random.standard_normal(p + nuisance)
        else:
            scales = np.ones(p + nuisance)

        observed *= scales

        # Compute true covariance if requested
        if get_covariance:
            cov = np.eye(p + nuisance)
            for i in range(p):
                for ip in range(p):
                    if i == ip:
                        cov[i, ip] = 1
                    elif i // k == ip // k:
                        # Within the same factor cluster
                        cov[i, ip] = (1 / np.sqrt(1 + noises[i // k][i % k])
                                      / np.sqrt(1 + noises[i // k][ip % k]))
                    elif correlate_sources:
                        eps = correlate_sources
                        cov[i, ip] = eps ** 2 / (1 + eps ** 2)
            cov *= scales * scales[:, np.newaxis]
            return observed, sources, cov
        else:
            return observed, sources


# Note: All other classes, imports, and functions not required for SyntheticData have been removed.