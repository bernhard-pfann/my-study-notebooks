"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """

    from scipy.stats import multivariate_normal
    
    k, d = mixture.mu.shape
    n = len(X)
    posteriors_all = np.zeros((n, k))

    # Iterate over observations in X
    for i in range(n):
        normalization = 0
        posteriors = np.zeros(k)

        # Iterate over k components
        for j in range(k):
            cov_matrix = np.identity(d) * mixture.var[j]
            likelihood = multivariate_normal.pdf(X[i], mean=mixture.mu[j], cov=cov_matrix)
            posterior = mixture.p[j] * likelihood
            posteriors[j] = posterior        
    
        posteriors_all[i] = posteriors

    normalization = posteriors_all.sum(axis=1)
    posteriors_all = posteriors_all / normalization.reshape(-1,1)

    log_likelihood = np.sum(np.log(normalization))

    return posteriors_all, log_likelihood


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    dim = X.shape[1]
    k = post.shape[1]

    hat_mu = np.zeros((k, dim))
    hat_var = np.zeros(k)

    hat_n = post.sum(axis=0)
    hat_p = hat_n / hat_n.sum()

    for j in range(k):
        # Estimate mu
        hat_mu_j = X*post[:,j].reshape(-1,1)
        hat_mu_j = hat_mu_j.sum(axis=0)/hat_n[j]
        hat_mu[j] = hat_mu_j
 
        # Estimate variance
        denom_j = hat_n[j] * dim
        dist_j = np.sum(np.square(X - hat_mu[j]), axis=1)
        hat_var_j = np.sum(post[:,j] * dist_j) / denom_j
        hat_var[j] = hat_var_j

    model = GaussianMixture(mu=hat_mu, var=hat_var, p=hat_p)
    return model


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    condition = True
    
    while condition:

        old_log_lh = new_log_lh

        # Update steps
        post, new_log_lh = estep(X=X, mixture=mixture)
        mixture = mstep(X=X, post=post)
        
        # Evaluate while condition  
        condition = (new_log_lh - old_log_lh) > 1e-6 * np.abs(new_log_lh)

    return mixture, new_log_lh




