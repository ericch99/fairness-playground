import numpy as np
from scipy.stats import beta
from scipy.stats import norm


def sample_dist(mean, var, ql, prob, dist):
    """
    Returns length-(ql * prob) list of relevances (in decreasing order) 
    as sampled from a chosen distribution with specified mean and variance
    """
    if dist == 'beta':
        return sample_beta(mean, var, ql, prob)
    elif dist == 'logit-normal':
        return sample_logit_normal(mean, var, ql, prob)
    else:
        raise NotImplementedError


# DISTRIBUTIONS =========================================================

def sample_beta(mean, var, ql, prob):
    a = (((1 - mean) / var) - (1 / mean)) * (mean ** 2)
    b = a * ((1 / mean) - 1)
    arr = np.array(beta.rvs(a, b, size=int(ql * prob)))
    return np.sort(arr)[::-1].tolist()


def sample_logit_normal(mean, var, ql, prob):
    x = np.array(norm.rvs(loc=mean, scale=var, size=int(ql * prob)))
    arr = 1 / (1 + np.exp(-x))
    return np.sort(arr)[::-1].tolist()
