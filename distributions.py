import numpy as np
from scipy.stats import beta
from scipy.stats import norm


# DISTRIBUTIONS =========================================================

def sample_dist(dist, mean, var, ql, prob):
    """
    Returns length-(QUERY_LEN * prob) list of relevances (in decreasing order) 
    as sampled from a chosen distribution with specified mean and variance
    """
    if dist == 'beta':
        return sample_beta(mean, var, ql, prob)
    elif dist == 'normal':
        return sample_normal(mean, var, ql, prob)
    else:
        # TODO
        pass


def sample_beta(mean, var, ql, prob):
    a = (((1 - mean) / var) - (1 / mean)) * (mean ** 2)
    b = a * ((1 / mean) - 1)
    arr = np.array(beta.rvs(a, b, size=int(ql * prob)))
    return np.sort(arr)[::-1]


def sample_normal(mean, var, ql, prob):
    arr = np.array(norm.rvs(loc=mean, scale=var, size=int(ql * prob)))
    return np.sort(arr)[::-1]
