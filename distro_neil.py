import numpy as np
from scipy.stats import beta
from scipy.stats import norm
from scipy.special import softmax
import matplotlib.pyplot as plt
import random
import seaborn as sns
import math

sns.set(style='darkgrid')

"""
TODO:
    - should stop changing distributions once means converge...
    - need to incorporate NDCG in fair rankings
        * NDCG vs. exposure?
    - implement more fair ranking policies for comparison to max-util
    - only look at top-k for success/failure rates?
"""


# RANKING POLICIES =====================================================


def rank_top_k(arr_a, arr_b, k, prob_a):
    """
    Returns optimal ranking as measured by NDCG and 
    subject to top-k demographic parity constraint; 
    returns rankings for subgroups separately.
    """
    # round k to nearest integers
    k_a = int(k * prob_a)
    k_b = int(k * (1 - prob_a))
    k = k_a + k_b

    a, b = 0, 0
    rank_a, rank_b = arr_a, arr_b

    # rank top k subjects subject to demographic parity constraint 
    while a < k_a and b < k_b:
        if arr_a[a] > arr_b[b]:
            rank_a[a] = a + b + 1
            a += 1
        else:
            rank_b[b] = a + b + 1
            b += 1

    # get the leftovers (top k)
    while a < k_a:
        rank_a[a] = a + b + 1
        a += 1
    while b < k_b:
        rank_b[b] = a + b + 1
        b += 1

    # rank remaining subjects by max-util strategy
    remain_a, remain_b = rank_max_util(arr_a[a:], arr_b[b:])
    # for s_a in remain_a:
    rank_a = np.append(rank_a, [s_a + k for s_a in remain_a])
    # for s_b in remain_b:
    rank_b = np.append(rank_b, [s_b + k for s_b in remain_b])

    return rank_a, rank_b


def rank_max_util(arr_a, arr_b):
    """
    Returns optimal ranking as measured by NDCG; 
    equivalent to ranking in order of relevance.
    """
    a, b = 0, 0
    rank_a, rank_b = arr_a, arr_b

    while a < len(arr_a) and b < len(arr_b):
        if arr_a[a] > arr_b[b]:
            rank_a[a] = a + b + 1
            a = a + 1
        else:
            rank_b[b] = a + b + 1
            b = b + 1

    # get the leftovers
    while a < len(arr_a):
        rank_a[a] = a + b + 1
        a = a + 1
    while b < len(arr_b):
        rank_b[b] = a + b + 1
        b = b + 1

    return rank_a, rank_b


def rank_stoch(arr_a, arr_b):
    rank_a, rank_b = np.empty(len(arr_a)), np.empty(len(arr_b))
    a, b = 0, 0

    s = softmax(np.append(arr_a, arr_b))

    while len(arr_a) > 0 or len(arr_b) > 0:
        s = softmax(np.append(arr_a, arr_b))
        rand = random.uniform(0, 1)
        summer = 0
        for i in range(len(s)):
            summer += s[i]
            if rand < summer:
                if i in range(0, len(arr_a)):
                    rank_a[a] = a + b + 1
                    a += 1
                    arr_a = np.delete(arr_a, [i])
                else:
                    rank_b[b] = a + b + 1
                    b += 1
                    arr_b = np.delete(arr_b, [i - len(arr_a)])
                break

    return rank_a, rank_b


# ///////////////////////////////////////////////////////////////////////

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


# ///////////////////////////////////////////////////////////////////////

# METRICS ===============================================================


def compute_metric(rank_a, rank_b, metric):
    """
    Computes chosen metric to track change over time.
    """
    if metric == 'avg_position':
        return avg_position(rank_a, rank_b)
    elif metric == 'avg_exposure':
        return avg_exposure(rank_a, rank_b)
    else:
        # TODO 
        pass


def avg_position(rank_a, rank_b):
    return np.mean(rank_a), np.mean(rank_b)


def avg_exposure(rank_a, rank_b):
    return 1 / math.log2(1 + avg_position(rank_a, rank_b)[0]), 1 / math.log2(1 + avg_position(rank_a, rank_b)[1])


# ///////////////////////////////////////////////////////////////////////


def update_mean(mean):
    sig = 1 / (1 + np.exp(-mean))
    # should play more with changes in the mean
    delta = sig * 0.5 - (1 - sig) * 0.5
    return delta

def main():
    PROB_A = 0.6
    PROB_B = 1 - PROB_A
    MEAN_A = 1.5
    MEAN_B = -1.5
    VAR_A = 1
    VAR_B = 1
    QUERY_LEN = 10
    NUM_QUERIES = 50
    NUM_ITER = 100
    # NUM_ITERS = [10, 25, 100]
    METRIC = 'avg_position'
    DIST = 'normal'

    # for NUM_ITER in NUM_ITERS:
    metric_a, mean_a = np.empty(NUM_ITER), np.empty(NUM_ITER)
    metric_b, mean_b = np.empty(NUM_ITER), np.empty(NUM_ITER)

    for i in range(NUM_ITER):
        a_metrics = np.empty(NUM_QUERIES)
        b_metrics = np.empty(NUM_QUERIES)

        for j in range(NUM_QUERIES):
            # sample QUERY_LEN subjects from underlying distribution
            arr_a = sample_dist(DIST, MEAN_A, VAR_A, QUERY_LEN, PROB_A)
            arr_b = sample_dist(DIST, MEAN_B, VAR_B, QUERY_LEN, PROB_B)

            # rank subjects according to chosen policy, compute metric
            # rank_a, rank_b = rank_top_k(arr_a, arr_b, 5, PROB_A)
            rank_a, rank_b = rank_max_util(arr_a, arr_b)
            # rank_a, rank_b = rank_stoch(arr_a, arr_b)
            a_metrics[j], b_metrics[j] = compute_metric(rank_a, rank_b, METRIC)

        # take the mean of the metrics over the queries at each step
        metric_a[i], metric_b[i] = np.mean(a_metrics), np.mean(b_metrics)

        # update population distributions for next iteration
        # keeping the sum the same
        mean_a[i], mean_b[i] = MEAN_A, MEAN_B
        if abs(MEAN_B - MEAN_A) > 0.01:
            if MEAN_B < MEAN_A:
                MEAN_B += update_mean(np.mean(arr_b)) / 2
                MEAN_A -= update_mean(np.mean(arr_b)) / 2
            if MEAN_A < MEAN_B:
                MEAN_A += update_mean(np.mean(arr_a)) / 2
                MEAN_B -= update_mean(np.mean(arr_a)) / 2

    # plot change in metric over time
    plt.plot(np.arange(NUM_ITER), metric_a, color='C2', label=f'Group A {METRIC}')
    plt.plot(np.arange(NUM_ITER), metric_b, color='C0', label=f'Group B {METRIC}')
    plt.savefig(f'sim-figures/{METRIC}_{NUM_ITER}.png', dpi=72)

    # plot the change in means over time
    plt.cla()
    plt.plot(np.arange(NUM_ITER), mean_a, color='C2', label=f'Group A mean')
    plt.plot(np.arange(NUM_ITER), mean_b, color='C0', label=f'Group B mean')
    plt.savefig(f'sim-figures/means_{NUM_ITER}.png', dpi=72)


if __name__ == '__main__':
    main()
