import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style='darkgrid')

from ranking_policies.py import *
from selection_policies.py import *
from metrics import *
from distributions import *

"""
TODO:
    - need to incorporate NDCG in fair rankings
        * NDCG vs. exposure?
    - implement more fair ranking policies for comparison to max-util
    - only look at top-k for success/failure rates?
"""


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

    METRIC = 'avg_position'
    DIST = 'logit_normal'
    k = 10

    # NUM_ITERS = [10, 25, 100]
    NUM_ITER = 100 
    # RANKING_POLICIES = ['top-k', 'max-util', 'stochastic']
    RANKING_POLICY = 'top-k'
    # SELECTION_POLICIES = ['top-k', 'stochastic']
    SELECTION_POLICY = 'stochastic'

    # for RANK_POLICY in RANKING_POLICIES:
    # for SELECT_POLICY in SELECTION_POLICIES:
    # for NUM_ITER in NUM_ITERS:
    metric_a, mean_a = np.empty(NUM_ITER), np.empty(NUM_ITER)
    metric_b, mean_b = np.empty(NUM_ITER), np.empty(NUM_ITER)

    for i in range(NUM_ITER):
        a_metrics = np.empty(NUM_QUERIES)
        b_metrics = np.empty(NUM_QUERIES)

        for j in range(NUM_QUERIES):
            # sample QUERY_LEN subjects from underlying distribution
            arr_a = sample_dist(MEAN_A, VAR_A, QUERY_LEN, PROB_A, DIST)
            arr_b = sample_dist(MEAN_B, VAR_B, QUERY_LEN, PROB_B, DIST)

            # rank subjects according to chosen policy, select individuals
            ranking = rank_policy(arr_a, arr_b, RANK_POLICY)
            result = select_policy(ranking, k, SELECT_POLICY)

            a_metrics[j], b_metrics[j] = compute_metric(ranking, METRIC).loc['A'], \
                                         compute_metric(ranking, METRIC).loc['B']

# ////////////////////////////////////////// TODO LATER //////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////

        # take the mean of the metrics over the queries at each step
        metric_a[i], metric_b[i] = np.mean(a_metrics), np.mean(b_metrics)

        # update population distributions for next iteration
        # keeping the sum the same
        mean_a[i], mean_b[i] = MEAN_A, MEAN_B

        # updating the means in a funny way, need to figure out top k way to do it
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
    plt.savefig(f'sim-figures-{POLICY}/{METRIC}_{NUM_ITER}.png', dpi=72)

    # plot the change in means over time
    plt.cla()
    plt.plot(np.arange(NUM_ITER), mean_a, color='C2', label=f'Group A mean')
    plt.plot(np.arange(NUM_ITER), mean_b, color='C0', label=f'Group B mean')
    plt.savefig(f'sim-figures-{POLICY}/means_{NUM_ITER}.png', dpi=72)

# ////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////


if __name__ == '__main__':
    main()
