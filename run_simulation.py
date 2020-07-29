import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style='darkgrid')
from tqdm import trange

from distributions import *
from metrics import *
from ranking_policies import *
from selection_policies import *

"""
NOTES:
    - treat Group B as the protected class
    - 
"""


def update_mean(ranking):
    ranking['delta'] = [0.5 if (row.selected and row.succeeded) 
                        else (-0.25 if row.selected else 0) 
                        for row in ranking.itertuples()]
    ranking = ranking.groupby('group').mean()['delta']
    return ranking['A'], ranking['B']


def main():
    # initialize constants =================================================================

    PROB_A = 0.6
    PROB_B = 1 - PROB_A

    MEAN_A = 1
    MEAN_B = 0
    VAR_A = 1
    VAR_B = 1
    
    QUERY_LEN = 20
    NUM_QUERIES = 10

    METRIC = 'avg-position'
    DIST = 'logit-normal'
    k = 6

    # NUM_ITERS = [10, 25, 100]
    # RANKING_POLICIES = ['top-k', 'max-util', 'stochastic']
    # SELECTION_POLICIES = ['top-k', 'stochastic']
    NUM_ITER = 10
    RANK_POLICY = 'top-k'
    SELECT_POLICY = 'stochastic'

    # run simulation ======================================================================

    # for RANK_POLICY in RANKING_POLICIES:
    # for SELECT_POLICY in SELECTION_POLICIES:
    # for NUM_ITER in NUM_ITERS:

    metric_a, mean_a = np.empty(NUM_ITER), np.empty(NUM_ITER)
    metric_b, mean_b = np.empty(NUM_ITER), np.empty(NUM_ITER)

    # ITERATIONS: length of time horizon
    for i in trange(NUM_ITER, desc='iterations'):
        a_metrics, deltas_a = np.empty(NUM_QUERIES), np.empty(NUM_QUERIES)
        b_metrics, deltas_b = np.empty(NUM_QUERIES), np.empty(NUM_QUERIES)

        # QUERIES: number of simulations per time step
        for j in trange(NUM_QUERIES, desc='queries'):
            # sample subjects from underlying distribution
            arr_a = sample_dist(MEAN_A, VAR_A, QUERY_LEN, PROB_A, DIST)
            arr_b = sample_dist(MEAN_B, VAR_B, QUERY_LEN, PROB_B, DIST)

            # rank subjects according to chosen policy, select individuals
            ranking = rank_policy(arr_a, arr_b, RANK_POLICY, k=k, p=PROB_A)
            result = select_policy(ranking, k, SELECT_POLICY)

            # compute chosen fairness metric 
            a_metrics[j], b_metrics[j] = compute_metric(ranking, METRIC).loc['A'], \
                                         compute_metric(ranking, METRIC).loc['B']

            # compute change in population distributions 
            deltas_a[j], deltas_b[j] = update_mean(result)

        # take the mean of the metrics over the queries at each step
        metric_a[i], metric_b[i] = np.mean(a_metrics), np.mean(b_metrics)

        # update population distributions for next iteration, keeping the sum the same
        mean_a[i], mean_b[i] = MEAN_A, MEAN_B
        MEAN_A += np.mean(deltas_a)
        MEAN_B += np.mean(deltas_b)

    # plot change in metric over time
    plt.plot(np.arange(NUM_ITER), metric_a, color='C2', label=f'Group A {METRIC}')
    plt.plot(np.arange(NUM_ITER), metric_b, color='C0', label=f'Group B {METRIC}')
    plt.savefig(f'sim-figures-{SELECT_POLICY}/{METRIC}_{NUM_ITER}.png', dpi=72)

    # plot the change in relevance over time
    plt.cla()
    plt.plot(np.arange(NUM_ITER), 1 / (1 + np.exp(-mean_a)), color='C2', label=f'Group A mean')
    plt.plot(np.arange(NUM_ITER), 1 / (1 + np.exp(-mean_b)), color='C0', label=f'Group B mean')
    plt.savefig(f'sim-figures-{SELECT_POLICY}/relevance_{NUM_ITER}.png', dpi=72)


if __name__ == '__main__':
    main()
