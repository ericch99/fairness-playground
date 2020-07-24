import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style='darkgrid')
import math

# from ranking_policies.py import *
# from selection_policies.py import *
from metrics import *
from distributions import *
import ranking_policies
import selection_policies

"""
TODO:
    - need to incorporate NDCG in fair rankings
        * NDCG vs. exposure?
    - implement more fair ranking policies for comparison to max-util
    - only look at top-k for success/failure rates?
"""


def update_mean(ranking):
    ranking = ranking.groupby(['group', 'selected', 'succeeded']).count()['rank']
    ranking = ranking.unstack().unstack()

    if not math.isnan(ranking.iloc[0, 2]) and not math.isnan(ranking.iloc[0, 3]):
        delta_a = (ranking.iloc[0, 3] - ranking.iloc[0, 2]) / \
                  (ranking.iloc[0, 3] + ranking.iloc[0, 2])
    else:
        if not math.isnan(ranking.iloc[0, 3]):
            delta_a = 1
        elif not math.isnan(ranking.iloc[0, 2]):
            delta_a = -1
        else:
            delta_a = 0

    if not math.isnan(ranking.iloc[1, 2]) and not math.isnan(ranking.iloc[1, 3]):
        delta_a = (ranking.iloc[1, 3] - ranking.iloc[1, 2]) / \
                  (ranking.iloc[1, 3] + ranking.iloc[1, 2])
    else:
        if not math.isnan(ranking.iloc[1, 3]):
            delta_b = 1
        elif not math.isnan(ranking.iloc[1, 2]):
            delta_b = -1
        else:
            delta_b = 0

    return delta_a, delta_b


def main():
    PROB_A = 0.6
    PROB_B = 1 - PROB_A
    MEAN_A = 1.5
    MEAN_B = -1.5
    VAR_A = 1
    VAR_B = 1
    QUERY_LEN = 10
    NUM_QUERIES = 25
    NUM_ITER = 25
    # NUM_ITERS = [10, 25, 100]
    METRIC = 'avg_position'
    DIST = 'logit_normal'

    # for NUM_ITER in NUM_ITERS:
    metric_a, mean_a = np.empty(NUM_ITER), np.empty(NUM_ITER)
    metric_b, mean_b = np.empty(NUM_ITER), np.empty(NUM_ITER)

    for i in range(NUM_ITER):
        a_metrics, deltas_a = np.empty(NUM_QUERIES), np.empty(NUM_QUERIES)
        b_metrics, deltas_b = np.empty(NUM_QUERIES), np.empty(NUM_QUERIES)

        for j in range(NUM_QUERIES):
            # sample QUERY_LEN subjects from underlying distribution
            arr_a = sample_dist(DIST, MEAN_A, VAR_A, QUERY_LEN, PROB_A)
            arr_b = sample_dist(DIST, MEAN_B, VAR_B, QUERY_LEN, PROB_B)

            # rank subjects according to chosen policy, compute metric
            # ranking = ranking_policies.rank_top_k_alt(arr_a, arr_b)
            # ranking = ranking_policies.rank_max_util(arr_a, arr_b)
            # ranking = ranking_policies.rank_top_k(arr_a, arr_b, 5, PROB_A)
            ranking = ranking_policies.rank_stochastic(arr_a, arr_b)
            a_metrics[j], b_metrics[j] = compute_metric(ranking, METRIC).loc['A'], \
                                         compute_metric(ranking, METRIC).loc['B']

            outcome = selection_policies.select_top_k(ranking, 5)
            deltas_a[j], deltas_b[j] = update_mean(outcome)

        # take the mean of the metrics over the queries at each step
        metric_a[i], metric_b[i] = np.mean(a_metrics), np.mean(b_metrics)

        # update population distributions for next iteration
        # keeping the sum the same
        mean_a[i], mean_b[i] = MEAN_A, MEAN_B
        MEAN_A += np.mean(deltas_a) / 2
        MEAN_B += np.mean(deltas_b)

        # # updating the means in a funny way, need to figure out top k way to do it
        # if abs(MEAN_B - MEAN_A) > 0.01:
        #     if MEAN_B < MEAN_A:
        #         MEAN_B += update_mean(np.mean(arr_b)) / 2
        #         MEAN_A -= update_mean(np.mean(arr_b)) / 2
        #     if MEAN_A < MEAN_B:
        #         MEAN_A += update_mean(np.mean(arr_a)) / 2
        #         MEAN_B -= update_mean(np.mean(arr_a)) / 2

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
