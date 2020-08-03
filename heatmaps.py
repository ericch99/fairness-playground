import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
from distributions import *
from metrics import *
from ranking_policies import *
from selection_policies import *

sns.set(style='darkgrid')

"""
NOTES:
    - treat Group B as the protected class
"""


def update_mean(ranking, succeed, fail, none):
    ranking['delta'] = [succeed if (row.selected and row.succeeded)
                        else (fail if row.selected else none)
                        for row in ranking.itertuples()]
    ranking = ranking.groupby('group').mean()['delta']
    return ranking['A'], ranking['B']


def main():
    # initialize constants =================================================================

    PROB_A = 0.6

    MEAN_A = 1
    MEAN_B = 0
    VAR_A = 1
    VAR_B = 1

    QUERY_LEN = 20
    NUM_QUERIES = 10

    METRIC = 'avg-position'
    DIST = 'logit-normal'
    k = 6

    NEUTRAL = 0
    SUCCESSES = [0.25, 0.5, 0.75, 1]
    FAILS = [-0.25, -0.5, -0.75, -1]
    matrix = np.zeros(shape=(len(SUCCESSES), len(FAILS)))

    # NUM_ITERS = [10, 25, 100]
    # RANKING_POLICIES = ['top-k', 'max-util']
    # SELECTION_POLICIES = ['top-k', 'stochastic']
    NUM_ITER = 10
    RANK_POLICY = 'top-k'
    SELECT_POLICY = 'stochastic'

    for r in range(len(SUCCESSES)):
        for c in range(len(FAILS)):
            # run simulation ======================================================================

            mean_a = np.empty(NUM_ITER)
            mean_b = np.empty(NUM_ITER)

            # ITERATIONS: length of time horizon
            for i in trange(NUM_ITER, desc='iterations'):
                deltas_a = np.empty(NUM_QUERIES)
                deltas_b = np.empty(NUM_QUERIES)

                # QUERIES: number of simulations per time step
                for j in trange(NUM_QUERIES, desc='queries'):
                    # sample subjects from underlying distribution
                    arr_a = sample_dist(MEAN_A, VAR_A, QUERY_LEN, PROB_A, DIST)
                    arr_b = sample_dist(MEAN_B, VAR_B, QUERY_LEN, 1 - PROB_A, DIST)

                    # rank subjects according to chosen policy, select individuals
                    ranking = rank_policy(arr_a, arr_b, RANK_POLICY, k=k, p=PROB_A)
                    result = select_policy(ranking, k, SELECT_POLICY)

                    # compute change in population distributions
                    deltas_a[j], deltas_b[j] = update_mean(result, SUCCESSES[r], FAILS[c], NEUTRAL)

                # update population distributions for next iteration
                mean_a[i], mean_b[i] = MEAN_A, MEAN_B
                MEAN_A += np.mean(deltas_a)
                MEAN_B += np.mean(deltas_b)

            matrix[r][c] = MEAN_A - MEAN_B
            # plot the change in relevance over time
            plt.cla()
            plt.plot(np.arange(NUM_ITER), mean_a, color='C2', label=f'Group A mean')
            plt.plot(np.arange(NUM_ITER), mean_b, color='C0', label=f'Group B mean')
            plt.savefig(f'figures/heatmaps/relevance-{SUCCESSES[r]}-{FAILS[c]}.png', dpi=72)

    plt.cla()
    heat_map = sns.heatmap(matrix)
    plt.savefig(f'figures/heatmaps/feedback_coefficients.png', dpi=72)


if __name__ == '__main__':
    main()
