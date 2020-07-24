import numpy as np
import pandas as pd
import random
from scipy.special import softmax


# RANKING POLICIES ======================================================

# TODO: not debugged yet

# ASSUMPTION: underlying population is p% Group A and (1-p)% Group B, 
# and we want the rankings to accurately reflect those proportions
#   - addendum: may want to generalize past 50-50 population proportions


def rank_top_k_alt(arr_a, arr_b):
    ranking = pd.DataFrame(columns=['rank', 'relevance', 'group'])
    ranking = ranking.astype({'rank': float, 'relevance': float, 'group': str})
    a, b = 0, 0

    while a < len(arr_a) and b < len(arr_b):
        if arr_a[a] > arr_b[b]:
            ranking = ranking.append({'rank': a + b + 1, 'relevance': arr_a[a], 'group': 'A'}, ignore_index=True)
            ranking = ranking.append({'rank': a + b + 2, 'relevance': arr_b[b], 'group': 'B'}, ignore_index=True)
        else:
            ranking = ranking.append({'rank': a + b + 1, 'relevance': arr_b[b], 'group': 'B'}, ignore_index=True)
            ranking = ranking.append({'rank': a + b + 2, 'relevance': arr_a[a], 'group': 'A'}, ignore_index=True)

        a += 1
        b += 1

    # leftovers
    while a < len(arr_a):
        ranking = ranking.append({'rank': a + b + 1, 'relevance': arr_a[a], 'group': 'A'}, ignore_index=True)
        a += 1
    while b < len(arr_b):
        ranking = ranking.append({'rank': a + b + 1, 'relevance': arr_b[b], 'group': 'B'}, ignore_index=True)
        b += 1

    return ranking


# TODO ===========================================================================================
# do we need to keep this?

def rank_top_k(arr_a, arr_b, k, prob_a):
    # round k to nearest integers
    k_a = int(k * prob_a)
    k_b = int(k * (1 - prob_a))

    ranking = rank_max_util(arr_a[:k_a], arr_b[:k_b]).append(rank_max_util(arr_a[k_a:], arr_b[k_b:]))
    for i in range(len(ranking.index)):
        ranking.iloc[i, ranking.columns.get_loc('rank')] = i + 1

    return ranking

# END TODO ///////////////////////////////////////////////////////////////////////////////////////


def rank_max_util(arr_a, arr_b):
    ranking = pd.DataFrame(columns=['rank', 'relevance', 'group'])
    ranking = ranking.astype({'rank': float, 'relevance': float, 'group': str})
    a, b = 0, 0

    while a < len(arr_a) and b < len(arr_b):
        if arr_a[a] > arr_b[b]:
            ranking = ranking.append({'rank': a + b + 1, 'relevance': arr_a[a], 'group': 'A'}, ignore_index=True)
            a += 1
        else:
            ranking = ranking.append({'rank': a + b + 1, 'relevance': arr_b[b], 'group': 'B'}, ignore_index=True)
            b += 1

    # leftovers
    while a < len(arr_a):
        ranking = ranking.append({'rank': a + b + 1, 'relevance': arr_a[a], 'group': 'A'}, ignore_index=True)
        a += 1
    while b < len(arr_b):
        ranking = ranking.append({'rank': a + b + 1, 'relevance': arr_b[b], 'group': 'B'}, ignore_index=True)
        b += 1

    return ranking


# TODO:
#   - talk to Amanda (and Alex) about how we should do this!

def rank_stochastic(arr_a, arr_b):
    ranking = pd.DataFrame(columns=['rank', 'relevance', 'group'])
    ranking = ranking.astype({'rank': float, 'relevance': float, 'group': str})
    a, b = 0, 0

    while len(arr_a) > 0 or len(arr_b) > 0:
        s = softmax(np.append(arr_a, arr_b))
        rand = random.uniform(0, 1)
        summer = 0
        for i in range(len(s)):
            summer += s[i]
            if rand < summer:
                if i in range(0, len(arr_a)):
                    ranking = ranking.append({'rank': a + b + 1, 'relevance': arr_a[i], 'group': 'A'},
                                             ignore_index=True)
                    a += 1
                    arr_a = np.delete(arr_a, [i])
                else:
                    ranking = ranking.append({'rank': a + b + 1, 'relevance': arr_b[i - len(arr_a)], 'group': 'B'},
                                             ignore_index=True)
                    b += 1
                    arr_b = np.delete(arr_b, [i - len(arr_a)])
                break

    return ranking