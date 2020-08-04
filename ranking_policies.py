import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

import fairsearchcore as fsc
from fairsearchcore.models import FairScoreDoc

from model import PlackettLuce
from utils import *


def rank_policy(arr_a, arr_b, policy, **kwargs):
    if policy == 'top-k':
        k, p = kwargs['k'], kwargs['p'] 
        return rank_top_k(arr_a, arr_b, k, p)
    elif policy == 'max-util':
        return rank_max_util(arr_a, arr_b)
    elif policy == 'stochastic':
        return rank_stochastic(arr_a, arr_b)
    else:
        raise NotImplementedError


# RANKING POLICIES ==============================================================================================

def rank_max_util(arr_a, arr_b):
    """
    Ranks subjects by max-util ("colorblind") strategy.
    Equivalent to sorting relevances in decreasing order. 
    """

    ranking = pd.DataFrame(columns=['index', 'relevance', 'group'])
    ranking = ranking.astype({'index': float, 'relevance': float, 'group': str})
    a, b = 0, 0

    while a < len(arr_a) and b < len(arr_b):
        if arr_a[a] > arr_b[b]:
            ranking = ranking.append({'index': a + b, 'relevance': arr_a[a], 'group': 'A'}, ignore_index=True)
            a += 1
        else:
            ranking = ranking.append({'index': a + b, 'relevance': arr_b[b], 'group': 'B'}, ignore_index=True)
            b += 1

    # leftovers
    while a < len(arr_a):
        ranking = ranking.append({'index': a + b, 'relevance': arr_a[a], 'group': 'A'}, ignore_index=True)
        a += 1
    while b < len(arr_b):
        ranking = ranking.append({'index': a + b, 'relevance': arr_b[b], 'group': 'B'}, ignore_index=True)
        b += 1

    return ranking


def rank_top_k(arr_a, arr_b, k, p):
    """
    Implements fair ranking with top-k constraint using the FA*IR algorithm as described in 
    "FA*IR: A Fair Top-k Ranking Algorithm" (2017).
    package:     https://github.com/fair-search/fairsearch-fair-python (named fairsearchcore)
      paper:     https://arxiv.org/pdf/1706.06368.pdf

    ASSUMPTION: 
        - underlying population (NOT representation in training data!) is 100p% Group A and 
          100(1-p)% Group B, and we want the rankings to accurately reflect those proportions

    INPUT:
        - k is the top-k parameter
        - p is the population proportion of the "disadvantaged" class (B)
        - arr_a and arr_b are ranked lists of subjects from each group, ordered by decreasing relevance 

    COMMENTS:
        - alpha value is hardcoded as 0.1; this is the value used in the authors' experiments
    """

    # combine lists, wrap in FairScoreDoc objects
    arr = [FairScoreDoc(i, arr_a[i], False) if i < len(arr_a) else \
           FairScoreDoc(i, arr_b[i - len(arr_a)], True) for i in range(len(arr_a) + len(arr_b))] 

    # create fair object, generate fair ranking
    fair = fsc.Fair(k, p, alpha=0.1)
    while True:
        ranking = fair.re_rank(arr)
        if fair.is_fair(ranking):
            break

    # transform ranking list to dataframe
    idx = [s.id for s in ranking]
    ranking = pd.DataFrame(columns=['index', 'relevance', 'group'])
    ranking = ranking.astype({'index': float, 'relevance': float, 'group': str})
    
    for s_id in idx:
        if s_id < len(arr_a):
            ranking = ranking.append({'index': s_id, 
                                      'relevance': arr[s_id].score, 
                                      'group': 'A'}, 
                                     ignore_index=True)
        else:
            ranking = ranking.append({'index': s_id, 
                                      'relevance': arr[s_id].score, 
                                      'group': 'B'},
                                     ignore_index=True)

    # handle "leftovers" (subjects not in top-k) using max-util strategy
    # (aka sort by decreasing relevance irrespective of group membership)
    idx_leftover = [i for i in range(len(arr_a) + len(arr_b)) if i not in idx]
    leftovers = [arr[i].score for i in idx_leftover] 
    idx_leftover = np.array(idx_leftover)[np.argsort(leftovers)][::-1]

    for s_id in idx_leftover:
        if s_id < len(arr_a):
            ranking = ranking.append({'index': s_id, 
                                      'relevance': arr[s_id].score, 
                                      'group': 'A'}, 
                                     ignore_index=True)
        else:
            ranking = ranking.append({'index': s_id, 
                                      'relevance': arr[s_id].score, 
                                      'group': 'B'},
                                     ignore_index=True)

    return ranking


def rank_stochastic(arr_a, arr_b):
    """
    Implements stochastic ranking policy with exposure constraint as given in 
    "Policy Learning for Fairness in Ranking" (2019). 
    repository:     https://github.com/ashudeep/Fair-PGRank/
         paper:     https://arxiv.org/pdf/1902.04056.pdf

    OUTPUT:
        10 sampled rankings from the optimized model

    COMMENTS:    
        - STOPPING CRITERIA: small change in NDCG 
        - 
    """

    rel = arr_a.copy()
    rel.extend(arr_b)
    model = PlackettLuce(rel, len(arr_a))
    
    optimizer = optim.SGD([model.s], lr=1e-2)

    delta_NDCG = 10000
    prev, counter = 0, 0
    
    while abs(delta_NDCG) > 0.01:
        # sample from distribution
        rankings = model.sample_rankings(10)
        
        # estimate gradient from samples
        optimizer.zero_grad()
        NDCG = model.calculate_loss(rankings)
    
        # take gradient step, recompute scores
        optimizer.step()

        # update delta_NDCG
        delta_NDCG = NDCG - prev
        prev = NDCG
        print(delta_NDCG)

        # display counter
        counter += 1
        if counter % 10 == 0:
            print(f'\tITERATION {counter}')

    return model.sample_rankings(10)
