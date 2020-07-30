import numpy as np
import pandas as pd
import random
import fairsearchcore as fsc
from fairsearchcore.models import FairScoreDoc
from scipy.special import softmax


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


def rank_top_k(arr_a, arr_b, k, p):
    """
    ASSUMPTION: 
        - underlying population (NOT representation in training data!) is 100p% Group A and 
          100(1-p)% Group B, and we want the rankings to accurately reflect those proportions

    INPUT:
        - k is the top-k parameter
        - p is the population proportion of the "disadvantaged" class (B)
        - arr_a and arr_b are ranked lists of subjects from each group, ordered by decreasing relevance 

    * this algorithm implements the FA*IR algorithm as described in https://arxiv.org/pdf/1706.06368.pdf
      using the library 'fairsearchcore', available here: https://github.com/fair-search/fairsearch-fair-python 
    * alpha value is hardcoded as 0.1; this is the value used in the authors' experiments
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
    ranking = pd.DataFrame(columns=['rank', 'relevance', 'group'])
    ranking = ranking.astype({'rank': float, 'relevance': float, 'group': str})
    
    for i, s_id in enumerate(idx):
        if s_id < len(arr_a):
            ranking = ranking.append({'rank': i + 1, 
                                      'relevance': arr_a[s_id], 
                                      'group': 'A'}, 
                                     ignore_index=True)
        else:
            ranking = ranking.append({'rank': i + 1, 
                                      'relevance': arr_b[s_id - len(arr_a)], 
                                      'group': 'B'},
                                     ignore_index=True)

    # handle "leftovers" (subjects not in top-k) using max-util strategy
    # (aka sort by decreasing relevance irrespective of group membership)
    idx_leftover = [i for i in range(len(arr_a) + len(arr_b)) if i not in idx]
    leftovers = [arr[i].score for i in idx_leftover] 
    idx_leftover = np.array(idx_leftover)[np.argsort(leftovers)][::-1]

    for i, s_id in enumerate(idx_leftover):
        if s_id < len(arr_a):
            ranking = ranking.append({'rank': k + i + 1, 
                                      'relevance': arr[s_id].score, 
                                      'group': 'A'}, 
                                     ignore_index=True)
        else:
            ranking = ranking.append({'rank': k + i + 1, 
                                      'relevance': arr[s_id].score, 
                                      'group': 'B'},
                                     ignore_index=True)

    return ranking


def rank_stochastic(arr_a, arr_b):
    """
    INPUT:
        - k is the top-k parameter
        - p is the population proportion of the "disadvantaged" class (B)
        - arr_a and arr_b are ranked lists of subjects from each group, ordered by decreasing relevance 

    * [leave comments here]
    """









    pass


# DEPRECATED, delete later //////////////////////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

def rank_stochastic_alt(arr_a, arr_b):
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
                    ranking = ranking.append({'rank': a + b + 1, 
                                              'relevance': arr_a[i], 
                                              'group': 'A'},
                                             ignore_index=True)
                    a += 1
                    arr_a = np.delete(arr_a, [i])
                else:
                    ranking = ranking.append({'rank': a + b + 1, 
                                              'relevance': arr_b[i - len(arr_a)], 
                                              'group': 'B'},
                                             ignore_index=True)
                    b += 1
                    arr_b = np.delete(arr_b, [i - len(arr_a)])
                break

    return ranking