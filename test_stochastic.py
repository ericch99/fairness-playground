import numpy as np
from ranking_policies import rank_stochastic

from utils import *


relevances = np.random.random_sample(20)

print('============================max-util ranking==================================')
arr_a = list(np.sort(relevances)[::-1][:10])
arr_b = list(np.sort(relevances)[::-1][10:])
# print('A:', arr_a)
# print('B:', arr_b)

rankings = rank_stochastic(arr_a, arr_b)

# for r in rankings:
print(rankings[0])

# print('NDCG:', NDCG(rankings[0]['relevance']))