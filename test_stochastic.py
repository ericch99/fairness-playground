import numpy as np
from ranking_policies import rank_stochastic

from utils import *


relevances = np.random.random_sample(20)

print('=========================== stochastic ranking =================================')
arr_a = list(np.sort(relevances)[::-1][:10])
arr_b = list(np.sort(relevances)[::-1][10:])

rankings = rank_stochastic(arr_a, arr_b)

for i, r in enumerate(rankings):
	print(i, 'NDCG:', NDCG(rankings[i]['relevance']))
	print(r)
