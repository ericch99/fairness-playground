import numpy as np
from ranking_policies import rank_top_k

from utils import *


relevances = np.random.random_sample(20)

print('====== top-k ranking ======')
arr_a = list(np.sort(relevances)[::-1][:10])
arr_b = list(np.sort(relevances)[::-1][10:])

ranking = rank_top_k(arr_a, arr_b, k=10, p=0.5)

print(ranking)

print('NDCG:', NDCG(ranking['relevance']))