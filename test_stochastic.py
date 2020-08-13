import numpy as np
from ranking_policies import *

from utils import *


relevances = np.random.random_sample(20)

print('=========================== stochastic ranking =================================')
arr_a = [0.9 - 0.01 * i for i in range(10)]
arr_b = [0.8 - 0.01 * i for i in range(10)]

rankings = rank_stochastic(arr_a, arr_b)
special = rank_max_util(arr_a, arr_b)

for i, r in enumerate(rankings):
	print(i, 'NDCG:', NDCG(rankings[i]['relevance']))
	print(r)


	merit = r.groupby('group')['relevance'].mean()
	M_G_A, M_G_B = merit.loc['A'], merit.loc['B']
	v_G_A = np.mean([1 / np.log2((idx + 1) + 1) for idx in list(np.where(r['group'] == 'A'))])
	v_G_B = np.mean([1 / np.log2((idx + 1) + 1) for idx in list(np.where(r['group'] == 'B'))])
	print('ratio for Group A:', np.mean(v_G_A) / M_G_A)
	print('ratio for Group B:', np.mean(v_G_B) / M_G_B)
	print(np.mean(v_G_A) / M_G_A < (np.mean(v_G_B) / M_G_B))

print('MAX-UTIL NDCG:', NDCG(special['relevance']))
print(special)
merit = special.groupby('group')['relevance'].mean()
M_G_A, M_G_B = merit.loc['A'], merit.loc['B']
v_G_A = np.mean([1 / np.log2((idx + 1) + 1) for idx in list(np.where(special['group'] == 'A'))])
v_G_B = np.mean([1 / np.log2((idx + 1) + 1) for idx in list(np.where(special['group'] == 'B'))])
print('ratio for Group A:', np.mean(v_G_A) / M_G_A)
print('ratio for Group B:', np.mean(v_G_B) / M_G_B)
print(np.mean(v_G_A) / M_G_A < (np.mean(v_G_B) / M_G_B))