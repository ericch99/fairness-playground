import numpy as np


def NDCG(rel_r):
	"""
	Computes normalized discounted cumulative gain of a given ranking.

	INPUT:
   		rel_r:	length-n list of relevances corresponding to ranking

	OUTPUT:
    	NDCG computed for given ranking (float) 
	"""
	
	discount = [1 / np.log2(i + 1) for i in range(1, len(rel_r) + 1)] 
	rel_sort = np.sort(rel_r)[::-1]

	DCG_max = np.sum(np.divide(np.power(2, rel_sort) - 1, discount))
	
	DCG_r = np.sum(np.divide(np.power(2, rel_r) - 1, discount))

	return DCG_r / DCG_max


def sample_PL(s):
	"""
	Samples Plackett-Luce model for ranking distribution. See 
	"Policy Learning for Fairness in Ranking" for more detail.
	"""

	rankings = []

	for i in range(10):
		







		pass

	return rankings


def loss(s):

	rankings = sample_PL(s)
	
