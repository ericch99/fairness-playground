import numpy as np
import json
import torch
from torch.autograd import Variable


def torchify(s):
	return Variable(torch.FloatTensor(s), requires_grad=True)


def NDCG(rel_r):
	"""
	Computes normalized discounted cumulative gain (NDCG) of a given ranking.
	
	INPUT:
   		rel_r:	length-n list of relevances corresponding to ranking

	OUTPUT:
    	NDCG computed for given ranking (float) 
	"""
	
	discount = [np.log2(i + 1) for i in range(1, len(rel_r) + 1)]
	rel_sort = np.sort(rel_r)[::-1]

	DCG_max = np.sum(np.divide(np.power(2, rel_sort) - 1, discount))
	DCG_r = np.sum(np.divide(np.power(2, rel_r) - 1, discount))

	return DCG_r / DCG_max


def compute_fairness_regularizer(rankings):
	"""
	Computes exposure-based regularization term for fairness. 
	
	INPUT:
   		rankings:	length-n list of sampled rankings

	OUTPUT:
    	value of regularizer (float) 
	"""

	merit = rankings[0].groupby('group')['relevance'].mean()
	M_G_A, M_G_B = merit.loc['A'], merit.loc['B']

	v_G_A, v_G_B = [], []
	for r in rankings:
		v_G_A.append(np.mean([1 / np.log2((idx + 1) + 1) for idx in list(np.where(r['group'] == 'A'))]))
		v_G_B.append(np.mean([1 / np.log2((idx + 1) + 1) for idx in list(np.where(r['group'] == 'B'))]))

	if M_G_A == 0 or M_G_B == 0:
		return 0
	elif M_G_A > M_G_B:
		return max(0, (np.mean(v_G_A) / M_G_A) - (np.mean(v_G_B) / M_G_B))
	else: # if M_G_B >= M_G_A 
		return max(0, (np.mean(v_G_B) / M_G_B) - (np.mean(v_G_A) / M_G_A))


def logsumexp(inputs):
    """
    Numerically stable logsumexp function.  
    From "Policy Learning for Fairness in Ranking" (2019). 
    repository: 	https://github.com/ashudeep/Fair-PGRank/
		 paper:		https://arxiv.org/pdf/1902.04056.pdf

    INPUT:
        inputs: A Variable with any shape.

    OUTPUT:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).

	NOTES:
		For a 1-D array x (any array along a single dimension),
	    logsumexp(x) = s + logsumexp(x - s)
	    with s = max(x) being a common choice.
    """

    s, _ = torch.max(inputs, dim=0, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=0, keepdim=True).log()

    return outputs

def calculate_ratio(ranking):
	"""
	TODO:
		- this function is for debugging purposes
		- it should return the exposure-to-merit ratio of each group
		- ensures stochastic ranking constraint is met and enforced 
	"""

	exposure_a, exposure_b = 0, 0

	return exposure_a, exposure_b