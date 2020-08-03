import numpy as np
import torch


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
	
	discount = np.fromiter(1 / torch.log2(i + 1) for i in range(1, len(rel_r) + 1), float)
	rel_sort = np.sort(rel_r)[::-1]

	DCG_max = np.sum(np.divide(np.power(2, rel_sort) - 1, discount))
	DCG_r = np.sum(np.divide(np.power(2, rel_r) - 1, discount))

	return DCG_r / DCG_max


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
