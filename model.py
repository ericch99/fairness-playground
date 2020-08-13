import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from utils import *


class PlackettLuce():
	"""
	A wrapper class for the Plackett-Luce model. 
	"""

	def __init__(self, rel, len_a):
		self.rel = rel
		self.len_a = len_a
		self.s = torchify([np.random.random_sample() for i in range(len(rel))])
		self.probs = nn.Softmax(dim=0)(self.s)


	def sample_rankings(self, size):
		"""
		Samples `size` rankings from the Plackett-Luce model. 
		From "Policy Learning for Fairness in Ranking" (2019). 
		repository: 	https://github.com/ashudeep/Fair-PGRank/
			 paper:		https://arxiv.org/pdf/1902.04056.pdf

		INPUT: 
			size:			number of rankings to sample

		OUTPUT:
			rankings:		length-(size) list of length-(num_subjects) rankings
			propensities:	TODO -- figure out what this is, lol
		"""

		num_subjects = self.probs.shape[0]
		rankings, propensities = [], []
		
		for i in range(size):
			probs_ = self.probs.data.numpy().flatten()

			ranking = pd.DataFrame(columns=['index', 'relevance', 'group'])

			indices = np.random.choice(num_subjects, size=num_subjects, p=probs_, replace=False)
			ranking = ranking.assign(index=indices)
			ranking['index'] = pd.to_numeric(ranking['index'])

			relevances = [self.rel[i] for i in indices]
			ranking = ranking.assign(relevance=relevances)
			ranking['relevance'] = pd.to_numeric(ranking['relevance'])

			groups = ['A' if i < self.len_a else 'B' for i in indices]
			ranking = ranking.assign(group=groups)
			
			rankings.append(ranking)
		
		return rankings


	def compute_log_probability(self, ranking):
		"""
		Numerically stable method of calculating log probability of model.
		From "Policy Learning for Fairness in Ranking" (2019). 
		repository: 	https://github.com/ashudeep/Fair-PGRank/
			 paper:		https://arxiv.org/pdf/1902.04056.pdf

		COMMENTS:
			- ranking is the "index" column of the DataFrame
			- uses logsumexp for numerical stability
		"""

		subtracts = torch.zeros_like(self.s)
		log_probs = torch.zeros_like(self.s)

		index = ranking['index']

		for j in range(self.s.size()[0]):
			# pos_j = item at position j
			pos_j = index.iloc[j]
			log_probs[j] = self.s[pos_j] - logsumexp(self.s - subtracts)
			subtracts[pos_j] = self.s[pos_j] + 1e3

		return torch.sum(log_probs)


	def calculate_loss(self, rankings):
		"""
		Calculates loss using "baseline for variance reduction" trick and stores gradients.
		From "Policy Learning for Fairness in Ranking" (2019). 
		repository: 	https://github.com/ashudeep/Fair-PGRank/
			 paper:		https://arxiv.org/pdf/1902.04056.pdf

		INPUT:
			rankings:	list of rankings, where each ranking is itself a 
						DataFrame with columns ["index", "relevance", "group"]	

		OUTPUT:
			average NDCG of the sampled rankings. 
	
		COMMENTS:
			- uses log-derivative trick to simplify computation of gradient
		"""

		rewards = []
		baseline = np.mean([NDCG(r['relevance']) for r in rankings])
		regularizer = compute_fairness_regularizer(rankings)
		
		for r in rankings:
			reward = NDCG(r['relevance'])
			rewards.append(reward)
			log_prob = self.compute_log_probability(r)
			loss = float(-1e2 * (reward - baseline - regularizer)) * log_prob
			loss.backward(retain_graph=True)

			# entropy regularizer to enforce stochasticity
			entropy_loss = torch.sum(torch.log(self.probs) * self.probs)
			entropy_loss.backward(retain_graph=True)

		return np.mean(rewards)


	def update_probs(self):
		self.probs = nn.Softmax(dim=0)(self.s) 
		# print('SCORES:', self.s)
		# print('PROBS:', self.probs)
