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
		self.s = torchify([0.1 for i in range(len(rel))])
		self.probs = nn.Softmax(dim=0)(self.s).data.numpy().flatten()


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
			probs_ = np.array(self.probs, copy=True) / self.probs.sum()
			ranking = pd.DataFrame(columns=['rank', 'relevance', 'group'])

			ranks = np.random.choice(num_subjects, size=num_subjects, p=probs_, replace=False)
			ranking = ranking.assign(rank=ranks)
			ranking['rank'] = pd.to_numeric(ranking['rank'])

			relevances = [self.rel[i] for i in ranks]
			ranking = ranking.assign(relevance=relevances)
			ranking['relevance'] = pd.to_numeric(ranking['relevance'])


			groups = ['A' if i < self.len_a else 'B' for i in ranks]
			ranking = ranking.assign(group=groups)

			propensity = 1.0
			# for i in range(num_subjects):
				# propensity *= probs_[ranking[i]]
				# probs_[ranking[i]] = 0.0
				# probs_ /= probs_.sum()

			rankings.append(ranking)
			propensities.append(propensity)
		
		return rankings


	def compute_log_probability(self, ranking):
		"""
		Numerically stable method of calculating log probability of model.
		From "Policy Learning for Fairness in Ranking" (2019). 
		repository: 	https://github.com/ashudeep/Fair-PGRank/
			 paper:		https://arxiv.org/pdf/1902.04056.pdf

		COMMENTS:
			- ranking is the "rank" column of the DataFrame
			- uses logsumexp for numerical stability
		"""

		subtracts = torch.zeros_like(self.s)
		log_probs = torch.zeros_like(self.s)

		relevances = ranking['relevance']
		ranks = ranking['rank']

		for j in range(self.s.size()[0]):
			# pos_j = position of item j
			idx = relevances[relevances == self.rel[j]].index[0]
			pos_j = ranks.iloc[idx]
			log_probs[j] = self.s[pos_j] - logsumexp(self.s - subtracts)
			subtracts[pos_j] = self.s[pos_j] + 1e6

		return torch.sum(log_probs)


	def reinforce_loss(self, rankings):
		"""
		Calculates total loss using "baseline for variance reduction" trick.
		From "Policy Learning for Fairness in Ranking" (2019). 
		repository: 	https://github.com/ashudeep/Fair-PGRank/
			 paper:		https://arxiv.org/pdf/1902.04056.pdf

		INPUT:
			rankings:	DataFrame of rankings, where each ranking is itself a 
						DataFrame with columns ["rank", "relevance", "group"]	
	
		COMMENTS:
			- uses log-derivative trick to simplify computation
		"""

		loss = 0


		baseline = np.mean([NDCG(r['relevance']) for r in rankings])
		print('BASELINE', baseline)

		for r in rankings:
			reward = NDCG(r['relevance'])
			log_prob = self.compute_log_probability(r)
			loss += float(-(reward - baseline)) * log_prob


		print('LOSS', loss)
		return loss / len(rankings)
