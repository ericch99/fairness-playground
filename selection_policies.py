import numpy as np
import pandas as pd
import math
from scipy.stats import bernoulli


def select_policy(ranking, k, policy):
	if policy == 'top-k':
		return select_top_k(ranking, k)
	elif policy == 'stochastic':
		return select_stochastic(ranking, k)
	else:
		# TODO
		pass


# SELECTION POLICIES =====================================================================

def select_top_k(ranking, k):
	"""
	Method 1 (deterministic):
	Selects top-k subjects as measured by the given ranking.
	Each selected individual then succeeds with probability of success equal to relevance.
	"""

	# selection
	selected = [1 for i in range(k)]
	selected.extend([0 for i in range(ranking.shape[0] - k)])

	succeeded = []

	# testing for success ('interviews')
	for i in range(k):
		trial = bernoulli.rvs(ranking.iloc[i]['relevance'])
		succeeded.append(trial)

	succeeded.extend([0 for i in range(ranking.shape[0] - k)])

	ranking['selected'] = selected
	ranking['succeeded'] = succeeded

	return ranking


def select_stochastic(ranking, k):
	"""
	Method 2 (stochastic):
	Selects k subjects in the following manner:
		- starting at the top-ranked subject (index 0), select them with probability based on position bias
			* we'll use a log discounting "click model": 1 / log_2(1 + i) for i = 1, 2, 3, ...
		- if not selected, move down a rank and "click on" the next subject
		- stop when we have k selected subjects or reach the bottom of the list
	Each selected individual then succeeds with probability of success equal to relevance.
	"""
	
	count = 0
	pos = 1
	selected = []

	# selection
	while count < k and pos - 1 < ranking.shape[0]:
		trial = bernoulli.rvs(1 / math.log2(1 + pos))
		if trial == 1:
			count += 1
		
		selected.append(trial)
		pos += 1

	if len(selected) != ranking.shape[0]:
		selected.extend([0 for i in range(ranking.shape[0] - len(selected))])
	ranking['selected'] = selected

	# testing for success ('interviews')
	succeeded = np.zeros(ranking.shape[0])

	for i in [i for i, x in enumerate(selected) if x == 1]:
		trial = bernoulli.rvs(ranking.iloc[i]['relevance'])
		succeeded[i] = trial

	ranking['succeeded'] = succeeded

	return ranking
