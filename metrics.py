import pandas as pd
import math

# METRICS ===============================================================

def compute_metric(ranking, metric):
    """
    Computes chosen metric to track change over time.
    """
    if metric == 'avg_position':
        return avg_position(ranking)
    elif metric == 'avg_exposure':
        return avg_exposure(ranking)
    else:
        # TODO 
        pass


def avg_position(ranking):
    return ranking.groupby('group')['rank'].mean()


def avg_exposure(ranking):
    # do we even need this?
    return avg_position(ranking).assign(exposure = 1 / math.log2(1 + avg_position(ranking)['rank']))['exposure']

