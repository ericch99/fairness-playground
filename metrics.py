import pandas as pd
import math


def compute_metric(ranking, metric):
    """
    Computes chosen metric to track change over time.
    """
    if metric == 'avg-position':
        return avg_position(ranking)
    elif metric == 'avg-exposure':
        return avg_exposure(ranking)
    else:
        # TODO 
        pass


# METRICS ===================================================================================================

def avg_position(ranking):
    return ranking.groupby('group')['rank'].mean()


def avg_exposure(ranking):
    # do we even need this? log discounting
    exposure = 1 / math.log2(1 + avg_position(ranking)['rank'])
    return avg_position(ranking).assign(exposure=exposure)['exposure']

