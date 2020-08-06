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
        raise NotImplementedError


# METRICS ============================================================

def avg_position(ranking):
    return ranking.groupby('group')['index'].mean()


def avg_exposure(ranking):
    # do we even need this? log discounting
    idx_a = ranking[ranking['group'] == 'A']['index']
    idx_b = ranking[ranking['group'] == 'B']['index']
    exposure_a = np.mean([1 / math.log2(1 + (i + 1)) for i in idx_a])
    exposure_b = np.mean([1 / math.log2(1 + (i + 1)) for i in idx_b])

    return exposure_a, exposure_b

