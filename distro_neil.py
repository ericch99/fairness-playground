from scipy.stats import beta
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

# TODO
# need to figure out how to update means to change distribution at each iteration
# def update_means():
# need to incorporate NDCG somewhere
# fair ranking policy to go along with max utility ranking (for comparison)


def rank_fair(arr_a, arr_b):
    a, b = 0, 0
    rank_a = arr_a
    rank_b = arr_b
    while a < len(arr_a) and b < len(arr_b):
        if arr_a[a] > arr_b[b]:
            rank_a[a] = a + b + 1
            a = a + 1
            # can decide to implement a different fair policy other than filing like this
            rank_b[b] = a + b + 1
            b = b + 1
        else:
            rank_b[b] = a + b + 1
            b = b + 1
            # can decide to implement a different fair policy other than filing like this
            rank_a[a] = a + b + 1
            a = a + 1
    # get the leftovers
    while a < len(arr_a):
        rank_a[a] = a + b + 1
        a = a + 1
    while b < len(arr_b):
        rank_b[b] = a + b + 1
        b = b + 1
    # can use metrics other than average
    return np.average(rank_a), np.average(rank_b)


def rank_max_util(arr_a, arr_b):
    a, b = 0, 0
    rank_a = arr_a
    rank_b = arr_b
    while a < len(arr_a) and b < len(arr_b):
        if arr_a[a] > arr_b[b]:
            rank_a[a] = a + b + 1
            a = a + 1
        else:
            rank_b[b] = a + b + 1
            b = b + 1
    # get the leftovers
    while a < len(arr_a):
        rank_a[a] = a + b + 1
        a = a + 1
    while b < len(arr_b):
        rank_b[b] = a + b + 1
        b = b + 1
    # can use metrics other than average
    return np.average(rank_a), np.average(rank_b)


# can play around with using a different distribution (Normal, Poisson)
def sample_beta(mean, var, l, prop):
    a = ((1 - mean) / var - 1 / mean) * mean * mean
    b = a * (1 / mean - 1)
    arr = np.array(beta.rvs(a, b, size=int(l * prop)))
    arr = np.sort(arr)
    arr = arr[::-1]
    return arr


def sample_norm(mean, var, l, prop):
    arr = np.array(norm.rvs(loc=mean, scale=var, size=int(l * prop)))
    arr = np.sort(arr)
    arr = arr[::-1]
    return arr


def main():
    average_a = np.empty(NUM_ITER)
    average_b = np.empty(NUM_ITER)
    for i in range(NUM_ITER):
        arr_a = sample_norm(MEAN_A, VAR_A, QUERY_LEN, PROP_A)
        arr_b = sample_norm(MEAN_B, VAR_B, QUERY_LEN, PROP_B)
        avg_a, avg_b = rank_max_util(arr_a, arr_b)
        average_a[i] = avg_a
        average_b[i] = avg_b
        # update_means()

    plt.plot(np.arange(NUM_ITER), average_a, color='C2', label='Group A Avg Pos')
    plt.plot(np.arange(NUM_ITER), average_b, color='C0', label='Group B Avg Pos')
    plt.show()


PROP_A = 0.7
PROP_B = 1 - PROP_A
QUERY_LEN = 10
MEAN_A = 0.55
MEAN_B = 0.45
VAR_A = 0.1
VAR_B = 0.1
NUM_ITER = 100

if __name__ == '__main__':
    main()
