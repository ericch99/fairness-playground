from ranking_policies import rank_top_k


arr_a = [0.81, 0.8, 0.76, 0.75, 0.73, 0.6, 0.55, 0.54, 0.53, 0.52, 0.5, 0.49]
arr_b = [0.53, 0.52, 0.51, 0.5, 0.44, 0.43, 0.36, 0.35, 0.3, 0.29, 0.2, 0.11]

k = 10
p = 0.5

ranking = rank_top_k(arr_a, arr_b, k, p)
print(ranking)