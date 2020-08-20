import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange

from distributions import *
from metrics import *
from ranking_policies import *
from selection_policies import *


class Simulation():
    def __init__(self, A, B, sim, penalty):
        self.MEAN_A = A['mean']
        self.MEAN_B = B['mean']
        self.VAR_A = A['var']
        self.VAR_B = B['var']
        self.PROB_A = A['prob']
        self.PROB_B = 1 - self.PROB_A

        self.METRIC = sim['metric']
        self.DIST = sim['dist']
        
        self.k = sim['k']
        self.RANK_POLICY = sim['r_policy']
        self.SELECT_POLICY = sim['s_policy']
        self.QUERY_LEN = sim['query_len']

        self.penalties = [penalty['pos'], penalty['neg'], penalty['non']]


    def run_simulation(self, n):
        metric_a, mean_a = np.empty(n), np.empty(n)
        metric_b, mean_b = np.empty(n), np.empty(n)
        
        for i in trange(n, desc='iterations'):
            # run a single iteration
            self.run_iteration(i, metric_a, mean_a, metric_b, mean_b)                

        return metric_a, mean_a, metric_b, mean_b 


    def run_iteration(self, i, metric_a, mean_a, metric_b, mean_b):
        metrics_a, deltas_a = np.empty(10), np.empty(10)
        metrics_b, deltas_b = np.empty(10), np.empty(10)

        for j in trange(10, desc='queries'):
            self.run_query(j, metrics_a, deltas_a, metrics_b, deltas_b)

        # take the mean of the metrics over the queries at each step
        metric_a[i], metric_b[i] = np.mean(metrics_a), np.mean(metrics_b)
    
        # update population distributions for next iteration, keeping the sum the same
        mean_a[i], mean_b[i] = self.MEAN_A, self.MEAN_B
        self.update_means(deltas_a, deltas_b)

    def run_query(self, j, metrics_a, deltas_a, metrics_b, deltas_b):
        # sample subjects from underlying distribution
        arr_a = sample_dist(self.MEAN_A, self.VAR_A, self.QUERY_LEN, self.PROB_A, self.DIST)
        arr_b = sample_dist(self.MEAN_B, self.VAR_B, self.QUERY_LEN, self.PROB_B, self.DIST)

        # rank subjects according to chosen policy
        rankings = rank_policy(arr_a, arr_b, self.RANK_POLICY, k=self.k, p=self.PROB_B)

        if self.RANK_POLICY == 'stochastic':
            deltas_a_j, deltas_b_j = np.zeros_like(rankings), np.zeros_like(rankings)
            metrics_a_j, metrics_b_j = np.zeros_like(rankings), np.zeros_like(rankings)

            for i, r in enumerate(rankings):

                # select individuals according to chosen policy
                result = select_policy(r, self.k, self.SELECT_POLICY)
               
                # compute chosen fairness metric
                metric = compute_metric(r, self.METRIC)
                metrics_a_j[i], metrics_b_j[i] = metric.loc['A'], metric.loc['B']
               
                # compute change in distributions
                deltas_a_j[i], deltas_b_j[i] = self.calculate_delta(result)

            metrics_a[j], metrics_b[j] = np.mean(metrics_a_j), np.mean(metrics_b_j)
            deltas_a[j], deltas_b[j] = np.mean(deltas_a_j), np.mean(deltas_b_j)

        else:
            # select individuals according to chosen policy 
            result = select_policy(rankings, self.k, self.SELECT_POLICY)

            # compute chosen fairness metric
            metric = compute_metric(rankings, self.METRIC)
            metrics_a[j], metrics_b[j] = metric.loc['A'], metric.loc['B']

            # compute change in distributions
            deltas_a[j], deltas_b[j] = self.calculate_delta(result)

    def calculate_delta(self, result):
        result['delta'] = [self.penalties[0] if (row.selected and row.succeeded) 
                           else (self.penalties[1] if row.selected else self.penalties[2]) 
                           for row in result.itertuples()]
        
        result = result.groupby('group').mean()['delta']

        return result['A'], result['B']

    def update_means(self, deltas_a, deltas_b):
        self.MEAN_A += np.mean(deltas_a)
        self.MEAN_B += np.mean(deltas_b)

    def plot_metrics(self, n, metric_a, mean_a, metric_b, mean_b):
        sns.set(style='darkgrid')

        # plot change in desired metric over time
        plt.cla()
        plt.plot(np.arange(n), metric_a, color='C2', label=f'Group A {self.METRIC}')
        plt.plot(np.arange(n), metric_b, color='C0', label=f'Group B {self.METRIC}')
        plt.savefig(f'figures/final_sims/varying_k/{self.RANK_POLICY}-{self.SELECT_POLICY}-{self.METRIC}_{n}.pdf', dpi=300)

        # plot the change in relevance over time
        plt.cla()
        plt.plot(np.arange(n), 1 / (1 + np.exp(-mean_a)), color='C2', label=f'Group A relevance')
        plt.plot(np.arange(n), 1 / (1 + np.exp(-mean_b)), color='C0', label=f'Group B relevance')
        plt.savefig(f'figures/final_sims/varying_k/{self.RANK_POLICY}-{self.SELECT_POLICY}-relevance_{n}.pdf', dpi=300)

        # plot the change in mean over time
        plt.cla()
        plt.plot(np.arange(n), mean_a, color='C2', label=f'Group A mean')
        plt.plot(np.arange(n), mean_b, color='C0', label=f'Group B mean')
        plt.savefig(f'figures/final_sims/varying_k/{self.RANK_POLICY}-{self.SELECT_POLICY}-mean_{n}.pdf', dpi=300)
