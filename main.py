from simulation import *
from utils import *
from gen_input import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import trange


def plot_all(k, rank, select, n, big_a, big_b, big_a_1, big_b_1, big_a_2, big_b_2):
    mean_a, max_a, min_a = np.mean(big_a, axis=0), big_a.max(axis=0), big_a.min(axis=0)
    mean_a_1, max_a_1, min_a_1 = np.mean(big_a_1, axis=0), big_a_1.max(axis=0), big_a_1.min(axis=0)
    mean_a_2, max_a_2, min_a_2 = np.mean(big_a_2, axis=0), big_a_2.max(axis=0), big_a_2.min(axis=0)
    mean_b, max_b, min_b = np.mean(big_b, axis=0), big_b.max(axis=0), big_b.min(axis=0)
    mean_b_1, max_b_1, min_b_1 = np.mean(big_b_1, axis=0), big_b_1.max(axis=0), big_b_1.min(axis=0)
    mean_b_2, max_b_2, min_b_2 = np.mean(big_b_2, axis=0), big_b_2.max(axis=0), big_b_2.min(axis=0)
    sns.set(style='darkgrid')

    plt.cla()
    plt.errorbar(np.arange(n), mean_a, yerr=np.std(big_a, axis=0), fmt='-', color='C2', label='Group A mean (B = -1, A = 0)')
    plt.errorbar(np.arange(n), mean_a_1, yerr=np.std(big_a_1, axis=0), fmt='--', color='C2', label='Group A mean (B = -0.5, A = 0)')
    plt.errorbar(np.arange(n), mean_a_2, yerr=np.std(big_a_2, axis=0), fmt=':', color='C2', label='Group A mean (B = -0.1, A = 0)')
    plt.errorbar(np.arange(n), mean_b, yerr=np.std(big_b, axis=0), fmt='-', color='C0', label='Group B mean (B = -1, A = 0)')
    plt.errorbar(np.arange(n), mean_b_1, yerr=np.std(big_b_1, axis=0), fmt='--', color='C0', label='Group B mean (B = -0.5, A = 0)')
    plt.errorbar(np.arange(n), mean_b_2, yerr=np.std(big_b_2, axis=0), fmt=':', color='C0', label='Group B mean (B = -0.1, A = 0)')
    plt.legend()
    plt.ylim((-5, 10))
    plt.ylabel('Time')
    plt.xlabel('Means')
    plt.savefig(f'figures/final_sims/varying_k/{k}/{rank}-{select}-{n}-all.pdf', dpi=300)


def main():
    ranks = ['max-util', 'top-k']
    selects = ['top-k', 'stochastic']
    ks = [3, 5, 10]
    num_iter = 25
    num_experiments = 10
    big_a = np.zeros((int(num_experiments), int(num_iter)))
    big_a_1 = np.zeros((int(num_experiments), int(num_iter)))
    big_a_2 = np.zeros((int(num_experiments), int(num_iter)))
    big_b = np.zeros((int(num_experiments), int(num_iter)))
    big_b_1 = np.zeros((int(num_experiments), int(num_iter)))
    big_b_2 = np.zeros((int(num_experiments), int(num_iter)))
    # x label, y label, title, fix legend, error bars or bands

    # for k in ks:
    for rank in ranks:
        for select in selects:
            for i in trange(num_experiments, desc='experiments'):
                # PARAMETERS ===============================================
                A, B, sim, penalty = generate_input()
                sim['r_policy'] = rank
                sim['s_policy'] = select
                sim['k'] = 10
                B_1 = {'mean': -0.5, 'var': 1, 'prob': 0.5}
                B_2 = {'mean': -0.1, 'var': 1, 'prob': 0.5}

                # RUN SIMULATION ===========================================
                s = Simulation(A, B, sim, penalty)
                s_1 = Simulation(A, B_1, sim, penalty)
                s_2 = Simulation(A, B_2, sim, penalty)

                print('PARAMETERS:\n')
                print([f'{k}: {v}' for k, v in s.__dict__.items()], '\n')

                print('RUNNING SIMULATION...\n')
                metric_a, mean_a, metric_b, mean_b = s.run_simulation(n=num_iter)

                print('PARAMETERS:\n')
                print([f'{k}: {v}' for k, v in s_1.__dict__.items()], '\n')

                print('RUNNING SIMULATION...\n')
                metric_a_1, mean_a_1, metric_b_1, mean_b_1 = s_1.run_simulation(n=num_iter)

                print('PARAMETERS:\n')
                print([f'{k}: {v}' for k, v in s_2.__dict__.items()], '\n')

                print('RUNNING SIMULATION...\n')
                metric_a_2, mean_a_2, metric_b_2, mean_b_2 = s_2.run_simulation(n=num_iter)

                # PLOT METRICS =============================================
                print('PLOTTING METRICS...')
                # s.plot_metrics(num_iter, metric_a, mean_a, metric_b, mean_b)
                #
                # print('PLOTTING METRICS...')
                # s_1.plot_metrics(num_iter, metric_a, mean_a, metric_b, mean_b)
                #
                # print('PLOTTING METRICS...')
                # s_2.plot_metrics(num_iter, metric_a, mean_a, metric_b, mean_b)
                big_a[i] = mean_a
                big_a_1[i] = mean_a_1
                big_a_2[i] = mean_a_2
                big_b[i] = mean_b
                big_b_1[i] = mean_b_1
                big_b_2[i] = mean_b_2

            plot_all(10, rank, select, num_iter, big_a, big_b, big_a_1, big_b_1, big_a_2, big_b_2)


if __name__ == '__main__':
    main()