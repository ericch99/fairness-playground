import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
from distributions import *
from metrics import *
from ranking_policies import *
from selection_policies import *
from gen_input import generate_input
from simulation import Simulation

sns.set(style='darkgrid')


def main():
    # initialize constants =================================================================
    # NEUTRAL = 0
    MEANS_A = [0, 0.25, 0.5, 0.75, 1]
    MEANS_B = [0, -0.25, -0.5, -0.75, -1]
    RANKINGS = ['top-k', 'max-util']
    SELECTIONS = ['top-k', 'stochastic']
    matrix = np.zeros(shape=(len(MEANS_A), len(MEANS_B)))
    NUM_ITER = 25

    for rank in RANKINGS:
        for sel in SELECTIONS:
            # run simulation once for each combination of r and c ==================================
            for r in range(len(MEANS_A)):
                for c in range(len(MEANS_B)):
                    A, B, sim, penalty = generate_input()
                    A = {'mean': MEANS_A[r], 'var': 1.0, 'prob': 0.5}
                    B = {'mean': MEANS_B[c], 'var': 1.0, 'prob': 0.5}
                    sim = {'metric': 'avg-position',
    	                   'dist': 'logit-normal',
    	                   'k': 10,
                           'r_policy': rank,
                           's_policy': sel,
                           'query_len': 20}

                    s = Simulation(A, B, sim, penalty)

                    print('PARAMETERS:\n')
                    print([f'{k}: {v}' for k, v in s.__dict__.items()], '\n')

                    print('RUNNING SIMULATION...\n')
                    _, mean_a, _, mean_b = s.run_simulation(n=NUM_ITER)
                    matrix[r][c] = s.MEAN_A - s.MEAN_B

                    # plotting relevances with given penalties
                    plt.cla()
                    plt.plot(np.arange(NUM_ITER), 1 / (1 + np.exp(-mean_a)), color='C2', label=f'Group A relevance')
                    plt.plot(np.arange(NUM_ITER), 1 / (1 + np.exp(-mean_b)), color='C0', label=f'Group B relevance')
                    plt.legend()
                    plt.ylabel('relevance')
                    plt.xlabel('iteration')
                    plt.savefig(f'figures/heatmaps/fullsim/{rank}-{sel}/relevances:{1 / (1 + np.exp(-MEANS_A[r]))},{1 / (1 + np.exp(-MEANS_B[c]))}.pdf', dpi=300)

                    # plotting means with given penalties
                    plt.cla()
                    plt.plot(np.arange(NUM_ITER), mean_a, color='C2', label=f'Group A mean')
                    plt.plot(np.arange(NUM_ITER), mean_b, color='C0', label=f'Group B mean')
                    plt.legend()
                    plt.ylabel('mean')
                    plt.xlabel('iteration')
                    plt.savefig(f'figures/heatmaps/fullsim/{rank}-{sel}/means:{MEANS_A[r]},{MEANS_B[c]}.pdf', dpi=300)

            # plot heatmap of difference in means ==================================================
            plt.cla()
            sns.heatmap(matrix, cmap="Blues")
            plt.xticks(np.arange(0, 5, 1), labels=MEANS_A)
            plt.yticks(np.arange(0, 5, 1), labels=MEANS_B)
            plt.ylabel('group b means')
            plt.xlabel('group a means')
            plt.title('Final Differences in Mean')
            plt.savefig(f'figures/heatmaps/fullsim/{rank}-{sel}/mean_differences.pdf', dpi=300)


if __name__ == '__main__':
    main()
