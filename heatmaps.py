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
    matrix = np.zeros(shape=(len(MEANS_A), len(MEANS_B)))
    NUM_ITER = 25

    # run simulation once for each combination of r and c ==================================
    for r in range(len(MEANS_A)):
        for c in range(len(MEANS_B)):
            A, B, sim, penalty = generate_input()
            A = {'mean': MEANS_A[r], 'var': 1.0, 'prob': 0.7}
            B = {'mean': MEANS_B[c], 'var': 1.0, 'prob': 0.3}

            s = Simulation(A, B, sim, penalty)
            
            print('PARAMETERS:\n')
            print([f'{k}: {v}' for k, v in s.__dict__.items()], '\n')

            print('RUNNING SIMULATION...\n')
            _, mean_a, _, mean_b = s.run_simulation(n=NUM_ITER)
            matrix[r][c] = s.MEAN_A - s.MEAN_B

            # plotting metrics with given penalties
            plt.cla()
            plt.plot(np.arange(NUM_ITER), mean_a, color='C2', label=f'Group A mean')
            plt.plot(np.arange(NUM_ITER), mean_b, color='C0', label=f'Group B mean')
            plt.ylabel('mean')
            plt.xlabel('iteration')
            plt.savefig(f'figures/heatmaps/means:{MEANS_A[r]},{MEANS_B[c]}.pdf', dpi=300)

    # plot heatmap of difference in means ================================================== 
    plt.cla()
    sns.heatmap(matrix, cmap="Blues")
    plt.xticks(np.arange(0, 5, 1), labels=MEANS_A)
    plt.yticks(np.arange(0, 5, 1), labels=MEANS_B)
    plt.ylabel('group b means')
    plt.xlabel('group a means')
    plt.savefig(f'figures/heatmaps/mean_differences.pdf', dpi=300)


if __name__ == '__main__':
    main()
