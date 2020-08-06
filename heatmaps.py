import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
from distributions import *
from metrics import *
from ranking_policies import *
from selection_policies import *

sns.set(style='darkgrid')


def main():
    # initialize constants =================================================================
    NEUTRAL = 0
    SUCCESSES = [0.25, 0.5, 0.75, 1]
    FAILS = [-0.25, -0.5, -0.75, -1]
    matrix = np.zeros(shape=(len(SUCCESSES), len(FAILS)))

    # run simulation once for each combination of r and c ==================================
    for r in SUCCESSES:
        for c in FAILS:
            A, B, sim, _ = generate_input()
            penalty = {'pos': r, 'neg': c, 'non': NEUTRAL}

            s = Simulation(A, B, sim, penalty)
            
            print('PARAMETERS:\n')
            print([f'{k}: {v}' for k, v in s.__dict__.items()], '\n')

            print('RUNNING SIMULATION...\n')
            _, mean_a, _, mean_b = s.run_simulation(n=10)
            matrix[r][c] = s.MEAN_A - s.MEAN_B

            # plotting metrics with given penalties
            plt.cla()
            plt.plot(np.arange(10), mean_a, color='C2', label=f'Group A mean')
            plt.plot(np.arange(10), mean_b, color='C0', label=f'Group B mean')
            plt.savefig(f'figures/heatmaps/means:{r},{c}.pdf', dpi=300)

    # plot heatmap of difference in means ================================================== 
    plt.cla()
    sns.heatmap(matrix)
    plt.savefig(f'figures/heatmaps/feedback_coefficients.pdf', dpi=300)


if __name__ == '__main__':
    main()
