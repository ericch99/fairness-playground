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
    NEG_PEN = [0, -0.25, -0.5, -0.75, -1]
    POS_PEN = MEANS_A
    RANKINGS = ['top-k', 'max-util']
    SELECTIONS = ['top-k', 'stochastic']
    matrix = np.zeros(shape=(len(POS_PEN), len(NEG_PEN)))
    NUM_ITER = 25

    for rank in RANKINGS:
        for sel in SELECTIONS:
            for ma in MEANS_A:
                # run simulation once for each combination of r and c ==================================
                for r in range(len(POS_PEN)):
                    for c in range(len(NEG_PEN)):
                        A = {'mean': ma, 'var': 1.0, 'prob': 0.5}
                        B = {'mean': 0, 'var': 1.0, 'prob': 0.5}
                        sim = {'metric': 'avg-position',
                               'dist': 'logit-normal',
                               'k': 20,
                               'r_policy': rank,
                               's_policy': sel,
                               'query_len': 20}
                        penalty = {'pos': POS_PEN[r], 'neg': NEG_PEN[c], 'non': 0}

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
                        plt.savefig(f'figures/heatmaps/fullsim/{rank}-{sel}/pens:{POS_PEN[r]},{NEG_PEN[c]}-rel:{1 / (1 + np.exp(-ma))}.pdf', dpi=300)

                        # plotting means with given penalties
                        plt.cla()
                        plt.plot(np.arange(NUM_ITER), mean_a, color='C2', label=f'Group A mean')
                        plt.plot(np.arange(NUM_ITER), mean_b, color='C0', label=f'Group B mean')
                        plt.legend()
                        plt.ylabel('mean')
                        plt.xlabel('iteration')
                        plt.savefig(f'figures/heatmaps/fullsim/{rank}-{sel}/pens:{POS_PEN[r]},{NEG_PEN[c]}-mean:{ma}.pdf', dpi=300)

                # plot heatmap of difference in means ==================================================
                plt.cla()
                sns.heatmap(matrix, cmap="Blues")
                plt.xticks(np.arange(0, 5, 1), labels=POS_PEN)
                plt.yticks(np.arange(0, 5, 1), labels=NEG_PEN)
                plt.ylabel('negative feedback penalty')
                plt.xlabel('positive feedback penalty')
                plt.title('Final Differences in Mean')
                plt.savefig(f'figures/heatmaps/fullsim/{rank}-{sel}/mean_differences.pdf', dpi=300)


if __name__ == '__main__':
    main()
