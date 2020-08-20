from simulation import *
from utils import *
from gen_input import *


def main():
    ranks = ['max-util', 'top-k']
    selects = ['top-k', 'stochastic']

    for rank in ranks:
        for select in selects:
            # PARAMETERS ===============================================
            A, B, sim, penalty = generate_input()
            # sim = {'metric': 'avg-position',
            #        'dist': 'logit-normal',
            #        'k': 10,
            #        'r_policy': rank,
            #        's_policy': select,
            #        'query_len': 20}
            sim['r_policy'] = rank
            sim['s_policy'] = select
            sim['k'] = 5
            num_iter = 25

            # RUN SIMULATION ===========================================
            s = Simulation(A, B, sim, penalty)

            print('PARAMETERS:\n')
            print([f'{k}: {v}' for k, v in s.__dict__.items()], '\n')

            print('RUNNING SIMULATION...\n')
            metric_a, mean_a, metric_b, mean_b = s.run_simulation(n=num_iter)

            # PLOT METRICS =============================================
            print('PLOTTING METRICS...')
            s.plot_metrics(num_iter, metric_a, mean_a, metric_b, mean_b)


if __name__ == '__main__':
    main()