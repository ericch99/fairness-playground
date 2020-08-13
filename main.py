from simulation import *
from utils import *
from gen_input import *


def main():
    # PARAMETERS ===============================================
    A, B, sim, penalty = generate_input()
    num_iter = 10

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