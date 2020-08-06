from simulation import *
from utils import *
from gen_input import *


def main():
    # PARAMETERS ===============================================
    A, B, sim, penalty = generate_input()

    # RUN SIMULATION ===========================================
    s = Simulation(A, B, sim, penalty)

    print('PARAMETERS:\n')
    print([f'{k}: {v}' for k, v in s.__dict__.items()], '\n')

    print('RUNNING SIMULATION...\n')
    s.run_simulation(n=10)

    # PLOT METRICS =============================================
    print('PLOTTING METRICS...')
    s.plot_metrics()  


if __name__ == '__main__':
    main()