import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style='darkgrid')


def prepare_data():
    # load
    df = pd.read_csv('compas-scores-two-years.csv', low_memory=False)

    # filter
    df = df[(df['days_b_screening_arrest'] >= -30) &
            (df['days_b_screening_arrest'] <= 30) &
            (df['is_recid'] != -1) &
            (df['v_score_text'] != 'N/A') &
            (df['c_charge_degree'] != 'O')]

    df = df[['race', 'decile_score', 'score_text', 'is_recid']]
    df = df[(df['race'] == 'Caucasian') | (df['race'] == 'African-American')]
    return df


# It seems that propublica decided to lock up everyone who had a 'High' as their text and let everyone with 'Low' go
# Not sure if this is the right way to think about FPR, FNR in our case but we can determine that ourselves I think
# 1-4 is Low, 8-10 is High


def plot_deciles(d, iter):
    # Do we want to normalize the counts to make this a true distribution?
    plt.subplots(2, 1, figsize=(5, 7))
    plt.subplots_adjust(hspace=0.5)
    plt.subplot(2, 1, 1)
    d1 = d[d['race'] == 'African-American']
    labels, counts = np.unique(d1['decile_score'].values, return_counts=True)
    afam = plt.bar(labels, counts, align='center', color='C2')
    plt.xticks(np.arange(10) + 1)
    plt.xlabel('Decile Score')
    plt.title('African-American')

    plt.subplot(2, 1, 2)
    d = d[d['race'] == 'Caucasian']
    labels, counts = np.unique(d['decile_score'].values, return_counts=True)
    white = plt.bar(labels, counts, align='center', color='C0')
    plt.xticks(np.arange(10) + 1)
    plt.xlabel('Decile Score')
    plt.title('Caucasian')

    plt.suptitle(f'Score Distribution, iteration {iter}')
    plt.savefig(f'figures/decile_dist_{iter}.png', dpi=72)
    plt.close()


def plot_rates(FPR_AA, FPR_C, FNR_AA, FNR_C, NUM_ITER):
    plt.plot(np.arange(NUM_ITER + 1), FPR_AA, color='C2', label='African-American FPR')
    plt.plot(np.arange(NUM_ITER + 1), FNR_AA, color='C0', label='African-American FNR')
    plt.plot(np.arange(NUM_ITER + 1), FPR_C, color='C2', ls='--', label='Caucasian FPR')
    plt.plot(np.arange(NUM_ITER + 1), FNR_C, color='C0', ls='--', label='Caucasian, FNR')
    plt.legend()
    plt.savefig('figures/rates', dpi=72)
    plt.close()


def get_rates(d, s):
    d = d[d['race'] == s]
    print(d.groupby('decile_score').mean())


def rates(d):
    d = d.assign(tp=(d['score_text'] == 'High') & (d['is_recid'] == 1),
                 fp=(d['score_text'] == 'High') & (d['is_recid'] == 0),
                 tn=(d['score_text'] == 'Low') & (d['is_recid'] == 0),
                 fn=(d['score_text'] == 'Low') & (d['is_recid'] == 1))
    sums = d.groupby('race').sum()
    sums = sums.assign(false_pos=lambda x: x.fp / (x.fp + x.tn),
                       false_neg=lambda x: x.fn / (x.fn + x.tp))
    return sums


def run_iteration(d):
    # order -- African-American, Caucasian
    FPR = rates(d, 'FPR')['false_pos'].values
    FNR = rates(d, 'FNR')['false_neg'].values
    prob_AA = stats.multinomial(1, [FPR[0], FNR[0], 1 - FPR[0] - FNR[0]])
    prob_C = stats.multinomial(1, [FPR[1], FNR[1], 1 - FPR[1] - FNR[1]])
    
    for i, row in d.iterrows():
        if row[0] == 'African-American':
            step = np.where(prob_AA.rvs()[0] == 1)[0][0]
        else: 
            step = np.where(prob_C.rvs()[0] == 1)[0][0]

        if step == 0 and row[1] != 1:
            d.at[i, 'decile_score'] = row[1] - 1
        elif step == 1 and row[1] != 10:
            d.at[i, 'decile_score'] = row[1] + 1

    return d


def main():
    df = prepare_data()
    FPR_AA, FPR_C = [], []
    FNR_AA, FNR_C = [], [] 
    NUM_ITER = 5

    # plot distribution without any modifications
    plot_deciles(df, 'PRE')
    FPR_AA.append(rates(df)['false_pos'].values[0])
    FPR_C.append(rates(df)['false_pos'].values[1])
    FNR_AA.append(rates(df)['false_neg'].values[0])
    FNR_C.append(rates(df)['false_neg'].values[1])
    
    # run iterations
    for i in tqdm(range(NUM_ITER)):
        df = run_iteration(df)
        plot_deciles(df, i + 1)

        # this needs to be fixed!
        FPR_AA.append(rates(df, 'FPR')['false_pos'].values[0])
        FPR_C.append(rates(df, 'FPR')['false_pos'].values[1])
        FNR_AA.append(rates(df, 'FNR')['false_neg'].values[0])
        FNR_C.append(rates(df, 'FNR')['false_neg'].values[1])

    plot_rates(FPR_AA, FPR_C, FNR_AA, FNR_C, NUM_ITER)


if __name__ == '__main__':
    main()
