import pandas as pd
# import matplotlib.pyplot as plt

# load
df = pd.read_csv('../compas-analysis/compas-scores-two-years.csv', low_memory=False)

# filter
df = df[(df['days_b_screening_arrest'] >= -30) &
        (df['days_b_screening_arrest'] <= 30) &
        (df['is_recid'] != -1) &
        (df['v_score_text'] != 'N/A') &
        (df['c_charge_degree'] != 'O')]

df = df[['race', 'decile_score', 'score_text', 'is_recid']]
df = df[(df['race'] == 'Caucasian') | (df['race'] == 'African-American')]

# It seems that propublica decided to lock up everyone who had a 'High' as their text and let everyone with 'Low' go
# Not sure if this is the right way to think about FPR, FNR in our case but we can determine that ourselves I think
# 1-4 is Low, 8-10 is High


def plot_deciles(d, s):
    # can clean up the plots a lot just a start
    d = d[d['race'] == s]
    d['decile_score'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0)


def get_rates(d, s):
    d = d[d['race'] == s]
    print(d.groupby('decile_score').mean())


def rates(d, s):
    if s == "TPR":
        d = d.assign(true_pos=(d['score_text'] == 'High') & (d['is_recid'] == 1))
        return d.groupby('race').mean()
    elif s == "FPR":
        d = d.assign(false_pos=(d['score_text'] == 'High') & (d['is_recid'] == 0))
        return d.groupby('race').mean()
    elif s == "TNR":
        d.assign(true_neg=(d['score_text'] == 'Low') & (d['is_recid'] == 0))
        return d.groupby('race').mean()
    else:
        d = d.assign(false_neg=(d['score_text'] == 'Low') & (d['is_recid'] == 1))
        return d.groupby('race').mean()


get_rates(df, 'African-American')
get_rates(df, 'Caucasian')

print(rates(df, 'FPR'))