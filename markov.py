import pandas as pd

# load
df = pd.read_csv('../compas-analysis/compas-scores-two-years.csv', low_memory=False)

# filter
df = df[(df['days_b_screening_arrest'] >= -30) &
        (df['days_b_screening_arrest'] <= 30) &
        (df['is_recid'] != -1) &
        (df['v_score_text'] != 'N/A') &
        (df['c_charge_degree'] != 'O')]

df = df[['race', 'decile_score', 'is_recid']]


def get_rates(d, s):
    d = d[d['race'] == s]
    print(d.groupby('decile_score').mean())


get_rates(df, 'African-American')
get_rates(df, 'Caucasian')

# df = df[(df['race'] == 'African-American') | (df['race'] == 'Caucasian')]
# df = df.assign(counter=1)
# df = df.groupby(['race', 'decile_score'])
# df = df.assign(percent=(df['is_recid'] / df['counter']))
#
# print(df)