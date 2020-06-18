import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# import the datasets
df = pd.read_csv('../compas-analysis/compas-scores-two-years-violent.csv', low_memory=False)

# filer to replicate propublica
df = df[['name',
         'sex',
         'age',
         'age_cat',
         'race',
         'juv_fel_count',
         'juv_misd_count',
         'juv_other_count',
         'priors_count',
         'days_b_screening_arrest',
         'c_jail_in',
         'c_jail_out',
         'c_days_from_compas',
         'r_charge_desc',
         'r_jail_in',
         'r_jail_out',
         'is_recid',
         'is_violent_recid',
         'vr_charge_desc',
         'decile_score',
         'v_decile_score',
         'score_text',
         'v_score_text',
         'c_charge_degree']]

df = df[(df['days_b_screening_arrest'] >= -30) &
        (df['days_b_screening_arrest'] <= 30) &
        (df['is_recid'] != -1) &
        (df['v_score_text'] != 'N/A') &
        (df['c_charge_degree'] != 'O')]

df = df[['sex',
         'age',
         'race',
         'juv_fel_count',
         'juv_misd_count',
         'juv_other_count',
         'priors_count',
         'is_violent_recid']]

df = df.assign(sex_b=df['sex'] == 'Male')
df = df.assign(race_b=df['race'] == 'African-American')
df = df.drop(columns=['sex', 'race'])
# df = df[df['race_b'] == 0]

# give the model variables and see if it predicts properly
x = df.drop('is_violent_recid', axis=1)
y = np.ravel(df['is_violent_recid'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=13)

scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# svc_model = SVC()
# svc_model.fit(x_train, y_train)
# y_predict = svc_model.predict(x_test)
clf = LogisticRegression(random_state=17).fit(x_train, y_train)
y_predict = clf.predict_proba(x_test)

comp = {'actual': y_test,
        'predicted': y_predict[:, 1]}

comper = pd.DataFrame(comp, columns=['actual', 'predicted'])

print(comper)

# cm = np.array(confusion_matrix(y_test, y_predict, labels=[0,1]))
# confusion = pd.DataFrame(cm, index=['not recid', 'recid'], columns=['predicted not recid', 'predicted recid'])
#
# print(confusion)
# print(classification_report(y_test, y_predict))

# print(df)
