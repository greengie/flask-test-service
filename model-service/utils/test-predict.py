import numpy as np
import pandas as pd
import pickle

msg_body = {
    "euribor3m": 4.857,
    "age": 38,
    "nr_employed": 5191,
    "pdays": 999,
    "day_of_week": "tue",
    "emp_var_rate": 1.1,
    "cons_conf_idx": -36.4,
    "poutcome": "nonexistent",
    "education": "high.school",
    "cons_price_idx": 93.994,
    "contact": "telephone",
    "month": "may",
    "campaign": 2
}
X_test = pd.DataFrame.from_dict(msg_body, orient='index').T
# print(X_test.T)
# print(X_test)
feature_list = ['nr_employed', 'pdays', 'euribor3m', 'cons_conf_idx', 'age', 'month_oct', 'contact_telephone', 'campaign', 'poutcome_failure', 'cons_price_idx', 'emp_var_rate', 'day_of_week_mon', 'education_university.degree', 'month_apr', 'day_of_week_wed']
X = pd.DataFrame()
for feature in feature_list:
	try:
		X[feature] = X_test[feature]
	except:
		X[feature] = 0
# print(X)
filename = './t1.sav'
model = pickle.load(open(filename, 'rb'))
pred = model.predict(X)
print(pred)

