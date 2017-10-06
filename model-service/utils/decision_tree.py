from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import pickle

def preprocess_data(filepath):
	df = pd.read_csv(filepath)
	# transform y to [0,1]
	y = df.y.values
	le = LabelEncoder()
	y = le.fit_transform(y)
	# preprocess X
	features = list(df.columns)
	del features[-1]
	del features[0]
	X = df[features]
	X = pd.get_dummies(X, columns=['job', 'marital', 'education', 'default', 'housing' ,'loan', 'contact', 'month', 'day_of_week', 'poutcome'])
	return X, y

# def my_train_test_split(X, y):
# 	# train-test split
# 	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
# 	return X_train, X_test, y_train, y_test

def hyper_parameter_search(X_train, y_train):
	# hyper-parameter search
	pipe_dt = make_pipeline(DecisionTreeClassifier(random_state=1))
	param_grid_dt = [{'decisiontreeclassifier__max_depth': [1, 2, 3, 4, 5, 6, 7, 10, None], 'decisiontreeclassifier__min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 10]}]
	gs_dt = GridSearchCV(estimator=pipe_dt,
	                  param_grid=param_grid_dt,
	                  scoring='roc_auc',
	                  cv=10,
	                  refit=True)
	gs_dt = gs_dt.fit(X_train, y_train)
	print(gs_dt.best_score_)
	print(gs_dt.best_params_)
	return gs_dt

def get_new_feature(gs_dt, X):
	# select important feature for X
	dt = gs_dt.best_estimator_.named_steps['decisiontreeclassifier'].feature_importances_
	feature_df = pd.DataFrame(list(zip(X.columns, np.transpose(dt))), columns=['feature', 'importance']).sort_values('importance', axis=0, ascending=False) 
	new_feature = list(feature_df.iloc[:15, 0].values)
	print(new_feature)
	X = X[new_feature]
	return X

def evaluate_model(best_dt, X_train, X_test, y_train, y_test):
	y_pred_dt_test = best_dt.predict_proba(X_test)[:,1]
	y_pred_dt_train = best_dt.predict_proba(X_train)[:,1]

	print("Training Auc Score")
	print("Decision Tree Train Auc Score: %.3f" % (roc_auc_score(y_true=y_train, y_score=y_pred_dt_train)))

	print("Testing Auc Score")
	print("Decision Tree Test Auc Score: %.3f" % (roc_auc_score(y_true=y_test, y_score=y_pred_dt_test)))	
	# get y probability
	y_proba = best_dt.predict_proba(X_test)[:, 1]
	print('AUC Score: %.3f' % roc_auc_score(y_true=y_test, y_score=y_proba))

def fit_and_save_model(best_dt, X, y, modelname):
	# fit model with entire dataset
	final_model = best_dt.fit(X, y)
	print(final_model)
	pickle.dump(final_model, open(modelname, 'wb'))

def train_model(filepath, modelname):
	print("Preprocess Data")
	X, y = preprocess_data(filepath)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
	print("Hyper Parameter Search")
	gs_dt = hyper_parameter_search(X_train, y_train)
	X = get_new_feature(gs_dt, X)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
	gs_dt = hyper_parameter_search(X_train, y_train)
	best_dt = gs_dt.best_estimator_
	print("---------------Evaluate Model----------------------")
	evaluate_model(best_dt, X_train, X_test, y_train, y_test)
	fit_and_save_model(best_dt, X, y, modelname)

# if __name__ == '__main__':
# 	print("Training Model")
# 	result = {"result": "model trained"}
# 	modelname = './t1.sav'
# 	train_model('/home/thanathip/platform/setup-hadoop/training.csv', modelname)