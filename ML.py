from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import os
import pandas as pd
import pickle as pickle
import numpy as np
import time

TRAIN_PERCENTAGE=0.7

algo_dict={'rf_cla':RandomForestClassifier,
	 'svm_cla':SVC,
	 'dt_cla':DecisionTreeClassifier,
	 'rf_reg':RandomForestRegressor,
	 'svm_reg':SVR,
	 'dt_reg':DecisionTreeRegressor}

def process(param_dict, root_path, job_name):
	data_path=root_path+'/tmp/data_'+job_name
	model_path=root_path+'/tmp/'+job_name+'.pkl'
	if len(param_dict['header_row'])==0:
		header_row=None
	else:
		header_row=int(param_dict['header_row'])
	df=pd.read_table(data_path, sep=None, header=header_row, engine='python')
	df = df.sample(frac=1).reset_index(drop=True)
	NO_TRAIN=int(TRAIN_PERCENTAGE*df.shape[0])
	NO_TEST=int(df.shape[0]-NO_TRAIN)
	algo_name=param_dict['algorithm']+'_'+param_dict['train_type']
	print('model name', algo_name)
	train=df.head(NO_TRAIN)
	test=df.tail(NO_TEST)
	y=df[int(param_dict['y_col'])]
	y_train=train[int(param_dict['y_col'])]
	y_test=test[int(param_dict['y_col'])]
	y_map={}
	if param_dict['train_type']=='cla':
		for ix, y_val in enumerate(y.unique()):
			y_map[y_val]=ix
		y_train=y_train.apply(lambda row: y_map[row])
		y_test=y_test.apply(lambda row: y_map[row])
	# y_train=df[int(param_dict['y_col'])].head(NO_TRAIN)
	# y_test=df[int(param_dict['y_col'])].tail(NO_TEST)
	# X=df.drop(int(param_dict['y_col']), axis=1)
	X_train=train.drop(int(param_dict['y_col']), axis=1)#X.head(NO_TRAIN)
	X_test=test.drop(int(param_dict['y_col']), axis=1)

	model=algo_dict[algo_name]()
	model.fit(X_train, y_train)
	predictions=model.predict(X_test)
	metric_value=-1
	if param_dict['train_type']=='cla':
		acc=accuracy_score(y_test, predictions)
		metric_value=acc
	else:
		rmse=np.sqrt(mean_squared_error(y_test, predictions))
		metric_value=rmse
	with open(model_path, 'wb') as f:
		pickle.dump(model, f)
	return root_path, metric_value

def make_predictions(model_path, data_path, header):
	X=pd.read_table(data_path, sep=None, header=header, engine='python')
	model=pickle.load(open(model_path, 'rb'))
	pred=model.predict(X)
	return pred
