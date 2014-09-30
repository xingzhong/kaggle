import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from math import exp, log, sqrt
import scipy as sp
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
from random import sample

def llfun(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

if __name__ == '__main__':
	
	data = pd.read_csv("train.csv", index_col=0, engine='c', nrows=200000)
	x = data.iloc[:, 1:14].fillna(0)
	y = data.Label
	vec = DictVectorizer()
	xc = data.iloc[:, 14:].fillna(-1)
	
	vec.fit(xc.T.to_dict().values())
	#import ipdb; ipdb.set_trace()
	N = len(xc)
	M = 2000
	svm = linear_model.SGDClassifier(class_weight='auto', 
			penalty='l2', n_jobs=-1, warm_start=True)
	for i in range(N/M):
		print i
		xs = x.iloc[i:i+M, :]
		ys = y[i:i+M]
		vecX = vec.transform(xc.iloc[i:i+M, :].T.to_dict().values()).toarray()
		X = np.hstack((xs.values, vecX))
		svm.fit(X, ys.values)
	
	rindex =  np.array(sample(xrange(len(data)), 10000))
	test = data.iloc[rindex]
	#import ipdb; ipdb.set_trace()

	testxi = test.iloc[:, 1:14].fillna(0)
	testxc = test.iloc[:, 14:].fillna(-1)
	vecX =  vec.transform(testxc.T.to_dict().values()).toarray()
	X = np.hstack((testxi.values, vecX))
	decision = svm.predict(X)
	test['pred'] = decision
	test['df'] = svm.decision_function(X)
	score = llfun(decision, test.Label.values)
	print test[test.Label==1][['Label', 'pred', 'df']]
	print score
	import ipdb; ipdb.set_trace()
	
