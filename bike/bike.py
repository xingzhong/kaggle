import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn import tree
from sklearn import cross_validation
from sklearn.externals.six import StringIO 
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
import pydot 

def test(clf1, clf2):
	df = pd.read_csv("test.csv", index_col=0, parse_dates=True)
	df['hour'] = (df.index.hour - 5)%24
	df['month'] = df.index.month
	df['weekday'] = df.index.weekday
	x = df[['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'hour', 'month', 'weekday']].values
	df['casual'] = np.exp(clf1.predict(x))-1
	df["registered"] = np.exp(clf2.predict(x))-1
	df['count'] = (df.casual+df.registered).astype(int)
	df['count'].to_csv("sub.csv", header=True)
	#import ipdb; ipdb.set_trace()

def score(pred, act):
	return np.sqrt(mean_squared_error(np.log(pred+1), np.log(act+1)))


def train():
	df = pd.read_csv("train.csv", index_col=0, parse_dates=True)
	df['hour'] = (df.index.hour - 5)%24
	df['month'] = df.index.month
	df['weekday'] = df.index.weekday
	import ipdb; ipdb.set_trace()
	#y = (df['count']>=10).values
	y1 = np.log(df['casual'].values+1)
	y2 = np.log(df['registered'].values+1)
	x = df[['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'hour', 'month', 'weekday']].values
	tuned_parameters = [{'splitter':["best", "random"], 
				"max_depth":[4,5,6,7,8,9,10,11,12,13,14, 15,16,None], 
				#"max_depth":[2], 
				"max_features":[None],
				"min_samples_split":[2, 20, 40, 80, 100, 150, 200, 300]}]
	cv = cross_validation.ShuffleSplit(len(df), n_iter=10, test_size=0.3, random_state=0)
	score_func = make_scorer(mean_squared_error, greater_is_better=False)

	clf1 = GridSearchCV(tree.DecisionTreeRegressor(), tuned_parameters, scoring=score_func, cv=10, n_jobs=8)
	#clf1 = GridSearchCV(tree.ExtraTreeRegressor(), tuned_parameters, cv=10)
	clf1.fit(x, y1)
	#for params, mean_score, scores in clf1.grid_scores_:
	#	print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params))
	best1 = clf1.best_estimator_
	print best1
	#scores = cross_validation.cross_val_score(best1, x, y1, cv=cv, score_func=score)
	#scores = np.array(scores)
	#print("%0.3f (+/-%0.03f)"%(np.mean(scores), scores.std()))
	#import ipdb; ipdb.set_trace()
	df["c1"] = (np.exp(best1.predict(x))-1).astype(int)

	clf2 = GridSearchCV(tree.DecisionTreeRegressor(), tuned_parameters, scoring=score_func, cv=10, n_jobs=8)
	#clf2 = GridSearchCV(tree.ExtraTreeRegressor(), tuned_parameters, cv=10)
	clf2.fit(x, y2)
	#for params, mean_score, scores in clf2.grid_scores_:
	#	print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params))
	best2 = clf2.best_estimator_
	print best2
	#scores = cross_validation.cross_val_score(best2, x, y2, cv=cv, score_func=score)
	#scores = np.array(scores)
	#print("%0.3f (+/-%0.03f)"%(np.mean(scores), scores.std()))
	df["r1"] = (np.exp(best2.predict(x))-1).astype(int)
	df['est'] = df.c1 + df.r1
	df['err1'] = np.sqrt( (np.log(df.casual+1) - np.log(df.c1+1))**2 )
	df['err2'] = np.sqrt( (np.log(df.registered+1) - np.log(df.r1+1))**2 )
	df['err'] = np.sqrt( (np.log(df['count']+1) - np.log(df.est+1))**2 )
	res = df[['casual', 'c1', 'err1', 'registered', 'r1', 'err2', 'count', 'est', 'err']]
	
	print score(df.casual.values, df.c1.values)
	print score(df.registered.values, df.r1.values)
	print score(df['count'].values, df.est.values)
	#res.plot(); plt.show()
	vis(best1, 'best1')
	vis(best2, 'best2')
	import ipdb; ipdb.set_trace()

	return best1, best2
def vis(clf, name):
	dot_data = StringIO() 
	tree.export_graphviz(clf, max_depth=4, out_file=dot_data) 
	graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
	graph.write_pdf("%s.pdf"%name) 

def distance(x, y):
	return ((x-y)**2 * np.array([1, 10, 10, 5, 2, 4, 4, 4, 4])).sum()

def explore():
	df = pd.read_csv("train.csv", index_col=0, parse_dates=True)
	df['hour'] = (df.index.hour - 5)%24
	x = df[['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'hour']].values
	df['lowc'] = df.casual < 20
	df['lowr'] = df.registered < 100
	#df = df.astype(int)
	clf = tree.DecisionTreeClassifier()
	clf.fit(x, df.lowc.values)
	import ipdb; ipdb.set_trace()

def time():
	df = pd.read_csv("train.csv", index_col=0, parse_dates=True)
	grps = df.groupby(df.index.year*100+df.index.month)
	v = DictVectorizer(sparse=False)
	df['weather'] = df['weather'].apply(str)
	y, y1, y2 = df['count'], df.casual, df.registered
	del df['count'], df["casual"], df['registered']
	X = v.fit_transform(df.T.to_dict().values())
	import ipdb; ipdb.set_trace()

if __name__ == '__main__':
	#import ipdb; ipdb.set_trace()
	#clf1, clf2 = train()
	#test(clf1, clf2)
	#explore()
	time()