import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import svm

def learning(data):
    # data is a dataframe
    # known ground truth at loss and default cols
    
    # shuffle
    temp = data.reindex(np.random.permutation(data.index))
    train = temp.head(int(len(temp)*0.7))
    evals = temp.tail(int(len(temp)*0.5))
    # train first svm model to classify default
    svm_model = svm.SVC(class_weight='auto', verbose=1)
    #svm_model = svm.LinearSVC(verbose=1)
    x = train.iloc[:, :778].values
    y = train.default.values
    print "there are %s default in train data"%train.default.sum()
    svm_model.fit(x, y)
    # train second linear model to do regression on default loan
    train_default = train[train.default]
    x = train_default.iloc[:, :778].values
    y = train_default.loss.values
    lrg_model = linear_model.LinearRegression()
    lrg_model.fit(x, y)
    
    # test on evals
    # test default ?
    x = evals.iloc[:, :778].values
    evals['pred_default'] = svm_model.predict(x)
    default_error = (evals.pred_default - evals.default).abs().sum()
    print "there are %s /%s default in evals data"%(evals.pred_default.sum(), evals.default.sum())
    print "default_error (%s / %s)"%(default_error, len(x))
    # regression on lost
    evals['pred_lost'] = lrg_model.predict(x)
    
    
    # combine two model
    evals['pred_final'] = evals.pred_default * evals.pred_lost
    loss_error = (evals.pred_final - evals.loss).abs().mean()
    print "loss error %s"%loss_error
    
    return svm_model, lrg_model, loss_error, evals

def test_model(data, svm_model, lrg_model):
    x = data.values
    data['pred_default'] = svm_model.predict(x)
    print "there are %s defaults predict from test data"%data.pred_default.sum()
    data['loss'] = lrg_model.predict(x) * data.pred_default
    return data

def main():
	trainFile = "./train.csv"
	testFile = "./test.csv"

	train = pd.read_csv(trainFile, dtype=float, index_col=0)
	test = pd.read_csv(testFile, dtype=float, index_col=0)
	train.index = train.index.astype(int)
	test.index = test.index.astype(int)
	print "finish loading data"

	train_sample = train.head(5000).copy()
	train_sample = train_sample.fillna(train_sample.mean())
	train_sample['default'] = train_sample.loss > 0
	test_sample = test.copy()
	test_sample = test_sample.fillna(test_sample.mean())
	print "train..."
	svm_model, lrg_model, error, evals = learning(train_sample)
	test_result = test_model(test_sample, svm_model, lrg_model)
	print "evaluate..."
	
	test_result.loss = test_result.loss.astype(int)
	test_result.loss.to_csv('./submit.csv', header=True)
	print "bye"
if __name__ == '__main__':
	main()



