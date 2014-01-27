import pandas as pd
import numpy as np
from sklearn import linear_model, svm, preprocessing, decomposition
import matplotlib.pyplot as plt

def learning(data):
    # data is a dataframe
    # known ground truth at loss and default cols
    scaler = preprocessing.MinMaxScaler()
    decomp = decomposition.PCA(n_components=150)
    # shuffle
    temp = data.reindex(np.random.permutation(data.index))
    train = temp.head(int(len(temp)*0.7))
    evals = temp.tail(int(len(temp)*0.5))
    # train first svm model to classify default
    gt_true = train.default.sum()
    gt_false = train.default.count() - gt_true
    default_model = svm.SVC(
        C = 1.0,
        class_weight={True:gt_false, False:gt_true},
        cache_size=500, 
        verbose=1)
    #default_model = svm.SVC(class_weight='auto', verbose=1)
    #svm_model = svm.LinearSVC(verbose=1)
    x = train.iloc[:, :778].values
    x_scaled = scaler.fit_transform(x)
    x_lowD = decomp.fit_transform(x_scaled)
    y = train.default.values
    weight = (train.loss/50.0+1).values

    print "there are %s default out of %s in train data"%(train.default.sum(), train.default.count())
    #import pdb; pdb.set_trace()
    #default_model.fit(x_lowD, y)
    default_model.fit(x_lowD, y, sample_weight=weight)
    # train second linear model to do regression on default loan
    train_default = train[train.default]
    x = train_default.iloc[:, :778].values
    x_scaled = scaler.transform(x)
    y = train_default.loss.values
    lrg_model = linear_model.LinearRegression()
    lrg_model.fit(x_scaled, y)
    
    # test on evals
    # test default ?
    x = evals.iloc[:, :778].values
    x_scaled = scaler.transform(x)
    x_lowD = decomp.fit_transform(x_scaled)
    evals['pred_default'] = default_model.predict(x_lowD)
    default_error = (evals.pred_default - evals.default).abs().sum()
    print "############"
    print "total cases %s"%len(evals)
    print "total default %s"%evals.default.sum()
    print "false default %s"%len(evals[(evals.pred_default==True) & (evals.default==False)])
    print "miss default %s"%len(evals[(evals.pred_default==False) & (evals.default==True)])
    print "positive default %s"%len(evals[(evals.pred_default==True) & (evals.default==True)])
    print "negitive default %s"%len(evals[(evals.pred_default==False) & (evals.default==False)])
    # regression on lost
    evals['pred_lost'] = lrg_model.predict(x_scaled)
    
    
    # combine two model
    evals['pred_final'] = evals.pred_default * evals.pred_lost
    loss_error = (evals.pred_final - evals.loss).abs().mean()
    print "loss error %s"%loss_error
    
    return default_model, lrg_model, scaler, decomp, loss_error, evals

def test_model(data, default_model, lrg_model, scaler, decomp):
    x = data.values
    x_scaled = scaler.transform(x)
    x_lowD = decomp.fit_transform(x_scaled)
    data['pred_default'] = default_model.predict(x_lowD)
    print "there are %s defaults predict from test data"%data.pred_default.sum()
    data['loss'] = lrg_model.predict(x) * data.pred_default
    return data

def main():
	trainFile = "./train.tiny"
	testFile = "./test.tiny"

	train = pd.read_csv(trainFile, dtype=float, index_col=0)
	test = pd.read_csv(testFile, dtype=float, index_col=0)
	train.index = train.index.astype(int)
	test.index = test.index.astype(int)
	print "finish loading data"

	train_sample = train.copy()
	train_sample = train_sample.fillna(train_sample.mean())
	train_sample['default'] = train_sample.loss > 0
	test_sample = test.copy()
	test_sample = test_sample.fillna(test_sample.mean())
	print "train..."
	svm_model, lrg_model, scaler, decomp, error, evals = learning(train_sample)
	test_result = test_model(test_sample, svm_model, lrg_model, scaler, decomp)
	print "evaluate..."
	
	test_result.loss = test_result.loss.astype(int)
	test_result.loss.to_csv('./submit.csv', header=True)
	print "bye"
if __name__ == '__main__':
	main()



