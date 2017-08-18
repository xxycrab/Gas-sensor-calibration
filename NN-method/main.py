import pandas as pd
import numpy as np
import sklearn.model_selection as sel
from sklearn import metrics
import dataprocess as dp
from net import BPNN
import random
from sklearn import preprocessing
from DNN import*

# read in dataset
dataset  = pd.read_excel('F:\Python projects\gas-dataset-regression\AirQualityUCI\AirQualityUCI.xlsx')

'''prepare data for training and test.'''
featureset = ['DT','PT08.S1(CO)', 'PT08.S2(NMHC)','T', 'PT08.S3(NOx)', 'PT08.S4(NO2)']
target = ['DT', 'CO(GT)']
d_t = dp.features(dataset, ['DT'])
feature_CO = dp.features(dataset, featureset)
target_CO = dp.target(dataset, target)
feature_CO['DT'], target_CO['DT'] = pd.to_datetime(feature_CO['DT']), pd.to_datetime(target_CO['DT'])
feature_CO = feature_CO.set_index('DT')
target_CO = target_CO.set_index('DT')
feature_CO, target_CO = dp.delMissing(feature_CO, target_CO)   #deal with missing data
#feature_CO, target_CO = dp.sort(feature_CO, target_CO, 'RH')   #sort by RH

skf = sel.KFold(n_splits=5)
MAE = MBE = RMSE = RE = 0
res = []
# print dp.baseline(taarget, reg_feature_CO['PT08.S1(CO)'])

#try on Dnn
param = preprocessing.scale(feature_CO)
param = np.array(param)
target = np.array(target_CO)
target = np.reshape(target, (len(target),))
for train_index, test_index in skf.split(param, target):
    # split training set and test set
    train_param, test_param = param[train_index], param[test_index]
    train_target, test_target = target[train_index], target[test_index]


'''
    # network parameters
    learning_rate = 0.01
    hidden_unit = 5
    input_demension = len(feature_CO.columns)
    output_dimension  = 1
    iteration = 100
    numTrain = len(train_target)
    numTest = len(test_target)

    # build network, train and test
    dnn = BPNN(ni = input_demension, nh = hidden_unit, no=output_dimension, func = 3,  rate = learning_rate, regularization= 1)
    dnn.numTrain, dnn.numTest = numTrain, numTest
    for i in range(iteration):
        for j in range(len(train_param)):
            dnn.train(np.array([train_param[j]]), np.array([[train_target[j]]]))
        print "iteration %d done"%(i)
    x = random.randint(0,99)
    #dnn.saveModel("run/model%d.txt"%x)
    test_target_pred = dnn.predict(test_param)
    print test_target_pred

    #evaluate results
    rmse_score = np.sqrt(metrics.mean_squared_error(test_target, test_target_pred))
    mae_score = dp.MAE(test_target, test_target_pred)
    mbe_score = dp.MBE(test_target, test_target_pred)
    relative_error = dp.relativeError(test_target, test_target_pred)
    print rmse_score, mae_score, mbe_score, relative_error
'''