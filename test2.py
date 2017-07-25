import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn import preprocessing
import sklearn.model_selection as sel
from sklearn import metrics
#import histogram as hist
from sklearn.metrics import accuracy_score
import matplotlib.dates as mdates
import dataprocess as dp
import histogram as hist
from DNN import *
from keras.models import Sequential
#import LWLinear as LW

# read in dataset
dataset  = pd.read_excel('F:\Python projects\gas-dataset-regression\AirQualityUCI\AirQualityUCI.xlsx')

'''prepare data for training and test.'''
featureset = ['DT','PT08.S1(CO)', 'PT08.S2(NMHC)','T', 'PT08.S3(NOx)', 'PT08.S4(NO2)']
target = ['DT','CO(GT)']
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

param = np.array(feature_CO)
target = np.array(target_CO)
target = np.reshape(target, (len(target),))
for train_index, test_index in skf.split(param, target):
    print len(train_index)
    train_param, test_param = param[train_index], param[test_index]
    train_target, test_target = target[train_index], target[test_index]
    dnn = DNN()
    test_target_pred = dnn.train(train_param, train_target, test_param,test_target)
    #test_target_pred = (test_target_pred - pred_min)/(pred_max - pred_min) * ran + test_target.min()
    print test_target_pred
    rmse_score = np.sqrt(metrics.mean_squared_error(test_target, test_target_pred))
    mae_score = dp.MAE(test_target, test_target_pred)
    mbe_score = dp.MBE(test_target, test_target_pred)
    relative_error = dp.relativeError(test_target, test_target_pred)
    print rmse_score, mae_score, mbe_score, relative_error

'''
for i in range (1,9):

    left, right = 10*i, 10*(i+1)
    #data = pd.concat([feature_CO, target_CO],axis = 1)
    #data = data.loc[(data['RH']>= left) & (data['RH']<= right)]
    #reg_feature_CO, reg_target_CO= data[feature_CO.columns], data[target_CO.columns]   #choose pieces of data

    #sensor = reg_feature_CO['PT08.S1(CO)']
    #sensitivity = sensor.std()

    #if i <= 9: reg_feature_CO, reg_target_CO = feature_CO['2004-%d' %(i+3)], target_CO['2004-%d' %(i+3)]
    #else: reg_feature_CO, reg_target_CO = feature_CO['2005-%d' %(i-9)], target_CO['2005-%d' %(i-9)]
    size = len(feature_CO)
    param = dp.polynomia(feature_CO, a=1)
    target = np.array(target_CO)    #change into ndarray
    target = np.reshape(target, (len(target),))    #change into (len,) instead of (len, 1)

    skf = sel.KFold(n_splits =5)
    MAE = MBE = RMSE = RE = 0
    res = []
    #print dp.baseline(target, reg_feature_CO['PT08.S1(CO)'])
    for train_index, test_index in skf.split(param, target):
        train_param, test_param = param[train_index], param[test_index]
        train_target, test_target = target[train_index], target[test_index]

        # regressiong
        #reg = ElasticNetCV(cv = c+2, l1_ratio= r/10.0)
        #reg = LinearRegression()
        reg = LassoCV(normalize = True , max_iter = 8000)
        #reg= BayesianRidge(normalize = True)
        #reg = KernelRidge(alpha=1.0, degree = 1)
        test_target_pred = reg.fit(train_param, train_target).predict(test_param)   # generate prediction results on test set
        print reg.alpha_

        # estimate the performance of regression
        #test_target_pred = preprocessing.scale(test_target_pred)
        #test_target = preprocessing.scale(test_target)
        mae_score = dp.MAE(test_target, test_target_pred)
        mbe_score = dp.MBE(test_target, test_target_pred)
        rmse_score = np.sqrt(metrics.mean_squared_error(test_target, test_target_pred))   # RMSE score
        relative_error = dp.relativeError(test_target, test_target_pred)
        MAE = MAE + mae_score
        MBE = MBE + mbe_score
        RMSE = rmse_score + RMSE
        RE = RE + relative_error
    res.append([MAE/5.0,  MBE/5.0, RMSE/5.0, RE/5.0])   # record the average score
    print res, size
'''
'''
res = []
for c in range(3):
    #for r in range(7,10,1):
    skf = sel.StratifiedKFold(n_splits =5)
    A = S = R = RMSE = 0
    for train_index, test_index in skf.split(reg_feature_CO, reg_target_CO):
        train_param, test_param = reg_feature_CO[train_index], reg_feature_CO[test_index]
        train_target, test_target = reg_target_CO[train_index], reg_target_CO[test_index]

        # regression
        err = 0   # count number of mistake prediction result
        test_size = len(test_target)
        #reg = ElasticNetCV(cv = c+2, l1_ratio= r/10.0)
        reg = LassoCV(cv=c + 2, normalize = True)
        test_target_pred = reg.fit(train_param, train_target).predict(test_param)   # generate prediction results on test set

        # normalize with average
        #min_max_scaler = preprocessing.MinMaxScaler()
        #test_target_norm = min_max_scaler.fit_transform(test_target.reshape(-1,1))
        #test_target_pred_norm = min_max_scaler.fit_transform(test_target_pred.reshape(-1,1))
        #normaliza with deviation
        test_target_norm = preprocessing.scale(test_target)
        test_target_pred_norm = preprocessing.scale(test_target_pred)

        avg = test_target_norm.mean()
        avg_pred = test_target_pred_norm.mean()
        # estimate the performance of regression
        for i in range(test_size):
            if abs(test_target_pred[i] - test_target[i])/ test_target[i] > 0.05:
                err = err + 1
        acc = (test_size - err) / float(test_size)   # the most traditional accuracy, the threshold value is set by an observation
                                                     # that refrence data is mostly smaller than 4 and larger than 2,
                                                     # thus a 5/% - 10% derivation should be accepted in prediction
        r2score = r2_score(test_target, test_target_pred)   # R^2 score
        rmse_score = np.sqrt(metrics.mean_squared_error(test_target_norm, test_target_pred_norm))   # RMSE score
        A = A + acc
        R = R+ r2score
        RMSE = rmse_score + RMSE
    res.append([c+2, A/5.0, RMSE/5.0, avg, avg_pred])   # record the average score
print res
'''
'''
# experiment 1: simplest version
res = []
#regression and cross validation
skf = sel.StratifiedKFold(n_splits = 5)
A = S= R = RMSE = 0
for train_index, test_index in skf.split(reg_feature_CO, reg_target_CO):
    train_param, test_param = reg_feature_CO[train_index], reg_feature_CO[test_index]
    train_target, test_target = reg_target_CO[train_index], reg_target_CO[test_index]
    # lasso regression, use class LassoCV
    err = 0   # count number of mistake prediction result
    test_size = len(test_target)
    lasso = LassoCV(normalize= False)
    test_target_pred = lasso.fit(train_param, train_target).predict(test_param)   # generate prediction results on test set

    # estimate the performance of regression
    for i in range(test_size):
        if abs(test_target_pred[i] - test_target[i]) >=0.2:
            err = err + 1
    acc = (test_size - err) / float(test_size)   # the most traditional accuracy, the threshold value is set by an observation
                                                 # that refrence data is mostly smaller than 8 and larger than 2,
                                                 # thus a 5/% - 10% derivation should be accepted in prediction
    score = lasso.score(train_param, train_target)   # default estimation defined by class LassoCV
    r2score = r2_score(test_target, test_target_pred)   # R^2 score
    rmse_score = np.sqrt(metrics.mean_squared_error(test_target, test_target_pred))   # RMSE score
    A = A + acc
    S = S + score
    R = R+ r2score
    RMSE = rmse_score + RMSE
    #print acc, r2score, score, rmse_score
    print lasso.coef_
res.append([A/5.0, S/5.0, R/5.0, RMSE/5.0])   # record the average score
print res
'''

'''
#experiment 2: influence of normalization
res = []
#regression and cross validation
skf = sel.StratifiedKFold(n_splits = 5)
A = S= R = RMSE = 0
#reg_feature_CO= preprocessing.normalize(reg_feature_CO)
for train_index, test_index in skf.split(reg_feature_CO, reg_target_CO):
    train_param, test_param = reg_feature_CO[train_index], reg_feature_CO[test_index]
    train_target, test_target = reg_target_CO[train_index], reg_target_CO[test_index]
    # lasso regression, use LassoCV
    err = 0  # count number of mistake prediction result
    test_size = len(test_target)
    lasso = LassoCV()
    test_target_pred = lasso.fit(train_param, train_target).predict(test_param)  # generate prediction results on test set

    # estimate the performance of regression
    for i in range(test_size):
        if abs(test_target_pred[i] - test_target[i]) >=0.2:
            err = err + 1
    acc = (test_size - err) / float(test_size)   # the most traditional accuracy, the threshold value is set by an observation
                                                 # that refrence data is mostly smaller than 8 and larger than 2,
                                                 # thus a 5/% - 10% derivation should be accepted in prediction
    score = lasso.score(train_param, train_target)   # default estimation defined by class LassoCV
    r2score = r2_score(test_target, test_target_pred)   # R^2 score
    rmse_score = np.sqrt(metrics.mean_squared_error(test_target, test_target_pred))   # RMSE score
    A = A + acc
    S = S + score
    R = R+ r2score
    RMSE = rmse_score + RMSE
    #print acc, r2score, score, rmse_score
    #print lasso.coef_
res.append([A/5.0, S/5.0, R/5.0, RMSE/5.0])   # record the average score
print res
'''

'''
#experiment 3: test for an optimal value of cv
res = []
for c in range(2,42,2):
    #regression and cross validation
    skf = sel.StratifiedKFold(n_splits = 5)
    A = S= R = RMSE = 0
    for train_index, test_index in skf.split(reg_feature_CO, reg_target_CO):
        train_param, test_param = reg_feature_CO[train_index], reg_feature_CO[test_index]
        train_target, test_target = reg_target_CO[train_index], reg_target_CO[test_index]
        # lasso regression, use class LassoCV
        err = 0   # count number of mistake prediction result
        test_size = len(test_target)
        lasso = LassoCV(cv = c, normalize= True)
        test_target_pred = lasso.fit(train_param, train_target).predict(test_param)   # generate prediction results on test set

        # estimate the performance of regression
        for i in range(test_size):
            if abs(test_target_pred[i] - test_target[i]) >=0.2:
                err = err + 1
        acc = (test_size - err) / float(test_size)   # the most traditional accuracy, the threshold value is set by an observation
                                                     # that refrence data is mostly smaller than 8 and larger than 2,
                                                     # thus a 5/% - 10% derivation should be accepted in prediction
        score = lasso.score(train_param, train_target)   # default estimation defined by class LassoCV
        r2score = r2_score(test_target, test_target_pred)   # R^2 score
        rmse_score = np.sqrt(metrics.mean_squared_error(test_target, test_target_pred))   # RMSE score
        A = A + acc
        S = S + score
        R = R+ r2score
        RMSE = rmse_score + RMSE
        #print acc, r2score, score, rmse_score
        #print lasso.coef_
    res.append([A/5.0, S/5.0, R/5.0, RMSE/5.0])   # record the average score
print res
res = np.array(res)
x = range(2, 42 , 2)
plt.plot(x, res[:,3], label='RMSE score', color = 'blue', linestyle = '-')
plt.scatter(x, res[:,3], label='RMSE score', color = 'blue')
plt.xlabel('cv value')
plt.ylabel('RMSE score')
plt.show()
plt.plot(x, res[:,0], label='accuracy', color = 'black', linestyle = '--')
plt.scatter(x, res[:,0], label='accuracy', color = 'black')
plt.xlabel('cv value')
plt.ylabel('RMSE score')
plt.show()
'''

'''
#experiment 4: test for combining L1 and L2 regularization
res = []
#for a in range(10):
for r in range(11):
    a = 0.2
#r = 0.8
    skf = sel.StratifiedKFold(n_splits = 5)
    A = S = R = RMSE = 0
    for train_index, test_index in skf.split(reg_feature_CO, reg_target_CO):
        train_param, test_param = reg_feature_CO[train_index], reg_feature_CO[test_index]
        train_target, test_target = reg_target_CO[train_index], reg_target_CO[test_index]
        # regression, use class elastic net
        err = 0   # count number of mistake prediction result
        test_size = len(test_target)
        reg = ElasticNet(alpha = a, l1_ratio=r/10.0, normalize = False)
        #reg = ElasticNetCV()
        test_target_pred = reg.fit(train_param, train_target).predict(test_param)   # generate prediction results on test set

        # estimate the performance of regression
        for i in range(test_size):
            if abs(test_target_pred[i] - test_target[i]) >=0.2:
                err = err + 1
        acc = (test_size - err) / float(test_size)   # the most traditional accuracy, the threshold value is set by an observation
                                                     # that refrence data is mostly smaller than 4 and larger than 2,
                                                     # thus a 5/% - 10% derivation should be accepted in prediction
        score = reg.score(train_param, train_target)   # default estimation defined by class LassoCV
        r2score = r2_score(test_target, test_target_pred)   # R^2 score
        rmse_score = np.sqrt(metrics.mean_squared_error(test_target, test_target_pred))   # RMSE score
        A = A + acc
        S = S + score
        R = R+ r2score
        RMSE = rmse_score + RMSE
        #print acc, r2score, score, rmse_score
        #print lasso.coef_
    res.append([A/5.0, S/5.0, R/5.0, RMSE/5.0])   # record the average score
#print res
'''
'''
res = np.array(res)
plt.plot([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], res[:,0], label='accuracy', color = 'blue', linestyle = '-')
plt.scatter([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], res[:,0], label='accuracy', color = 'blue')
plt.xlabel('l1_ratio')
plt.ylabel('accuracy')
plt.show()
plt.plot([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], res[:,1],  label='default score', color = 'green', linestyle = '--')
plt.scatter([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], res[:,1], label='default score', color = 'green')
plt.xlabel('l1_ratio')
plt.ylabel('default score')
plt.show()
plt.plot([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], res[:,2], label='R^2 score', color = 'grey', linestyle = '-.')
plt.scatter([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], res[:,2], label='R^2 score', color = 'grey')
plt.xlabel('l1_ratio')
plt.ylabel('R^2 score')
plt.show()
plt.plot([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], res[:,3], label='RMSE score', color = 'black', linestyle = ':')
plt.scatter([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], res[:,3], label='RMSE score', color = 'black')
plt.xlabel('l1_ratio')
plt.ylabel('RMSE score')
plt.show()
'''

