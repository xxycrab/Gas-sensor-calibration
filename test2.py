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
from sklearn.svm import SVR, LinearSVR
from sklearn import preprocessing
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
import sklearn.model_selection as sel
from sklearn import metrics
#import histogram as hist
import matplotlib.dates as mdates
import dataprocess as dp


# read in dataset
dataset  = pd.read_excel('F:\Python projects\gas-dataset-regression\AirQualityUCI\AirQualityUCI.xlsx')

'''prepare data for training and test.'''
featureset = ['DT','PT08.S1(CO)', 'PT08.S2(NMHC)','T', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'RH']
target = ['DT', 'C6H6(GT)']
d_t = dp.features(dataset, ['DT'])
feature_CO = dp.features(dataset, featureset)
target_CO = dp.target(dataset, target)
feature_CO['DT'], target_CO['DT'] = pd.to_datetime(feature_CO['DT']), pd.to_datetime(target_CO['DT'])
feature_CO = feature_CO.set_index('DT')
target_CO = target_CO.set_index('DT')
feature_CO, target_CO = dp.delMissing(feature_CO, target_CO)   #deal with missing data
#feature_CO, target_CO = dp.sort(feature_CO, target_CO, 'RH')   #sort by RH

# print dp.baseline(taarget, reg_feature_CO['PT08.S1(CO)'])
'''
param = np.array(feature_CO)
#param = dp.polynomia(feature_CO, a=2)
target = np.array(target_CO)
target = np.reshape(target, (len(target),))
reg = LassoCV(normalize= True, cv = 3, max_iter= 3000)
pred = []
for i in range(336, len(param)):
    train_param, test_param = param[i-336: i], [param[i]]
    train_target, test_target = target[i-336: i], [target[i]]
    test_target_pred = reg.fit(train_param, train_target).predict(test_param)[0]
    pred.append(test_target_pred)
pred = np.reshape(np.array(pred), (len(pred),))
print pred
golden = target[336:]
rmse_score = np.sqrt(metrics.mean_squared_error(golden, pred))
mae_score = dp.MAE(golden, pred)
mbe_score = dp.MBE(golden, pred)
relative_error = dp.relativeError(golden, pred)
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
    param = dp.polynomia(feature_CO, a=2)
    #param = np.array(feature_CO)
    target = np.array(target_CO)    #change into ndarray
    target = np.reshape(target, (len(target),))    #change into (len,) instead of (len, 1)

    skf = sel.KFold(n_splits =5)
    MAE = MBE = RMSE = RE = 0
    res = []
    for train_index, test_index in skf.split(param, target):
        train_param, test_param = param[0:2000,:], param[2000:3000,:]
        train_target, test_target = target[0:2000], target[2000:3000]

        #train_param, test_param = param[train_index], param[test_index]
        #train_target, test_target = target[train_index], target[test_index]

        # regressiong
        #reg = ElasticNetCV(cv = c+2, l1_ratio= r/10.0)
        #reg = LinearRegression()
        reg = LassoCV(normalize = True , max_iter = 3000)
        #reg= BayesianRidge(normalize = True)
        #reg = KernelRidge(alpha=1.0, degree = 1)
        #reg = SVR(kernel = 'poly')
        #lasso = LassoCV()
        #reg = GradientBoostingRegressor(loss = 'lad', learning_rate = 0.1, n_estimators=200)
        #reg = BaggingRegressor(LassoCV(max_iter=5000, normalize= True), max_samples=0.2)

        test_target_pred = reg.fit(train_param, train_target).predict(test_param)   # generate prediction results on test set

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

        CI = dp.confidence_interval(test_param, test_target_pred)
        err = 0
        err_value= []
        for i in range(len(test_target)):
            if test_target[i] < CI[i,0] or test_target[i]> CI[i,1]:
                err = err + 1
                err_value.append(test_target[i])
        print err, len(test_target)
        break


    print(RMSE,MAE,MBE,RE)
    x = range(len(test_target))
    y = np.dstack((test_target,test_target_pred,CI[:,0], CI[:,1]))[0]
    y = y[np.lexsort(y[:,::-1].T)]
    y1 = np.array(y[:,0])    #golden
    y2 = np.array(y[:,1])    #predict
    y3 = np.array(y[:,2])    #low
    y4 = np.array(y[:,3])    #high
    plt.plot(x,y1,color='red', lw=1, alpha=1, label = 'Golden')
    plt.plot(x,y2,color='black',lw=1, label = 'linear', alpha=0.75)
    plt.plot(x,y3, color='blue', lw=0.5, linestyle='--', alpha=0)
    plt.plot(x, y4, color='blue', lw=0.5, linestyle='--', alpha=0)
    plt.fill_between(x, y3, y4, color='blue', alpha=0)
    plt.legend()
    plt.show()

    #res.append([MAE/5.0,  MBE/5.0, RMSE/5.0, RE/5.0])   # record the average score
    #print res, size