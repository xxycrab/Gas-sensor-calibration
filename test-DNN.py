import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
import sklearn.model_selection as sel
from sklearn import metrics
import dataprocess as dp
from DNN import *
import matplotlib.pyplot as plt

# read in dataset
dataset  = pd.read_excel('F:\Python projects\gas-dataset-regression\AirQualityUCI\AirQualityUCI.xlsx')

'''prepare data for training and test.'''
featureset = ['DT','PT08.S1(CO)', 'PT08.S2(NMHC)','T', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'RH']
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
param = np.array(feature_CO)
#param = dp.polynomia(feature_CO, a=2)
target = np.array(target_CO)
target = np.reshape(target, (len(target),))
'''
for train_index, test_index in skf.split(param, target):
    print len(train_index)
    train_param, test_param = param[train_index], param[test_index]
    train_target, test_target = target[train_index], target[test_index]
    std = np.std(train_target)
    avg = np.mean(train_target)
    dnn = DNN()
    dnn.fit(train_param, train_target, epochs=2000)
    test_target_pred =dnn.predict(test_param)
    test_target_pred = test_target_pred * std + avg
    rmse_score = np.sqrt(metrics.mean_squared_error(test_target, test_target_pred))
    mae_score = dp.MAE(test_target, test_target_pred)
    mbe_score = dp.MBE(test_target, test_target_pred)
    relative_error = dp.relativeError(test_target, test_target_pred)
    print rmse_score, mae_score, mbe_score, relative_error
'''

train_param, test_param = param[0:2000,:], param[2000:3000,:]
train_target, test_target = target[0:2000], target[2000:3000]
std, avg = np.std(train_target), np.mean(train_target)
dnn = DNN()
dnn.fit(train_param, train_target)
test_target_pred =dnn.predict(test_param)
test_target_pred = test_target_pred * std + avg
#test_target_pred = (test_target_pred - pred_min)/(pred_max - pred_min) * ran + test_target.min()
rmse_score = np.sqrt(metrics.mean_squared_error(test_target, test_target_pred))
mae_score = dp.MAE(test_target, test_target_pred)
mbe_score = dp.MBE(test_target, test_target_pred)
relative_error = dp.relativeError(test_target, test_target_pred)
print rmse_score, mae_score, mbe_score, relative_error


CI = dp.confidence_interval(test_param, test_target_pred)
err = 0
err_value = []
for i in range(len(test_target)):
    if test_target[i] < CI[i, 0] or test_target[i] > CI[i, 1]:
        err = err + 1
        err_value.append(test_target[i])
print err, len(test_target)

x = range(len(test_target))
test_target_pred=np.reshape(test_target_pred,(len(test_target_pred),))
y = np.dstack((test_target, test_target_pred, CI[:, 0], CI[:, 1]))[0]
#y = y[np.lexsort(y[:, ::-1].T)]
y1 = np.array(y[:, 0])  # golden
y2 = np.array(y[:, 1])  # predict
y3 = np.array(y[:, 2])  # low
y4 = np.array(y[:, 3])  # high
plt.plot(x, y1, color='red', lw=1)
plt.plot(x, y2, color='black', lw=1)
plt.plot(x, y3, color='blue', lw=0.5, linestyle='--')
plt.plot(x, y4, color='blue', lw=0.5, linestyle='--')
plt.fill_between(x, y3, y4, color='blue', alpha=0.25)
plt.show()

'''
    #x = range(len(test_param))
    #plt.plot(x, test_target, color = 'blue')
    #plt.plot(x, test_target_pred, color='red')
    #plt.show()
    x = np.sort(test_target)
    y = np.sort(test_target_pred, axis = None)

    xx=[0,1.2,3,4,5,6,7,8]
    yy = [0,1.2,3,4,5,6,7,8]
    plt.plot(xx,yy, color='red', lw=0.5 )
    plt.plot(x, y, color='black', lw = 0.5)
    plt.scatter(x, y, color='blue', s=1)
    plt.show()
'''
