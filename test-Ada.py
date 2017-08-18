import pandas as pd
import numpy as np
import sklearn.model_selection as sel
from sklearn import metrics
import dataprocess as dp
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasRegressor
import matplotlib.pyplot as plt
from DNN import *

# read in dataset
def data_read():
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
    return feature_CO, target_CO

def model_build(input_dim, hidden_dim, output_dim, activation):
    model = Sequential()
    idim = input_dim
    odim = output_dim
    hw = hidden_dim
    act = activation

    def tansig(x):
        return 2/(1+np.exp(-2*x))-1
    for i in range(len(hw)):
        if i==0:
            model.add(Dense(input_dim=idim, units=hw[i], activation=act[i], init='normal'))
        else:
            model.add(Dense(units=hw[i], activation=act[i], init='normal'))
    model.add(Dense(units=odim, activation='linear', init='normal'))

    sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    return model

def TDdata(TDL, data):
    TDdata = data[0:len(data)-(TDL-1),:]
    for i in range(1, TDL):
        TDdata = np.concatenate((TDdata, data[i:len(data)-(TDL-1)+i,: ]), axis=1)
    return TDdata

def ada():
    feature, target = data_read()
    skf = sel.KFold(n_splits=5)
    MAE = MBE = RMSE = RE = 0
    param = np.array(feature)
    target = np.array(target)  # change into ndarray
    target = np.reshape(target, (len(target),))  # change into (len,) instead of (len, 1)

    for train_index, test_index in skf.split(param, target):
        train_param, test_param = param[0:2000,:], param[2000:3000,:]
        train_target, test_target = target[0:2000], target[2000:3000]
        #train_param, test_param = param[train_index], param[test_index]
        #train_target, test_target = target[train_index], target[test_index]
        avg = np.mean(test_target)
        std = np.std(test_target)

        train_param, test_param = preprocessing.scale(train_param), preprocessing.scale(test_param)
        train_target = preprocessing.scale(train_target)

        idim = feature.shape[1]
        outdim = 1
        hiddim = [10,10]
        act = [tansig, 'relu']
        TDL = 12

        base = KerasRegressor(build_fn=model_build, input_dim=idim*TDL, hidden_dim=hiddim, output_dim=outdim,activation=act,\
                              verbose=0, epochs = 1000)
        ada = AdaBoostRegressor(base_estimator= base, n_estimators=10, learning_rate=0.1, loss = 'square')

        train_param, test_param = TDdata(TDL = TDL, data = train_param), TDdata(TDL=TDL, data = test_param)
        train_target, test_target = train_target[TDL-1:], test_target[TDL - 1:]
        ada.fit(train_param, train_target)
        test_target_pred = ada.predict(test_param)
        test_target_pred = test_target_pred*std+avg


        # estimate the performance of regression
        test_target_pred = np.reshape(test_target_pred, (len(test_target_pred),))
        mae_score = dp.MAE(test_target, test_target_pred)
        mbe_score = dp.MBE(test_target, test_target_pred)
        rmse_score = np.sqrt(metrics.mean_squared_error(test_target, test_target_pred))
        relative_error = dp.relativeError(test_target, test_target_pred)
        MAE = MAE + mae_score
        MBE = MBE + mbe_score
        RMSE = rmse_score + RMSE
        RE = RE + relative_error
        print (RMSE,MAE,MBE,RE)

        CI = dp.confidence_interval(test_param, test_target_pred)
        err = 0
        err_value = []
        for i in range(len(test_target)):
            if test_target[i] < CI[i, 0] or test_target[i] > CI[i, 1]:
                err = err + 1
                err_value.append(test_target[i])
        print err, len(test_target)

        x = range(len(test_target))
        test_target_pred = np.reshape(test_target_pred, (len(test_target_pred),))
        y = np.dstack((test_target, test_target_pred, CI[:, 0], CI[:, 1]))[0]
        y = y[np.lexsort(y[:, ::-1].T)]
        y1 = np.array(y[:, 0])  # golden
        y2 = np.array(y[:, 1])  # predict
        y3 = np.array(y[:, 2])  # low
        y4 = np.array(y[:, 3])  # high
        plt.plot(x, y1, color='red', lw=1, label='Golden', alpha=1)
        plt.plot(x, y2, color='blue', lw=1, label='NN', alpha=0.75)
        plt.plot(x, y3, color='blue', lw=0.5, linestyle='--', alpha=0.25)
        plt.plot(x, y4, color='blue', lw=0.5, linestyle='--', alpha=0.25)
        plt.fill_between(x, y3, y4, color='blue', alpha=0.25)

        plt.legend()
        plt.show()
        break

    print(RMSE/5.0,MAE/5.0,MBE/5.0,RE/5.0)
    return test_target_pred

ada()