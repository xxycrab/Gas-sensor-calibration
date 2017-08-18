import pandas as pd
import numpy as np
import sklearn.model_selection as sel
from sklearn import metrics
import dataprocess as dp
from DNN import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def data_read():
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


def tdnn():
    feature_CO, target_CO = data_read()

    MAE = MBE = RMSE = RE = 0
    param = np.array(feature_CO)
    target = np.array(target_CO)
    target = np.reshape(target, (len(target),))

    train_param, test_param = param[0:2000,:], param[2000:3000,:]
    train_target, test_target = target[0:2000], target[2000:3000]
    std, avg = np.std(test_target), np.mean(test_target)
    scale = StandardScaler()
    train_param, test_param, train_target = scale.fit_transform(train_param), scale.fit_transform(test_param), \
                                            scale.fit_transform(train_target)

    idim, hiddim, odim=feature_CO.shape[1], [10,10], 1

    #tdnn = TDNN(filter_size=6)
    #tdnn.build(input_dim=idim, hidden_dim=hiddim, output_dim=odim, activations=[tansig, 'relu'])
    #tdnn.fit(train_param, train_target, batch_size=len(train_target)-tdnn1.TDL+1, epochs=1000)

    tdnn = Multi_TDNN(filter_sizes=[6,8,10])
    tdnn.build(input_dim=idim, hidden_dim=hiddim, output_dim=odim, activations=[tansig, 'relu'])
    tdnn.fit(train_param, train_target, batch_size=len(train_target)-max(tdnn.TDL)+1, epochs=1000)

    test_target_pred =tdnn.predict(test_param)
    test_target_pred = test_target_pred*std+avg
    test_target = test_target[max(tdnn.TDL)-1:]

    rmse_score = np.sqrt(metrics.mean_squared_error(test_target, test_target_pred))
    mae_score = dp.MAE(test_target, test_target_pred)
    mbe_score = dp.MBE(test_target, test_target_pred)
    relative_error = dp.relativeError(test_target, test_target_pred)
    print rmse_score, mae_score, mbe_score, relative_error

    CI = dp.confidence_interval(test_param[max(tdnn.TDL)-1:,:], test_target_pred)
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
    y = y[np.lexsort(y[:, ::-1].T)]
    y1 = np.array(y[:, 0])  # golden
    y2 = np.array(y[:, 1])  # predict
    y3 = np.array(y[:, 2])  # low
    y4 = np.array(y[:, 3])  # high
    plt.plot(x, y1, color='red', lw=1, label = 'Golden', alpha = 1)
    plt.plot(x, y2, color='blue', lw=1, label = 'TDNN', alpha = 0.75)
    plt.plot(x, y3, color='blue', lw=0.5, linestyle='--', alpha=0.25)
    plt.plot(x, y4, color='blue', lw=0.5, linestyle='--',alpha=0.25)
    plt.fill_between(x, y3, y4, color='blue', alpha=0.25)

    plt.legend()
    plt.show()

    return test_target_pred
