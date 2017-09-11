import pandas as pd
import numpy as np
import sklearn.model_selection as sel
from sklearn import metrics
import dataprocess as dp
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.linear_model import LassoCV
from read_data import*

warnings.filterwarnings('ignore')

def linear():
    feature, target = data_read()  #read in dataset
    MAE = MBE = RMSE = RE = 0
    param = dp.polynomia(feature, a=2)  #polynomialization
    #param = np.array(feature)
    target = np.array(target)  #change traning target vector into an array
    target = np.reshape(target, (len(target),))   # reshape from (dataser_size , 1) into (daraset_size, )

    train_param, test_param = param[0000:3000, :], param[3000:4000, :]
    train_target, test_target = target[0000:3000], target[3000:4000]

    # train_param, test_param = param[train_index], param[test_index]
    # train_target, test_target = target[train_index], target[test_index]

    # regressiong
    # reg = ElasticNetCV(cv = c+2, l1_ratio= r/10.0)
    # reg = LinearRegression()
    reg = LassoCV(normalize=True, max_iter=1000)  #initialize a lasso regressor
    # reg = BaggingRegressor(LassoCV(max_iter=5000, normalize= True), max_samples=0.2)

    test_target_pred = reg.fit(train_param, train_target).predict(test_param)  # generate prediction results on test set

    # estimate the performance of regression
    """evaluate the results"""
    mae_score = dp.MAE(test_target, test_target_pred)
    mbe_score = dp.MBE(test_target, test_target_pred)
    rmse_score = np.sqrt(metrics.mean_squared_error(test_target, test_target_pred))  # RMSE score
    relative_error = dp.relativeError(test_target, test_target_pred)
    MAE = MAE + mae_score
    MBE = MBE + mbe_score
    RMSE = rmse_score + RMSE
    RE = RE + relative_error

    print RMSE, MAE, MBE, RE
    print reg.coef_

    CI = dp.confidence_interval(test_param, test_target_pred) #calculate confidence intervals
    err = 0
    err_value = []
    for i in range(len(test_target)): #count number of golden data that are not within confidence intervals
        if test_target[i] < CI[i, 0] or test_target[i] > CI[i, 1]:
            err = err + 1
            err_value.append(test_target[i])
    print err, len(test_target)

    """draw plots"""
    x = range(len(test_target))
    test_target_pred=np.reshape(test_target_pred,(len(test_target_pred),))
    y = np.dstack((test_target, test_target_pred, CI[:, 0], CI[:, 1]))[0]
    #y = y[np.lexsort(y[:, ::-1].T)]
    y1 = np.array(y[:, 0])  # golden
    y2 = np.array(y[:, 1])  # predict
    y3 = np.array(y[:, 2])  # low
    y4 = np.array(y[:, 3])  # high
    plt.plot(x, y1, color='red', lw=1, label = 'Golden', alpha = 0.75)
    plt.plot(x, y2, color='blue', lw=1, label = 'Linear', alpha = 0.75)
    #plt.plot(x, y3, color='blue', lw=0.5, linestyle='--', alpha=0.25)
    #plt.plot(x, y4, color='blue', lw=0.5, linestyle='--',alpha=0.25)
    #plt.fill_between(x, y3, y4, color='blue', alpha=0.25)

    plt.legend()
    plt.show()

    return test_target_pred
linear()
