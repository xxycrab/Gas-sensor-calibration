import pandas as pd
import numpy as np
from sklearn import preprocessing
from math import *
from sklearn import metrics
from sklearn.linear_model import LinearRegression

"""given the raw dataset and the features' name in list type, return a dataframe consisting of thess features"""
def get_features(dataset, featureset):
    for feature in featureset:
        if not (feature in dataset.columns.tolist()):
            print feature,"is not in dataset"
            return

    return dataset[featureset]

"""given the raw dataset and the target name in list type, return a dataframe consisting of traning target"""
def get_target(dataset, target):
    return dataset[target]

"""replace the missing data with numpy's mark(np.nan) for missing data"""
"""in the UCI dataset, the missing data are marked as -200.00"""
def delMissing(featureset, target):
    if not (type(target) is 'DataFrame'):
        target = pd.DataFrame(target)
    data = pd.concat([featureset, target], axis= 1)
    data= data.replace(-200.00, np.nan).dropna(how = 'any')
    featureset, target = data[featureset.columns], data[target.columns]
    return featureset, target

"""given dataset and exponent, return polynomial"""
def polynomia(dataset, a= 2):
    poly = preprocessing.PolynomialFeatures(a)
    dataset = poly.fit_transform(dataset)
    print poly.get_feature_names()
    return dataset

"""given golden data and prediction results, calculate mean absolute error"""
def MAE(golden, pred):
    golden, pred = np.reshape(golden, (len(golden),)), np.reshape(pred,(len(pred),))
    score = np.average(abs(golden - pred))
    return score

"""given golden data and prediction results, calculate mean bias error"""
def MBE(golden, pred):
    golden, pred = np.reshape(golden, (len(golden),)), np.reshape(pred,(len(pred),))
    score = np.average(golden - pred)
    return score

"""given golden data and prediction results, calculate mean relative error"""
def relativeError(golden, pred):
    RE=0
    golden, pred = np.reshape(golden, (len(golden),)), np.reshape(pred,(len(pred),))
    for i in range(len(golden)):
        RE = RE + abs(pred[i] - golden[i]) / golden[i]
    return RE/float(len(golden))

"""given learning samples, target and the pivot, sort the whole dataset upon the pivot"""
"""the pivot should be the column name of the dataset(dataframe type)"""
def sort(featureset, target, pivot):
    if not (type(target) is 'DataFrame'):
        target = pd.DataFrame(target)
    data = pd.concat([featureset, target], axis = 1)

    data = data.sort_values(by = [pivot])
    featureset, target = data[featureset.columns], data[target.columns]
    return featureset, target

"""given sample sets and prediction results, calculate the confidence interval"""
def confidence_interval(X, y_pred):
    y_interval = np.zeros([len(X),2])
    y_std = np.std(y_pred, ddof = 1)
    #k = X.shape[0]
    #n = len(X)
    X_mat, y_pred_mat = np.matrix(X),np.matrix(y_pred)
    XTX = X_mat.T * X_mat
    for i in range(len(y_pred)):
        x_var = np.sqrt(X_mat[i,:] * (XTX ** -1) * (X_mat[i,:].T))
        interval = y_std * x_var[0,0] * 1.96
        y_interval[i,0], y_interval[i,1] = y_pred[i]-interval, y_pred[i]+interval
    return y_interval