import pandas as pd
import numpy as np
import sklearn.model_selection as sel
from sklearn import metrics
import dataprocess as dp
from NN_class import *
import matplotlib.pyplot as plt
import warnings
from read_data import *
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

def DNN(mode = 1, splits = 5.0):
    feature, target = data_read()  #read in dataset
    skf = sel.KFold(n_splits= splits)  #splits of cross-validation, default is 5
    MAE = MBE = RMSE = RE = 0  #record scores
    param = np.array(feature)  #change feature matrix into ndarray
    #param = dp.polynomia(feature, a=2)
    target = np.array(target)   #change target vector into ndarray
    target = np.reshape(target, (len(target),))  # reshape from (dataser_size , 1) into (daraset_size, )
    
    idim, hiddim, odim = param.shape[1], [[20, 10], [5, 5], [7, 7], [15, 15], [20, 20]], 1 #assign input dimesion,
    epochs = [200, 500, 1000, 3000, 5000]  #learning epochs
    activations = [[tansig, 'relu'], [tansig, tansig]]# decide activation function for each layer, should be a list
                                                        # if no activation function, assign 'None' for that layer
    for h in hiddim:
        for e in epochs:
            for act in activations:
                for i in range(10):
                    for train_index, test_index in skf.split(param, target):
                        if mode == 2:  # cross-validation mode, training and test set are divided randomly by KFold
                            train_param, test_param = param[train_index], param[test_index]
                            train_target, test_target = target[train_index], target[test_index]
                        elif mode == 1: # time-series mode
                            train_param, test_param = param[0:2000, :], param[2000:3000, :]
                            train_target, test_target = target[0:2000], target[2000:3000]
                        std, avg = np.std(test_target), np.mean(test_target)   # calculate standard diviation and mean value for renormalization
                        scale = StandardScaler() # normalizer, normalize on standard diviation
                        train_param, test_param, train_target = scale.fit_transform(train_param), scale.fit_transform(
                            test_param), scale.fit_transform(train_target)  # normalization on sample features and training target

                        dnn = DNN()  #initialize a BPNN model
                        dnn.build(input_dim=idim, hidden_dim=h, output_dim=odim, activations=act, drop_rate=0, \
                                   kernel_regularizer=None) #assign input and output dimension, hidden layer structure,
                                                            #activation functions, dropout rate, regularizers
                        dnn.fit(train_param, train_target, batch_size=len(train_target) ,epochs=e) # fit the model
                        dnn.Model.save("model.h5") # save the model into a hdf5 file
                        # dnn.Model.save_weights("model_weights.h5")

                        test_target_pred = dnn.predict(test_param)  #generate prediction results(normalized)
                        test_target_pred = test_target_pred * std + avg  #renormalization

                        """evaluate the results"""
                        rmse_score = np.sqrt(metrics.mean_squared_error(test_target, test_target_pred))
                        mae_score = dp.MAE(test_target, test_target_pred)
                        mbe_score = dp.MBE(test_target, test_target_pred)
                        relative_error = dp.relativeError(test_target, test_target_pred)

                        if mode ==1 :
                            print "hiddim:", h, "epoch:", e, "activation:", act
                            print "scores:", rmse_score, mae_score, mbe_score, relative_error
                            break
                        elif mode == 2:
                            MAE = MAE + mae_score
                            MBE = MBE + mbe_score
                            RMSE = RMSE + rmse_score
                            RE = RE + relative_error

                    if mode == 2:
                        print "hiddim:", h, "epoch:", e, "activation:", act
                        print "scores:", RMSE / splits, MAE / splits, MBE / splits, RE / splits
                    break
                break
            break
        break
    
    CI = dp.confidence_interval(test_param, test_target_pred)    #calculate confidence intervals
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
    plt.plot(x, y1, color='red', lw=1)
    plt.plot(x, y2, color='black', lw=1)
    plt.plot(x, y3, color='blue', lw=0.5, linestyle='--')
    plt.plot(x, y4, color='blue', lw=0.5, linestyle='--')
    plt.fill_between(x, y3, y4, color='blue', alpha=0.25)
    plt.show()

DNN(mode = 1)
