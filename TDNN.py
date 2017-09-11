import pandas as pd
import numpy as np
import sklearn.model_selection as sel
from sklearn import metrics
import dataprocess as dp
from NN_class import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
from keras.layers.core import regularizers
from read_data import*

warnings.filterwarnings('ignore')

def tdnn():
    feature, target = data_read()  # read in dataset
    param = np.array(feature)  # change feature matrix into ndarray type
    #param = dp.polynomia(feature, a=2)  #polynomialize
    target = np.reshape(target, (len(target),))  # reshape target array from (dataser_size , 1) into (daraset_size, )
    target = np.array(target) # change target vector into ndarray type

    idim, hiddim, odim = param.shape[1], [[20,10],[5,5],[7,7],[15,15],[20,20]], 1  #assign input dimesion,
                                                                    # hidden layer size (type: list), and output dimension
    TDL = [[12],[6,24],[12,24], [6,12,24],[3,6],[3,9],[3,6,9],[6,8,10],[4,6],[4,6,12]] #assign TDL values, must be a list
    epochs = [200,500,1000,3000,5000]  #learning epochs
    activations = [[tansig, 'relu'],[tansig,tansig]]  # decide activation function for each layer, should be a list
                                                        # if no activation function, assign 'None' for that layer
    for h in hiddim:
        for tdl in TDL:
            for e in epochs:
                for act in activations:
                    for i in range(10):
                        train_param, test_param = param[0000:5000, :], param[5000:6000, :]   #sample features for training and test respectively
                        train_target, test_target = target[0000:5000], target[5000:6000]  #target for training and test respectively
                        std, avg = np.std(test_target), np.mean(test_target)  # calculate standard diviation and mean value for renormalization
                        scale = StandardScaler()  # normalizer, normalize on standard diviation
                        train_param, test_param, train_target = scale.fit_transform(train_param), scale.fit_transform(
                            test_param), scale.fit_transform(train_target)  # normalization on sample features and training target

                        tdnn = Multi_TDNN(filter_sizes=tdl)  #initialize a tdnn model
                        tdnn.build(input_dim=idim, hidden_dim=h, output_dim=odim, activations=act, drop_rate=0, \
                                   kernel_regularizer=None, act_regularizer=None)  #assign input and output dimension, hidden layer structure,
                                                                                    #activation functions, dropout rate, regularizers
                        tdnn.fit(train_param, train_target, batch_size=(len(train_target)-max(tdnn.TDL)+1), epochs=e) # fit the model
                        tdnn.Model.save("model.h5")  # save the model into a hdf5 file
                        #tdnn.Model.save_weights("model_weights.h5")

                        test_target_pred =tdnn.predict(test_param)  #generate prediction results(normalized)
                        test_target_pred = test_target_pred*std+avg #renormalization
                        test_target = test_target[max(tdnn.TDL)-1:]

                        """evaluate the results"""
                        rmse_score = np.sqrt(metrics.mean_squared_error(test_target, test_target_pred))
                        mae_score = dp.MAE(test_target, test_target_pred)
                        mbe_score = dp.MBE(test_target, test_target_pred)
                        relative_error = dp.relativeError(test_target, test_target_pred)
                        print "hiddim:",h, "TDL:", tdl, "epoch:", e, "activation:", act
                        print "scores:", rmse_score, mae_score, mbe_score, relative_error
                        break
                    break
                break
            break
        break

    return test_target_pred

    CI = dp.confidence_interval(test_param[max(tdnn.TDL)-1:,:], test_target_pred)  #calculate confidence intervals
    err = 0
    err_value = []
    for i in range(len(test_target)):   #count number of golden data that are not within confidence intervals
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
    #y3 = np.array(y[:, 2])  # low
    #y4 = np.array(y[:, 3])  # high
    plt.plot(x, y1, color='red', lw=1, label = 'Golden', alpha = 0.75)
    plt.plot(x, y2, color='blue', lw=1, label = 'TDNN', alpha = 0.75)
    #plt.plot(x, y3, color='blue', lw=0.5, linestyle='--', alpha=0.25)
    #plt.plot(x, y4, color='blue', lw=0.5, linestyle='--',alpha=0.25)
    #plt.fill_between(x, y3, y4, color='blue', alpha=0.25)

    plt.legend()
    plt.show()

tdnn()
