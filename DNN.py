from sklearn import decomposition
import sys
import keras
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras import regularizers
from keras.layers import advanced_activations
from keras.optimizers import RMSprop
from keras.utils import np_utils
from sklearn import preprocessing
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Merge

def tansig(X):
    return 2/(1+np.exp(-2*X))-1

get_custom_objects().update({'custom_activation': Activation(tansig)})

class DNN:
    def __init__(self, rate=0, hw=5):
        self.model = Sequential()
        self.dropout_rate = rate
        self.hidden_width = hw

    def fit(self, feature, target, epochs = 4000):
        feature = preprocessing.scale(feature)
        target = preprocessing.scale(target)

        model = self.model
        hw = self.hidden_width
        ndim = len(feature[0])
        ldim = 1
        batch_size = len(feature)

        model.add(Dense(units = hw, input_dim= ndim, activation=tansig, init='normal',use_bias=True, bias_initializer = 'zeros'))
        model.add(Dense(units= hw, activation=tansig, init='normal',use_bias=True, bias_initializer = 'zeros',
                        kernel_regularizer=regularizers.l2(0.0)))
        model.add(Dense(ldim))

        sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
        model.fit(feature, target, epochs=epochs, batch_size= batch_size, verbose=0)
        self.model = model
        return model

    def predict(self, testset):
        testset = preprocessing.scale(testset)
        model = self.model
        pred = model.predict(testset)
        return pred

class TDNN:
    def __init__(self, filter_size = 12):
        self.model = Sequential()
        self.TDL = filter_size

    def TDdata(self, data):
        TDdata = data[0:len(data)-(self.TDL-1),:]
        for i in range(1, self.TDL):
            TDdata = np.concatenate((TDdata, data[i:len(data)-(self.TDL-1)+i,: ]), axis=1)
        return TDdata

    def build(self, input_dim, hidden_dim, output_dim, activations):
        hw = hidden_dim
        idim =input_dim * self.TDL
        odim = output_dim
        act = activations

        for i in range(len(hw)):
            if i == 0:
                self.model.add(Dense(input_dim=idim, units=hw[i], activation=act[i], init='normal'))
            else:
                self.model.add(Dense(units=hw[i], activation=act[i], init='normal'))
        self.model.add(Dense(units=odim, activation='linear', init='normal'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
        return self.model

    def fit(self, feature, target, batch_size, epochs = 1000):
        model = self.model
        feature = self.TDdata(feature)
        target = target[self.TDL-1:]
        model.fit(feature, target, epochs=epochs, batch_size= batch_size, verbose=0)
        self.model = model
        return model

    def predict(self, testset):
        testset = self.TDdata(testset)
        model = self.model
        pred = model.predict(testset)
        return pred

class Multi_TDNN:
    def __init__(self, filter_sizes=[3,6,9]):
        self.TDL = filter_sizes

    def TDdata(self, data, t):
        TDdata = data[0:len(data) - (t - 1), :]
        for i in range(1, t):
            TDdata = np.concatenate((TDdata, data[i:len(data) - (t - 1) + i, :]), axis=1)
        return TDdata

    def build(self, input_dim, hidden_dim, output_dim, activations):
        hw = hidden_dim
        odim = output_dim
        idim = input_dim
        act = activations
        models = []

        for t in self.TDL:
            models.append(Sequential())
            k = self.TDL.index(t)
            for i in range(len(hw)):
                if i == 0:
                    models[k].add(Dense(input_dim=idim*t, units=hw[i], activation=act[i], init='normal'))
                else:
                    models[k].add(Dense(units=hw[i], activation=act[i], init='normal'))
            models[k].add(Dense(units=odim, activation='linear', init='normal'))

        self.Model = Sequential()
        self.Model.add(Merge(models, mode='concat'))
        self.Model.add(Dense(units=odim, activation='linear'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.Model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
        return self.Model

    def fit(self, feature, target, batch_size, epochs=1000):
        Features = []
        for t in self.TDL:
            Features.append(self.TDdata(feature, t = t)[max(self.TDL)-t:,:])

        target = target[max(self.TDL)-1:]
        self.Model.fit(Features, target, epochs=epochs, batch_size=batch_size, verbose=0)
        return self.Model

    def predict(self, testset):
        Test = []
        for t in self.TDL:
            Test.append(self.TDdata(testset, t=t)[max(self.TDL)-t:,:])
        pred = self.Model.predict(Test)
        return pred
