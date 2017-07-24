from sklearn import decomposition
import sys
import keras
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.layers import advanced_activations
from keras.optimizers import RMSprop
from keras.utils import np_utils
from sklearn import preprocessing


class DNN:
    def __init__(self, rate=0, hw=5):
        self.model = Sequential()
        self.dropout_rate = rate
        self.hidden_width = hw
    def train(self, feature, target, test_feature, test_target, layer = 3):
        target_data = np.concatenate((target, test_target))
        std = target_data.std()
        mean = target_data.mean()
        feature = preprocessing.scale(feature)
        #target = preprocessing.scale(target)
        test_feature= preprocessing.scale(test_feature)

        model = self.model
        hidden_width = self.hidden_width
        dropout_rate = self.dropout_rate
        ndim = len(feature[0])
        ldim = 1
        batch_size = len(feature)

        model.add(Dense(units = hidden_width, input_dim= ndim, activation='elu', init = 'normal'))
        #model.add(Dropout(dropout_rate))
        #model.add(Dense(hidden_width, activation='linear'))
        #model.add(Dropout(dropout_rate))
        #model.add(Dense(hidden_width, activation='sigmoid'))
        #model.add(Dropout(dropout_rate))
        model.add(Dense(ldim))

        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])
        model.fit(feature, target, epochs=5000, batch_size= batch_size, verbose=0)
        pred = model.predict(test_feature)+0.21#*std+ mean
        return pred

    def prediction(self, testset):
        model = self.model
        pred = model.predict(testset)
        return pred