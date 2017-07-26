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
from keras.utils.generic_utils import get_custom_objects

def tansig(X):
    return 2/(1+np.exp(-2*X))-1

get_custom_objects().update({'custom_activation': Activation(tansig)})

class DNN:
    def __init__(self, rate=0, hw=5):
        self.model = Sequential()
        self.dropout_rate = rate
        self.hidden_width = hw

    def train(self, feature, target, test_feature, test_target, layer = 3):
        target_data = np.concatenate((target, test_target))
        feature = preprocessing.scale(feature)
        #target = preprocessing.scale(target)
        test_feature= preprocessing.scale(test_feature)

        model = self.model
        hidden_width = self.hidden_width
        dropout_rate = self.dropout_rate
        ndim = len(feature[0])
        ldim = 1
        batch_size = len(feature)

        model.add(Dense(units = hidden_width, input_dim= ndim, activation=tansig, init = 'normal'))
        #model.add(Dropout(dropout_rate))
        model.add(Dense(units= hidden_width, input_dim=ndim, activation=tansig, init='normal'))
        #model.add(Dropout(dropout_rate))
        #model.add(Dense(hidden_width, activation='sigmoid'))
        #model.add(Dropout(dropout_rate))
        model.add(Dense(ldim))

        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
        model.fit(feature, target, epochs=8000, batch_size= batch_size, verbose=0)
        pred = model.predict(test_feature)
        return pred

    def prediction(self, testset):
        model = self.model
        pred = model.predict(testset)
        return pred