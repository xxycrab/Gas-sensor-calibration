import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, regularizers
from keras.optimizers import SGD
from keras import regularizers
from keras.layers import advanced_activations
from keras.optimizers import RMSprop
from sklearn import preprocessing
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Merge
from keras import backend as K
from keras.callbacks import EarlyStopping

"""tansig activation function"""
def tansig(X):
    return 2/(1+np.exp(-2*X))-1
get_custom_objects().update({'custom_activation': Activation(tansig)})  #make tansig function able to be called by keras

"""basic BPNN model. The keras core layre "Dense" layer is a full-connected layer with backpropagation"""
class DNN:
    def __init__(self):
        self.Model = Sequential() #initialize a keras basic neural network framework

    def build(self, input_dim, hidden_dim, output_dim, activations, drop_rate = 0, \
              kernel_regularizer = None, act_regularizer = None):
        hw = hidden_dim
        idim =input_dim
        odim = output_dim  #hidden layer size, output and input dimension
        act = activations  #activation functions

        for i in range(len(hw)):
            if act[i] == 'None':  #deal with 'None' activation function
                activation = None
            else:
                activation = act[i]
            if i == 0: # for the first layer, input dimension should be specified
                self.Model.add(Dense(input_dim=idim, units=hw[i], activation=activation, init='normal',\
                                     kernel_regularizer=kernel_regularizer, activity_regularizer=act_regularizer))
            else:
                self.model.add(Dense(units=hw[i], activation=activation, init='normal'))
            if drop_rate != 0: self.Model.add(Dropout(rate=drop_rate))  #if dropout rate is not 0, add dropout layer
            else:continue
        self.Model.add(Dense(units=odim, activation='linear', init='normal'))  #output layer

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)  #optimizer
        self.Model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])  #compile the model
        return self.Model

    """fit the model with training samples"""
    def fit(self, feature, target, batch_size, epochs = 1000, verbose = 0):
        self.Model.fit(feature, target, epochs=epochs, batch_size= batch_size, verbose=verbose)
        return self.Model

    """generate prediction results"""
    def predict(self, testset):
        pred = self.Model.predict(testset)
        return pred

class TDNN:
    def __init__(self, filter_size = 12):
        self.Model = Sequential()  #initialize a keras basic neural network framework
        self.TDL = filter_size #the tdl of TDNN

    """initialize input of the network. The original input shape is (dataset_size, feature_number)"""
    """The transferred input shape is (dataset_size-TDL+1, feature_number*TDL)"""
    def TDdata(self, data):
        TDdata = data[0:len(data)-(self.TDL-1),:]
        for i in range(1, self.TDL):
            TDdata = np.concatenate((TDdata, data[i:len(data)-(self.TDL-1)+i,: ]), axis=1)
        return TDdata

    def build(self, input_dim, hidden_dim, output_dim, activations, drop_rate = 0, \
              kernel_regularizer=None, act_regularizer=None):
        hw = hidden_dim
        idim =input_dim * self.TDL
        odim = output_dim   #hidden layer size, output and input dimension
        act = activations  #activation functions

        for i in range(len(hw)):   #build each layer of the network
            if act[i] == 'None':  #deal with 'None' activation function
                activation = None
            else:
                activation = act[i]
            if i == 0: # for the first layer, input dimension should be specified
                self.Model.add(Dense(input_dim=idim, units=hw[i], activation=activation, init='normal',\
                                     kernel_regularizer=kernel_regularizer, activity_regularizer=act_regularizer))
            else:
                self.Model.add(Dense(units=hw[i], activation=activation, init='normal',\
                                     kernel_regularizer=kernel_regularizer, activity_regularizer=act_regularizer))
            if drop_rate != 0: self.Model.add(Dropout(rate=drop_rate)) #if dropout rate is not 0, add a dropour layer
            else:continue
        self.Model.add(Dense(units=odim, activation='linear', init='normal')) #ouput layer

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)  #set optimizer
        self.Model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy']) #compile model
        return self.Model

    """fit the model with training samples"""
    def fit(self, feature, target, batch_size, epochs = 1000):
        feature = self.TDdata(feature)
        target = target[self.TDL-1:]
        self.Model.fit(feature, target, epochs=epochs, batch_size= batch_size, verbose=0)
        return self.Model

    """generate prediction results"""
    def predict(self, testset):
        testset = self.TDdata(testset)
        pred = self.Model.predict(testset)
        return pred

class Multi_TDNN:
    def __init__(self, filter_sizes=[3,6,9]):
        self.Model = Sequential()  #initialize a keras basic neural network framework
        self.TDL = filter_sizes  #the tdl of TDNN

    """initialize input of the network. The original input shape is (dataset_size, feature_number)"""
    """The transferred input shape is (dataset_size-TDL+1, feature_number*TDL)"""
    def TDdata(self, data, tdl):
        TDdata = data[0:len(data) - (tdl - 1), :]
        for i in range(1, tdl):
            TDdata = np.concatenate((TDdata, data[i:len(data) - (tdl - 1) + i, :]), axis=1)
        return TDdata

    """build the model"""
    def build(self, input_dim, hidden_dim, output_dim, activations, drop_rate=0, kernel_regularizer = None, act_regularizer = None):
        hw, odim, idim = hidden_dim, output_dim, input_dim #hidden layer size, output and input dimension
        act = activations  #activation functions
        rate = drop_rate  #dropout rate

        if len(self.TDL) == 1:  #if only one TDL, build a single TDNN
            return self.build_single(input_dim = idim, hidden_dim = hw, output_dim= odim, activations = act, drop_rate = rate,\
                                kernel_regularizer = kernel_regularizer, act_regularizer = act_regularizer)
        else:  #if given several TDL values, build a Multi-TDNN
            return self.build_multi(input_dim = idim, hidden_dim = hw, output_dim= odim, activations = act, drop_rate = rate,\
                                kernel_regularizer = kernel_regularizer, act_regularizer = act_regularizer)

    """build a single TDNN"""
    def build_single(self, input_dim, hidden_dim, output_dim, activations, drop_rate=0,\
                     kernel_regularizer = None, act_regularizer = None):
        hw, odim, idim = hidden_dim, output_dim, input_dim
        act = activations
        t = self.TDL[0]

        for i in range(len(hw)):  #build each layer of the network
            if act[i] == 'None':  #deal with the 'None' activation function
                activation = None
            else:
                activation = act[i]
            if i == 0:   # for the first layer, input dimension should be specified
                self.Model.add(Dense(input_dim=idim*t, units=hw[i], activation=activation, init='normal',\
                                    kernel_regularizer= kernel_regularizer, activity_regularizer= act_regularizer))
            else:
                self.Model.add(Dense(units=hw[i], activation=activation, init='normal', \
                                    kernel_regularizer= kernel_regularizer, activity_regularizer= act_regularizer))
            if drop_rate != 0: self.Model.add(Dropout(rate=drop_rate))  # if dropout rate is not 0, add a dropout layer
            else:continue
        output_layer = Dense(units = odim, activation = 'linear', init = 'normal')  #add an output layer
        self.Model.add(output_layer)

        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)  #optimizer
        self.Model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])  # compile the model
        return self.Model

    """build a Multi-TDNN, the input shape should be (TDL, dataset_size-TDL+1, feature_number)"""
    def build_multi(self, input_dim, hidden_dim, output_dim, activations, drop_rate, kernel_regularizer = None, act_regularizer = None):
        hw, odim, idim = hidden_dim, output_dim, input_dim
        act = activations
        models = []

        for t in self.TDL:  # for each TDL value, build a TDNN, all same as build_single except that compile is not needed
            models.append(Sequential())
            k = self.TDL.index(t)
            for i in range(len(hw)):
                if act[i] == 'None': activation = None
                else: activation = act[i]
                if i == 0:
                    models[k].add(Dense(input_dim=idim*t, units=hw[i], activation=activation, init='normal',\
                                        kernel_regularizer= kernel_regularizer, activity_regularizer= act_regularizer))
                else:
                    models[k].add(Dense(units=hw[i], activation=activation, init='normal', \
                                        kernel_regularizer= kernel_regularizer, activity_regularizer= act_regularizer))
                if drop_rate != 0: models[k].add(Dropout(rate=drop_rate))
                else: continue
            output_layer = Dense(units = odim, activation = 'linear', init = 'normal')
            models[k].add(output_layer)

        self.Model.add(Merge(models, mode='concat'))  #combine the output of each single TDNN into a tensor
        self.Model.add(Dense(units=odim, activation='linear'))  #a full connection layer whose input is the output of
                                                                # all single TDNN and the output is the final result

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.Model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
        return self.Model

    """fit the model with training samples"""
    def fit(self, feature, target, batch_size, epochs=1000, EarlyStop = False):
        Features = []
        for t in self.TDL:
            Features.append(self.TDdata(feature, tdl = t)[max(self.TDL)-t:,:])  #prepare input for each single TDNN
        target = target[max(self.TDL)-1:]

        early_stop = EarlyStopping(monitor='loss', patience=50) #set the early-stopping strategy
        if EarlyStop == True:
            self.Model.fit(Features, target, epochs=epochs, batch_size=batch_size, verbose=0, callbacks= [early_stop])
        else:
            self.Model.fit(Features, target, epochs=epochs, batch_size=batch_size, verbose=0)

        return self.Model

    """generate prediction results"""
    def predict(self, testset):
        Test = []
        for t in self.TDL:
            Test.append(self.TDdata(testset, tdl=t)[max(self.TDL)-t:,:])
        pred = self.Model.predict(Test)
        return pred
'''
class Ada_TDNN:
    def __init__(self,filter_size=[3,6,9]):
        self.TDL = filter_size

    def TDdata(self, data, tdl):
        TDdata = data[0:len(data) - (tdl - 1), :]
        for i in range(1, tdl):
            TDdata = np.concatenate((TDdata, data[i:len(data) - (tdl - 1) + i, :]), axis=1)
        return TDdata

    def build(self, input_dim, hidden_dim, output_dim, activation):
        hw = hidden_dim
        odim = output_dim
        idim = input_dim
        act = activation
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
            Features.append(self.TDdata(feature, tdl = t)[max(self.TDL)-t:,:])

        target = target[max(self.TDL)-1:]
        self.Model.fit(Features, target, epochs=epochs, batch_size=batch_size, verbose=0)
        return self.Model

    def predict(self, testset):
        Test = []
        for t in self.TDL:
            Test.append(self.TDdata(testset, tdl=t)[max(self.TDL)-t:,:])
        pred = self.Model.predict(Test)
        return pred
'''