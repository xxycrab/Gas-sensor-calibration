from sklearn import preprocessing
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Merge
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasRegressor
import matplotlib.pyplot as plt
from DNN import *
import warnings

warnings.filterwarnings('ignore')

"""read in dataset"""
def data_read():
    dataset  = pd.read_excel('F:\Python projects\gas-dataset-regression\data\AirQualityUCI.xlsx')

    '''prepare data for training and test.'''
    featureset = ['DT','PT08.S1(CO)', 'PT08.S2(NMHC)','T', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'RH']  #features chosen for training
    target = ['DT', 'NO2(GT)']  #learning target
    feature_CO = dp.features(dataset, featureset)  #extract samples for learning from datasets
    target_CO = dp.target(dataset, target)  # extract learning target from raw dataset
    feature_CO['DT'], target_CO['DT'] = pd.to_datetime(feature_CO['DT']), pd.to_datetime(target_CO['DT'])
    feature_CO = feature_CO.set_index('DT') #reset time as index of data
    target_CO = target_CO.set_index('DT')
    feature_CO, target_CO = dp.delMissing(feature_CO, target_CO)   #deal with missing data
    #feature_CO, target_CO = dp.sort(feature_CO, target_CO, 'RH')   #sort by RH
    return feature_CO, target_CO

def TDdata(data, tdl):
    TDdata = data[0:len(data) - (tdl - 1), :]
    for i in range(1, tdl):
        TDdata = np.concatenate((TDdata, data[i:len(data) - (tdl - 1) + i, :]), axis=1)
    return TDdata

def tansig(x):
    return 2/(1+np.exp(-2*x))-1

def model_build(input_dim, hidden_dim, output_dim, activations):
    model = Sequential()
    idim = input_dim
    odim = output_dim
    hw = hidden_dim
    act = activations

    for i in range(len(hw)):
        if i==0:
            model.add(Dense(input_dim=idim, units=hw[i], activation=act[i], init='normal'))
        else:
            model.add(Dense(units=hw[i], activation=act[i], init='normal'))
    model.add(Dense(units=odim, activation='linear', init='normal'))

    sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    return model

def tdnn_build(input_dim, hidden_dim, output_dim, activation, filter_size):
    hw = hidden_dim
    odim = output_dim
    idim = input_dim
    act = activation
    TDL = filter_size
    models = []

    for t in TDL:
        models.append(Sequential())
        k = TDL.index(t)
        for i in range(len(hw)):
            if i == 0:
                models[k].add(Dense(input_dim=idim * t, units=hw[i], activation=act[i], init='normal'))
            else:
                models[k].add(Dense(units=hw[i], activation=act[i], init='normal'))
        models[k].add(Dense(units=odim, activation='linear', init='normal'))

    Model = Sequential()
    Model.add(Merge(models, mode='concat'))
    Model.add(Dense(units=odim, activation='linear'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    Model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    return Model

def ada():
    feature, target = data_read()
    skf = sel.KFold(n_splits=5)
    param = np.array(feature)
    target = np.array(target)  # change into ndarray
    target = np.reshape(target, (len(target),))  # change into (len,) instead of (len, 1)

    #for train_index, test_index in skf.split(param, target):
    idim = feature.shape[1]
    outdim = 1
    hiddim = [10]
    act = [tansig]
    TDL = 15

    #tdnn = Multi_TDNN(filter_sizes=TDL)
    base = KerasRegressor(build_fn=model_build, input_dim=idim*TDL, hidden_dim=hiddim, output_dim=outdim,activations=act,\
                          verbose=0, epochs = 500)
    ada = AdaBoostRegressor(base_estimator= base, n_estimators=100, learning_rate=0.1, loss = 'square')

    train_param, test_param = param[0:2000, :], param[2000:3000, :]
    train_target, test_target = target[0:2000], target[2000:3000]
    avg, std = np.mean(test_target), np.std(test_target)
    train_param, test_param = preprocessing.scale(train_param), preprocessing.scale(test_param)
    train_target = preprocessing.scale(train_target)

    train_param, test_param = TDdata(data = train_param, tdl=TDL), TDdata(data=test_param, tdl=TDL)
    train_target = train_target[TDL-1:]
    ada.fit(train_param, train_target)

    test_target_pred = ada.predict(test_param)
    test_target_pred = test_target_pred*std+avg
    test_target = test_target[TDL-1:]

    # estimate the performance of regression
    test_target_pred = np.reshape(test_target_pred, (len(test_target_pred),))
    mae_score = dp.MAE(test_target, test_target_pred)
    mbe_score = dp.MBE(test_target, test_target_pred)
    rmse_score = np.sqrt(metrics.mean_squared_error(test_target, test_target_pred))
    relative_error = dp.relativeError(test_target, test_target_pred)
    print rmse_score,mae_score,mbe_score,relative_error

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

    print(RMSE/5.0,MAE/5.0,MBE/5.0,RE/5.0)
    return test_target_pred

ada()