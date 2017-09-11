import dataprocess as dp
import numpy as np
import pandas as pd

def data_read():
    dataset  = pd.read_excel('F:\Python projects\gas-dataset-regression\data\AirQualityUCI.xlsx') #open file and read in dataset
    featureset = ['DT','PT08.S1(CO)', 'PT08.S2(NMHC)','T', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'RH']  #features chosen for training
    target_name = ['DT', 'CO(GT)'] #learning target
    feature = dp.get_features(dataset, featureset) #extract samples for learning from datasets
    target= dp.get_target(dataset, target_name) #extract learning target from raw dataset
    feature['DT'], target['DT'] = pd.to_datetime(feature['DT']), pd.to_datetime(target['DT'])
    feature = feature.set_index('DT')
    target = target.set_index('DT')  #reset time as index of data
    feature, target = dp.delMissing(feature, target)   #deal with missing data and transfer into ndarray
    #feature, target = dp.sort(feature, target, 'RH')   #sort by RH
    return feature, target  # type is ndarray

"""read files of dataset from UCSD, return samples for learning and the target"""
def data_read_new():
    dataset  = pd.read_csv('F:\Python projects\gas-dataset-regression\data\B4.csv') #open file and read in dataset
    featureset = ['datetime', 'CO_W', 'CO_A', 'RH_ref'] #features chosen for training
    target_name = ['datetime', 'NO2_ref'] #learning target
    feature = dp.geft_eatures(dataset, featureset)  #extract samples for learning from datasets
    target = dp.get_target(dataset, target_name) #extract learning target from raw dataset
    feature['datetime'], target['datetime'] = pd.to_datetime(feature['datetime']), pd.to_datetime(target['datetime'])
    feature = feature.set_index('datetime')
    target = target.set_index('datetime') #reset time as index of data
    feature, target = np.array(feature), np.array(target) #change into type of ndarray
    return feature, target # type is ndarray