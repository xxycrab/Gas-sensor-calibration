import os
import sys
import time
import numpy
import cPickle
import theano
from adaboost_m2 import adaboost_m2, init_adaboost_m2 
from adaboost_m1 import adaboost_m1, init_adaboost_m1
from logistic_sgd import LogisticRegression, load_data
from build_dbn import build_DBN
from build_cnn import build_cnn
from build_sda import build_sda

def test_sda_model():
    '''
    # this function is the single model, it can invoke one model.
    # whole frame.
    '''
    dataset='mnist.pkl.gz'
    datasets = load_data(dataset)

    # build cnn model
    path = '../config-example/model_sda_0.cfg'
    out = build_sda(path = path, datasets = datasets , name = 'model_sda_0')

def test_dbn_model():
    '''
    # this function is the single model, it can invoke one model.
    # whole frame.
    '''
    dataset='mnist.pkl.gz'
    datasets = load_data(dataset)

    # build cnn model
    path = '../config-example/model_dbn_1.cfg'
    out = build_DBN(path = path, datasets = datasets , name = 'model_dbn_1')

def test_cnn_model():
    '''
    # this function is the single model, it can invoke one model.
    # whole frame.
    '''
    dataset='mnist.pkl.gz'
    datasets = load_data(dataset)

    # build cnn model
    path = '../config-example/model_cnn_2.cfg'
    out = build_cnn(path = path, datasets = datasets , name = 'model_cnn_2')

test_sda_model()


