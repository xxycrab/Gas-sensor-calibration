import os
import sys
import time
import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from config_parser import read_parser


def build_cnn(path):
    conf          = read_parser(path)
    learning_rate = float(    conf['parm']['learning_rate'])
    n_epochs      = int(      conf['parm']['n_epochs'])
    batch_size    = int(      conf['parm']['batch_size'])
    input_layer   =           conf['input_layer']
    ConvPool      = {}
    hidden        = {}
    lr_layer      =           conf['lr_layer']

    for i in xrange(len(conf)):
        tem = 'ConvPool' + str(i)
        if tem in conf.keys():
            ConvPool[tem] = conf[tem]
    print ConvPool
    for i in xrange(len(conf)):
        tem = 'hidden_layer_' + str(i)
        if tem in conf.keys():
            hidden[tem]  = conf[tem]
    print hidden
    
    """
    out = evaluate_lenet5(path = path,                   \
                   k = k,                                \
                   name = name,                          \
                   finetune_lr=finetune_lr,              \
                   pretraining_epochs=pretraining_epochs,\
                   training_epochs=training_epochs,      \
                   pretrain_lr = pretrain_lr,            \
                   dataset=datasets,                     \
                   batch_size=batch_size,                \
                   hidden_layers_sizes=n_hidden ,        \
                   input_num=input_num,                  \
                   out_num=out_num)
    return out
    """


build_cnn("../config-example/model_cnn_2.cfg")
