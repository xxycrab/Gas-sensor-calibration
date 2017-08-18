import os
import sys
import numpy
import theano
import copy
import theano.tensor as T
from logistic_sgd import load_data

class adaboost_m1():
    def __init__(self,Dist=None, n_y=10 , Hyp_out=None , train_y=None):
       
        self.Dist = Dist
        #n_y is the length of classify, for example , the mnist dataset
        # has the 10 length.
        self.n_y = n_y
        #Hyp_out is the matrix, the shape is (m rows, 10) for the mnist
        # dataset, it is the result of the neural network.
        self.train_y = train_y

        self.Hyp_out = Hyp_out

    def get_coe1(self):

        PV = numpy.ones((self.train_y.shape[0],self.n_y),dtype = numpy.float32)
        max_D = max(self.Dist)
        
        for i in xrange(self.train_y.shape[0]):
            PV[i] = self.Dist[i] / max_D

        true_out = numpy.zeros((self.train_y.shape[0],self.n_y),dtype = numpy.float32)
        for i in xrange(self.train_y.shape[0]):
            true_out[i][self.train_y[i]] = 1.

        return PV,true_out
   

    def m1_once_boost(self):
        #calculate the error in the distribute 
        error = 0.0
        tem = 0.0
        j = 0
        k = 0
        for i in xrange(self.train_y.shape[0]):
            if numpy.argmax(self.Hyp_out[i]) != self.train_y[i]:
                j = j+1
                error = error + self.Dist[i]
            if numpy.argmax(self.Hyp_out[i]) == self.train_y[i]:
                k = k+1
                tem = tem + self.Dist[i]
        print "                                                                               "
        print "Boost error :" + str(error)

        # beta is the weight value in different classifer
        #alph = 0.5 * numpy.log( (1. - error) / error)
        beta = error / (1. - error) 
        #update the Dist
        new_Dist = copy.copy(self.Dist)
        for i in xrange(self.train_y.shape[0]):
            if numpy.argmax(self.Hyp_out[i]) == self.train_y[i]:
                new_Dist[i] = (self.Dist[i] * beta)
            if numpy.argmax(self.Hyp_out[i]) != self.train_y[i]:
                new_Dist[i] = (self.Dist[i] * 1.)
        N = numpy.sum(new_Dist)
        for i in xrange(self.train_y.shape[0]):
            new_Dist[i] = new_Dist[i] / N

        return  beta, new_Dist


def init_adaboost_m1(n_train_y, n_y):
    '''
    # this function is to init the first iter weight.
    #
    # the correct element is not need to be inited, which can not be used 
    # in the adboost.m2 algorithm.

    '''
    #print n_train_y
    tem = 1.0 / n_train_y    
    Dist = tem * numpy.ones((n_train_y,),dtype = numpy.float32)
    return Dist

