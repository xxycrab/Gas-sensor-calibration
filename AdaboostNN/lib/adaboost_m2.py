import os
import sys
import numpy
import theano
import copy
import theano.tensor as T
from logistic_sgd import load_data

class adaboost_m2():
    def __init__(self,W=None , n_y=10 , Hyp_out=None , train_y=None):
        #W is a matrix, which has the shape (m rows, n column), erery
        # element in the matrix indicates the weight of element
        self.W = W
        #n_y is the length of classify, for example , the mnist dataset
        # has the 10 length.
        self.n_y = n_y
        #Hyp_out is the matrix, the shape is (m rows, 10) for the mnist
        # dataset, it is the result of the neural network.
        self.train_y = train_y

        self.Hyp_out = Hyp_out

    def get_coe2(self):
        V = copy.copy(self.W)
        PV =copy.copy(self.W)

        max_y = numpy.ones((self.train_y.shape[0],),dtype = numpy.float32)
        # GET THE V
        for i in xrange(self.train_y.shape[0]):
            tem = self.W[i][self.train_y[i]]
            self.W[i][self.train_y[i]] = 0.
            max_y[i] = max(self.W[i])
            V[i] = V[i] / max_y[i]
            V[i][self.train_y[i]] = 1.
            self.W[i][self.train_y[i]] = tem
        
        # GET THE P = Dist
        P  = numpy.ones((self.train_y.shape[0],),dtype = numpy.float32)
        sum_W = 0.

        for i in xrange(self.train_y.shape[0]):
            sum_w = numpy.sum(self.W[i]) - self.W[i][self.train_y[i]]
            sum_W =sum_W + sum_w

        for i in xrange(self.train_y.shape[0]):
            sum_w = numpy.sum(self.W[i]) - self.W[i][self.train_y[i]]
            P[i] = sum_w / sum_W
        max_P = max(P)
        print P
        for i in xrange(self.train_y.shape[0]):
            PV[i] = 0.5 * (P[i] / max_P) * V[i] 
        true_out = numpy.zeros((self.train_y.shape[0],self.n_y),dtype = numpy.float32)
        for i in xrange(self.train_y.shape[0]):
            true_out[i][self.train_y[i]] = 1.
        print PV[0]
        print true_out[0]
        return PV,true_out


    def m2_once_boost(self):
        #calculate the error in the distribute 
        error = 0.0
        query = numpy.ones((self.train_y.shape[0],self.n_y),dtype = numpy.float32)
        Dist  = numpy.ones((self.train_y.shape[0],        ),dtype = numpy.float32)
        sum_W = 0.

        # GET THE QUERY
        for i in xrange(self.train_y.shape[0]):
            sum_w = numpy.sum(self.W[i]) - self.W[i][self.train_y[i]]
            sum_W =sum_W + sum_w
            query[i] = self.W[i] / sum_w

        # GET THE DIST
        for i in xrange(self.train_y.shape[0]):
            sum_w = numpy.sum(self.W[i]) - self.W[i][self.train_y[i]]
            Dist[i] = sum_w / sum_W

        # GET THE ERROR
        for i in xrange(self.train_y.shape[0]):
            index = self.train_y[i]
            error = error + 0.5 * Dist[i] *             \
                    (1. - self.Hyp_out[i][index] +           \
                    (numpy.sum(query[i] * self.Hyp_out[i]) - \
                     query[i][index] * self.Hyp_out[i][index]))
        print "                                                                               "
        print "Boost error :" + str(error)

        # BETA IS THE WEIGHT VALUE IN DIFFERENT CLASSIFER.
        beta = error / (1. - error) 

        # UPDATE W
        new_W = copy.copy(self.W)

        for i in xrange(self.train_y.shape[0]):
            for j in xrange(self.n_y):
                if j != self.train_y[i]:
                    new_W[i][j] = (self.W[i][j] * (beta ** (0.5 * (1 +  \
                           self.Hyp_out[i][self.train_y[i]]  -          \
                           self.Hyp_out[i][j]))))
                if j == self.train_y[i]:
                    new_W[i][j] = 0.

        return  beta, new_W


def init_adaboost_m2(n_train_y, n_y):
    '''
    # this function is to init the first iter weight.
    #
    # the correct element is not need to be inited, which can not be used 
    # in the adboost.m2 algorithm.

    '''
    tem = (1.0 / n_train_y ) / float(n_y - 1)   
    W = tem * numpy.ones((n_train_y,n_y),dtype = numpy.float32)
    return W

