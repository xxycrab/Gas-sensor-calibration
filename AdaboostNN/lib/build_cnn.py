"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import os
import sys
import time
import cPickle
import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from config_parser import read_parser

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape,                           \
                                         activation=None, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        lin_output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        self.output = (
            T.maximum(0., lin_output) if activation is None
            else activation(lin_output)
        )
        

        # store parameters of this layer
        self.params = [self.W, self.b]


def cnn(pre_run,kind, PV, true_out ,learning_rate=0.1, n_epochs=200,
        datasets='mnist.pkl.gz',batch_size=100,
        path="", name="",input_layer={},
        hidden={},ConvPool={},out_layer={}):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)


    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    array_PV = PV
    PV = theano.shared(value=0.5*numpy.ones(PV.shape,dtype="float32"),borrow=True) 
    
    true_out = theano.shared(value=true_out,borrow=True)
    assert PV.get_value().shape[0] == train_set_x.get_value().shape[0]
    z1 = T.matrix('z1')
    z2 = T.matrix('z2')

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, int(input_layer['channel']), 
                                          int(input_layer['width']),
                                          int(input_layer['height'])))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    #
    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    #
    # CP is a list storing the ConvPool layer.
    CP = []
    for i in xrange(len(ConvPool)):
        tem = 'ConvPool'+ str(i)
        if i == 0:
            activation = None
            if int(ConvPool[tem]['activation']) == 1:
                activation = T.nnet.sigmoid
            if int(ConvPool[tem]['activation']) == 2:
                activation = T.tanh
            CP.append(LeNetConvPoolLayer( rng,
                                          activation = activation,
                                          input=layer0_input,
                                          image_shape=( batch_size, 
                                                        int(ConvPool[tem]['channel']), 
                                                        int(ConvPool[tem]['width']),
                                                        int(ConvPool[tem]['height'])),
                                          filter_shape=(int(ConvPool[tem]['filters']), 
                                                        int(ConvPool[tem]['channel']), 
                                                        int(ConvPool[tem]['filter_width']), 
                                                        int(ConvPool[tem]['filter_height'])),
                                          poolsize=(    int(ConvPool[tem]['pool_width']), 
                                                        int(ConvPool[tem]['pool_height']))))
        if i != 0:
            activation = None
            if int(ConvPool[tem]['activation']) == 1:
                activation = T.nnet.sigmoid
            if int(ConvPool[tem]['activation']) == 2:
                activation = T.tanh
            CP.append(LeNetConvPoolLayer( rng,
                                          activation = activation,
                                          input=CP[-1].output,
                                          image_shape=( batch_size, 
                                                        int(ConvPool[tem]['channel']), 
                                                        int(ConvPool[tem]['width']),
                                                        int(ConvPool[tem]['height'])),
                                          filter_shape=(int(ConvPool[tem]['filters']), 
                                                        int(ConvPool[tem]['channel']), 
                                                        int(ConvPool[tem]['filter_width']), 
                                                        int(ConvPool[tem]['filter_height'])),
                                          poolsize=(    int(ConvPool[tem]['pool_width']), 
                                                        int(ConvPool[tem]['pool_height']))))


    ConvPool_output = CP[-1].output.flatten(2)

    # construct a fully-connected sigmoidal layer
    # HL is a list storing the Hidden layer.
    HL = []
    for i in xrange(len(hidden)):
        ite = len(ConvPool) + i
        tem = 'hidden_layer_'+ str(ite)

        if ite == len(ConvPool):
            activation = None
            if int(hidden[tem]['activation']) == 1:
                activation = T.nnet.sigmoid
            if int(hidden[tem]['activation']) == 2:
                activation = T.tanh
            HL.append( HiddenLayer(rng,
                                   input=ConvPool_output,
                                   n_in =int(hidden[tem]['n_in']),
                                   n_out=int(hidden[tem]['n_out']),
                                   activation=activation))

        if ite != len(ConvPool):
            activation = None
            if int(hidden[tem]['activation']) == 1:
                activation = T.nnet.sigmoid
            if int(hidden[tem]['activation']) == 2:
                activation = T.tanh
            HL.append( HiddenLayer(rng,
                                   input=HL[-1].output,
                                   n_in =int(hidden[tem]['n_in']),
                                   n_out=int(hidden[tem]['n_out']),
                                   activation=activation))

    hidden_output = HL[-1].output

    # classify the values of the fully-connected output layer
    OutLayer = HiddenLayer(rng=rng,                        \
                           input=hidden_output,            \
                           n_in=int(out_layer['n_in']),    \
                           n_out=int(out_layer['n_out']),  \
                           activation=T.nnet.sigmoid,      \
                           kind=2)

    # the cost we minimize during training is the NLL of the model
    cost = OutLayer.sq_loss(z1,z2)
    y_x = OutLayer.output

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        [OutLayer.errors(y),y_x],
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        OutLayer.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = OutLayer.params 
    tem = len(HL)
    for i in xrange(len(HL)):
        params += HL[tem-1].params
        tem = tem -1
    tem = len(CP)
    for i in xrange(len(CP)):
        params += CP[tem-1].params
        tem = tem -1   

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        [ cost,y_x],
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],            
            z1:         PV[index * batch_size: (index + 1) * batch_size],
            z2:   true_out[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False
    Hpy_out = []

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1

        if epoch == pre_run:
            PV.set_value(array_PV)

        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            
            tem = train_model(minibatch_index)
            cost_ij = tem[0]
            if epoch == n_epochs:
                Hpy_out.append(tem[1])

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)[0]
                        for i in xrange(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            #if patience <= iter:
            #    done_looping = True
            #    break
        

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    # save the test result and return the train Hpy after train finished.
    test_out = []
    for minibatch_index in xrange(n_test_batches):
        tem = test_model(minibatch_index)
        test_out.append(tem[1])

    test_tem = numpy.asarray(test_out).reshape((n_test_batches * batch_size,  \
                                                        int(out_layer['n_out'])))
    cPickle.dump(test_tem,open("./config-example/test_tem/"+name+".pkl","wb"))
    return Hpy_out

def build_cnn(kind, PV,true_out ,path, datasets, name):
    conf          = read_parser(path)
    learning_rate = float(    conf['parm']['learning_rate'])
    n_epochs      = int(      conf['parm']['n_epochs'])
    batch_size    = int(      conf['parm']['batch_size'])
    pre_run       = int(   conf['parm']['pre_run'])
    input_layer   =           conf['input_layer']
    ConvPool      = {}
    hidden        = {}
    out_layer      =           conf['out_layer']

    for i in xrange(len(conf)):
        tem = 'ConvPool' + str(i)
        if tem in conf.keys():
            ConvPool[tem] = conf[tem]

    for i in xrange(len(conf)):
        tem = 'hidden_layer_' + str(i)
        if tem in conf.keys():
            hidden[tem]  = conf[tem]

    out = cnn( pre_run = pre_run,
               kind = kind,                    \
               PV =PV,                         \
               true_out = true_out,            \
               learning_rate = learning_rate,  \
               n_epochs = n_epochs,            \
               datasets =datasets,             \
               batch_size=batch_size,          \
               path = path ,                   \
               name = name,                    \
               input_layer=input_layer,        \
               hidden=hidden,                  \
               ConvPool=ConvPool,              \
               out_layer=out_layer)
    return out

    
if __name__ == '__main__':
    pass

def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
