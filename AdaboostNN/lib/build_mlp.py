"""
A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""
__docformat__ = 'restructedtext en'


import os
import sys
import time
import cPickle
import numpy

import theano
import theano.tensor as T
from adaboost_m2 import adaboost_m2, init_adaboost_m2
from mlp import HiddenLayer
from logistic_sgd import LogisticRegression, load_data
from config_parser import read_parser


def test_mlp(pre_run,kind, PV,true_out, name='',learning_rate=0.01, L1_reg=0.00,L2_reg=0.0001, \
             n_epochs=3,datasets='mnist.pkl.gz', batch_size=100, n_in=784,n_hidden=[500],      \
             h_activation = [0] , n_out=10):

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    array_PV = PV
    PV = theano.shared(value=0.5*numpy.ones(PV.shape,dtype="float32"),borrow=True)  

    n_train_x = train_set_x.get_value(borrow=True).shape[0]

    true_out = theano.shared(value=true_out,borrow=True)
    assert PV.get_value().shape[0] == train_set_x.get_value().shape[0]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    z1 = T.matrix('z1')
    z2 = T.matrix('z2')
    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    MLP = []
    for i in xrange(len(n_hidden) + 1):
        if i == 0:
            activation = None
            if h_activation == 1:
                activation = T.nnet.sigmoid
            MLP.append( HiddenLayer(rng=rng,input=x,n_in=n_in,        \
                        n_out=n_hidden[i],activation=activation))

        if i > 0 and i < len(n_hidden):
            activation = None
            if h_activation == 1:
                activation = T.nnet.sigmoid
            MLP.append( HiddenLayer(rng=rng,input=MLP[i-1].output,    \
                        n_in=n_hidden[i-1], n_out=n_hidden[i],        \
                        activation=activation))

        if i == len(n_hidden):
            MLP.append( HiddenLayer(rng=rng,input=MLP[i-1].output,    \
                        n_in=n_hidden[i-1], n_out=n_out,              \
                        activation=T.nnet.sigmoid,kind=2))  
    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically

    cost  = MLP[-1].sq_loss(z1,z2)

    params = []
    for i in xrange(len(MLP)):
        cost = cost +  L1_reg * abs(MLP[i].W).sum() + \
                       L2_reg * abs(MLP[i].W ** 2).sum()
        params = params + MLP[i].params

    
    y_x = MLP[-1].output
    # end-snippet-4

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=[MLP[-1].errors(y) , y_x],
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=MLP[-1].errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # start-snippet-5
    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
    # same length, zip generates a list C of same size, where each element
    # is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=[cost,y_x],
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            z1:         PV[index * batch_size: (index + 1) * batch_size],
            z2:   true_out[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-5

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
    #boost_error = 1.
    epoch = 0
    done_looping = False
    Hpy_out = []

    while (epoch < n_epochs) and (not done_looping ):
        epoch = epoch + 1

        if epoch == pre_run:
            PV.set_value(array_PV)

        for minibatch_index in xrange(n_train_batches):

            tem = train_model(minibatch_index)
            minibatch_avg_cost = tem[0]
            # the output
            if  epoch ==n_epochs:
                Hpy_out.append(tem[1])

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i)[0] for i
                                   in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            #if patience <= iter:
            #    done_looping = True
            #    break
        '''
        #----------------------------------------------------------------------
        import copy
        hpy = copy.deepcopy(Hpy_out)
        if (kind == 1 and epoch % 10 == 0 ) and epoch>1:
            boost_error = 0.
            tem = numpy.asarray(hpy).reshape((n_train_x,n_out)) 
            for i in xrange(n_train_x):
                if numpy.argmax(tem[i]) != train_set_y[i]:
                    boost_error = boost_error + PV.get_value()[i][0] 
            print boost_error
        #---------------------------------------------------------------------- 
        '''
    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    # save the test result and return the train Hpy after train finished.
    test_out = []

    for minibatch_index in xrange(n_test_batches):
        tem = test_model(minibatch_index)
        test_out.append(tem[1])

    test_tem = numpy.asarray(test_out).reshape((n_test_batches * batch_size,n_out))
    cPickle.dump(test_tem,open("./config-example/test_tem/"+name+".pkl","wb")) 
    return Hpy_out


def build_mlps(kind, PV,true_out ,path, datasets, name):
    conf          = read_parser(path)
    learning_rate = float( conf['parm']['learning_rate'])
    L1_reg        = float( conf['parm']['l1_reg'])
    L2_reg        = float( conf['parm']['l2_reg'])
    n_epochs      = int(   conf['parm']['n_epochs'])
    batch_size    = int(   conf['parm']['batch_size'])
    pre_run       = int(   conf['parm']['pre_run'])
    n_in          = int(   conf['input_layer']['input_num'])
    n_hidden      = []
    h_activation  = []
    n_out         = int(   conf['out_layer']['out_num'])

    for i in xrange(len(conf)):
        tem = 'hidden_layer_' + str(i)
        if tem in conf.keys():
            n_hidden.append(int(conf[tem]['hidden_num']))
            h_activation.append(int(conf[tem]['activation']))

    out = test_mlp(pre_run = pre_run,               
                   kind = kind,                 \
                   PV =PV,                      \
                   true_out = true_out,         \
                   name = name,                 \
                   learning_rate=learning_rate, \
                   L1_reg=L1_reg,               \
                   L2_reg=L2_reg,               \
                   n_epochs=n_epochs,           \
                   datasets=datasets,           \
                   batch_size=batch_size,       \
                   n_hidden=n_hidden,           \
                   h_activation = h_activation, \
                   n_out=n_out)
    return out
    
if __name__ == '__main__':
    pass
