import os
import sys
import time
import numpy
import cPickle
import copy
import theano
from adaboost_m2 import adaboost_m2, init_adaboost_m2 
from adaboost_m1 import adaboost_m1, init_adaboost_m1
from config_parser import read_config_name, read_pkl_name,remove_pkl_file
from logistic_sgd import LogisticRegression, load_data
from build_mlp import build_mlps
from build_cnn import build_cnn
from build_sda import build_sda
from build_dbn import build_dbn

class Controller(object):
    def __init__(self):
        pass

    def test_m1_accurate(self,dirctory, beta, y):

        # function:
        # It will test the accuracy rate on the testdatas. The formula is:
        # H(x) = argmax ( sigma|1,T|(log 1/beta) * h(x,y))
        data = 0.0
        paths = read_pkl_name(dirctory)
        for i in xrange(len(os.listdir(dirctory))):
            data += numpy.log(1./beta[i])* cPickle.load(\
                    open(dirctory +"/"+ paths[str(i)][2],"rb"))
            print "----Model "+str(i) +" weight value :"+str(numpy.log(1./beta[i]))
        
        out = []
        for i in xrange(data.shape[0]):
            out.append(numpy.argmax(data[i]))

        assert y.shape[0] == len(out) 
        s = 0.
        for i in xrange(data.shape[0]):
            if y[i] == out[i]:
                s += 1.
        return s / float(data.shape[0])


    def test_m2_accurate(self,dirctory, beta, y):

        # function:
        # It will test the accuracy rate on the testdatas. The formula is:
        # H(x) = argmax ( sigma|1,T|(log 1/beta) * h(x,y))
        data = 0.0
        paths = read_pkl_name(dirctory)
        for i in xrange(len(os.listdir(dirctory))):
            data += numpy.log(1./beta[i])* cPickle.load(\
                    open(dirctory +"/"+ paths[str(i)][2],"rb"))
            print "----Model "+str(i) +" weight value: "+str(numpy.log(1./beta[i]))
        
        out = []
        for i in xrange(data.shape[0]):
            out.append(numpy.argmax(data[i]))

        assert y.shape[0] == len(out) 
        s = 0.
        for i in xrange(data.shape[0]):
            if y[i] == out[i]:
                s += 1.
        return s / float(data.shape[0])   
            
    ###########################################################################################        
    
    def m1_controller(self,dirctory,dataset,n_y):
        '''
        # this function is the controller, it will control the running run of 
        # whole frame.
        '''

        dataset=dataset
        datasets = load_data(dataset)
    
        # the datasets[0] is a tuple, it includes two elements. One is
        # train_set_x , another is train_set_y.
        train_set_x, train_set_y = datasets[0]
        n_train_y = train_set_y.eval().shape[0]
        x,y = datasets[2]
        ty = y.eval()
        n_y = n_y

        # init the adaboost.m1, it will get the a array W (40000 rows,)
        # about mnist dataset, which is the weight of erery element.
        Dist = init_adaboost_m1(n_train_y, n_y)
        new_Dist = Dist
        beta = []
        error= []

        print "read the confile file .."
        paths = read_config_name(dirctory)

        print "clean the pickle file"
        remove_pkl_file(os.path.join(dirctory,"test_tem"))

        print "*******************************************************************************"  
        for i in xrange(len(paths)):
            instance = adaboost_m1(Dist = new_Dist, n_y = n_y,train_y = train_set_y.eval())
            PV,true_out = instance.get_coe1() 

            out = []
            # build model
            path = os.path.join(dirctory, paths[str(i)][2])
            if paths[str(i)][0] == "mlp":
                print "Building the " + str(i) + " th model : " + paths[str(i)][0]
                out = build_mlps(kind = 1, PV = PV,true_out = true_out, path = path,      \
                                         datasets = datasets , name = paths[str(i)][1])

            if paths[str(i)][0] == "cnn":
                print "Building the " + str(i) + " th model : " + paths[str(i)][0] 
                out = build_cnn(kind = 1, PV = PV,true_out = true_out, path = path,       \
                                         datasets = datasets , name = paths[str(i)][1])
            if paths[str(i)][0] == "dbn":
                print "Building the " + str(i) + " th model : " + paths[str(i)][0] 
                out = build_dbn(kind = 1, PV = PV,true_out = true_out, path = path,       \
                                         datasets = datasets , name = paths[str(i)][1])

            if paths[str(i)][0] == "sda":
                print "Building the " + str(i) + " th model : " + paths[str(i)][0] 
                out = build_sda(kind = 1, PV = PV,true_out = true_out, path = path,       \
                                         datasets = datasets , name = paths[str(i)][1])

            print "-------------------------------------------------------------------------------"
            # get Hyp_out from the model. (50000,10)
            Hyp_out = numpy.asarray(out).reshape((n_train_y,n_y))

            # invoke onece boost, and get the model's beta, and the model's nest W.
            # this can judge the weight of current model.
            instance = adaboost_m1(new_Dist,n_y,Hyp_out,train_set_y.eval())

            tem_beta , new_Dist = instance.m1_once_boost()

            if tem_beta >= 1.:
                os.remove(os.path.join(dirctory,"test_tem/")+paths[str(i)][1]+".pkl")
                break
            beta.append(tem_beta)
            #new_Dist = tem_Dist
            # sum the all model , and get the final output.
            
            error.append(self.test_m1_accurate(dirctory+'/test_tem',beta,ty))
            print "----The error rate of the testdatas:"
 
            print error
            print "*******************************************************************************"
            print "                                                                               "
        # save the beta and ty into /out_tem/m1_beta_ty.pkl
        cPickle.dump([beta,ty],open(dirctory + "/out_tem/m1_beta_ty.pkl","wb")) 

    ###########################################################################################

    def m2_controller(self,dirctory,dataset,n_y):

        '''
        # this function is the controller, it will control the running run of 
        # whole frame.
        '''
        dataset=dataset
        datasets = load_data(dataset)
    
        # the datasets[0] is a tuple, it includes two elements. One is
        # train_set_x , another is train_set_y.
        train_set_x, train_set_y = datasets[0]
        n_train_y = train_set_y.eval().shape[0]
        x,y = datasets[2]
        ty = y.eval()
        n_y = n_y

        # init the adaboost.m1, it will get the a array W (40000 rows,)
        # about mnist dataset, which is the weight of erery element.
        W = init_adaboost_m2(n_train_y, n_y)
        new_W = W
        beta = []
        error= []

        print "read the confile file .."
        paths = read_config_name(dirctory)

        print "clean the pickle file"
        remove_pkl_file(os.path.join(dirctory,"test_tem"))
        print "*******************************************************************************" 

        for i in xrange(len(paths)):

            instance = adaboost_m2(W = new_W, n_y = n_y,train_y = train_set_y.eval())
            PV,true_out = instance.get_coe2()  
            out = []

            # build model
            path = os.path.join(dirctory, paths[str(i)][2])
            if paths[str(i)][0] == "mlp":
                print "Building the " + str(i) + " th model : " + paths[str(i)][0]
                out = build_mlps(kind = 2, PV = PV,true_out = true_out, path = path,      \
                                         datasets = datasets , name = paths[str(i)][1])
            
            if paths[str(i)][0] == "cnn":
                print "Building the " + str(i) + " th model : " + paths[str(i)][0]
                out = build_cnn(kind = 2, PV = PV,true_out = true_out, path = path,       \
                                         datasets = datasets , name = paths[str(i)][1])

            if paths[str(i)][0] == "dbn":
                print "Building the " + str(i) + " th model : " + paths[str(i)][0]
                out = build_dbn(kind = 2, PV = PV,true_out = true_out, path = path,       \
                                         datasets = datasets , name = paths[str(i)][1])
            
            if paths[str(i)][0] == "sda":
                print "Building the " + str(i) + " th model : " + paths[str(i)][0]
                out = build_sda(kind = 2, PV = PV,true_out = true_out, path = path,       \
                                         datasets = datasets , name = paths[str(i)][1])
            

            # get Hyp_out from the model. (50000,10)
            Hyp_out = numpy.asarray(out).reshape((n_train_y,n_y))
            # invoke onece boost, and get the model's beta, and the model's nest W.
            # this can judge the weight of current model.
            instance = adaboost_m2(new_W,n_y,Hyp_out,train_set_y.eval())
       
            tem_beta , new_W = instance.m2_once_boost()
            beta.append(tem_beta)
            #new_Dist = tem_Dist
            print "-------------------------------------------------------------------------------"
            print "                                                                               "

            # sum the all model , and get the final output.
            error.append(self.test_m2_accurate(dirctory+'/test_tem',beta,ty))
            print "----The error rate of the testdatas: " 
            print error

            print "*******************************************************************************"
            print "                                                                               "
        # save the beta and ty into /out_tem/m2_beta_ty.pkl
        cPickle.dump([beta,ty],open(dirctory + "/out_tem/m2_beta_ty.pkl","wb"))

if __name__ == "__main__":
    dataset = 'mnist.pkl.gz'
    config_path = "../config-example"
    n_y = 10 
    instance = Controller()
    instance.m2_controller(dirctory = config_path, dataset = dataset,n_y = n_y)
