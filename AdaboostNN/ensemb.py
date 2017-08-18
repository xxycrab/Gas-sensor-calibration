import getopt
import sys
import os
sys.path.append("./data-deal")
from deal_date import DealDate_deal
from samples import Samples
sys.path.append("./lib")
from controller import Controller
def command_parser():
    ''' 
        THIS PART CAN PARSER THE COMMAND PARAMETERS.
        Value 0 is off, 1 is open.
 
    '''
    try:        
        source_path= ""
        save_path  = ""
        # The default value of data_deal model.
        iamges_deal  = 0      
        size_x       = 28
        size_y       = 28
        channel      = 3
        # samples model parameters
        samples      = 0
        no_replace   = 0
        replace      = 0
        number       = 10000
         
        # train and test the model
        train_test = 0
        dataset    = 'mnist.pkl.gz'
        config_path= "../config-example"
        n_y        = 10
        m1         = 0
        m2         = 0

        opts, args = getopt.getopt(sys.argv[1:], "ho:",      \
                                  ["help",                   \
                                   "images_deal",            \
                                   "images_path=",           \
                                   "size_x=",                \
                                   "size_y=" ,               \
                                   "channel=",               \

                                   "samples",                \
                                   "no_replace=",            \
                                   "replace=",               \
                                   "source_path=" ,          \
                                   "number=",                \
                                   "save_path=",             \

                                   "train_test",             \
                                   "dataset=",               \
                                   "config_path=",           \
                                   "n_y=" ,                  \
                                   "m1=",                    \
                                   "m2="                     
                                  ])
        #print opts,args
        for x,y in opts:
            if x in ("-h","--help"):
                print "-----------------------------------------------------------"                
                print "ensemb: command line brew"
                print "usage : python ensemb.py <command> <args>"
                print "---------------------------------------------------------------------"
                print "images_deal : deal the image to pickle file"
                print "----------    ensemb.py --images_deal     "
                print "----------              --source_path=/home/gd/ensemble-nn/data/image"
                print "----------              --size_x=28"
                print "----------              --size_y=28 "
                print "----------              --channel=3"
                print "---------------------------------------------------------------------"
                print "samples     : sample the samples from the datasets"
                print "----------    ensemb.py --samples     "
                print "----------              --no_replace=1"
                print "----------              --replace=0 "
                print "----------              --source_path=./data/mnist.pkl.gz" 
                print "----------              --number=30000"
                print "----------              --save_path=./data/mnist-sample-replace.pkl"
                print "---------------------------------------------------------------------"
                print "train_test  : train and test the model"
                print "----------    ensemb.py --train_test     "
                print "----------              --dataset =/home/gd/ensemble-nn/data/mnist.pkl.gz"
                print "----------              --config_path=./config-example "
                print "----------              --n_y =10" 
                print "----------              --m1  = 1"
                print "---------------------------------------------------------------------"
  
            # MODEL IMAGES_DEAL 
            if x in ("--images_deal"):
                print "Deal the iamge .."
                iamges_deal = 1 
            if x in ("--source_path"):
                images_path = y
            if x in ("--size_x"):
                size_x = int(y)
            if x in ("--size_y"):
                size_y = int(y)
            if x in ("--channel"):
                channel = int(y)

            # MODEL SAMPLES
            if x in ("--samples"):
                print "sampling .."
                samples = 1 
            if x in ("--no_replace"):
                no_replace = int(y)
            if x in ("--replace"):
                replace = int(y)
            if x in ("--source_path"):
                source_path = y
            if x in ("--number"):
                number = int(y)
            if x in ("--save_path"):
                save_path = y          

            # TRAIN AND TEST A MODEL
            if x in ("--train_test"):
                print "train and test .."
                train_test = 1 
            if x in ("--dataset"):
                dataset = y
            if x in ("--config_path"):
                config_path = y
            if x in ("--n_y"):
                n_y = int(y)
            if x in ("--m1"):
                m1 = int(y)
            if x in ("--m2"):
                m2 = int(y)

        if iamges_deal == 1:
            source_path="/home/gd/ensemble-nn/data/image"
            DealDate_deal(image_path=source_path,size_x=size_x,size_y = size_y, channel =channel)
        if samples     == 1:
            source_path="/home/gd/ensemble-nn/data/mnist.pkl.gz"
            save_path="/home/gd/ensemble-nn/data/mnist-sample-replace.pkl" 
            if no_replace == 0 and replace == 0:
                print "no_replace or replace ?"
            if no_replace == 1 and replace == 1:
                print "no_replace or replace ?"
            if no_replace == 1:
                instance = Samples()
                instance.no_replace_sample(source_path, number, save_path)
            if replace    == 1:
                instance = Samples.Samples()
                instance.replace_sample(source_path, number, save_path)
        if train_test     == 1:
            if m1 == 0 and m2 == 0:
                print "adaboost-m1 or adaboost-m2 ?"
            if m1 == 1 and m2 == 1:
                print "adaboost-m1 or adaboost-m2 ?"
            if m1 == 1 and m2 == 0:
                instance = Controller()
                instance.m1_controller(dirctory = config_path, dataset = dataset,n_y = n_y)
            if m1 == 0 and m2 == 1:
                instance = Controller()
                instance.m2_controller(dirctory = config_path, dataset = dataset,n_y = n_y)
    

        sys.exit()
    except Exception as err:
        print err



command_parser()
'''
filepath ='../tmp/test.txt'
print os.path.split(filepath)
os.mkdir(os.path.split(filepath)[0])
'''
