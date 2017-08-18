import numpy
import sys
import os
import cPickle
import gzip
import random

class Samples(object):
    def __init__(self):
        pass
    
    # LOAD THE DATA
    def load_data(self,path):
        if os.path.isfile(path):
            if path[-3:] == ".gz":
                f = gzip.open(path, 'rb')
                datasets = cPickle.load(f)
                f.close()
                return datasets
            if path[-4:] ==".pkl":
                f = open(path, 'rb')
                datasets = cPickle.load(f)
                f.close()  
                return datasets
        return "The path is illegal!"   

    def no_replace_sample(self, source_path, number ,save_path):
        # LOAD THE DATASETS
        print "sample with no replace .."
        train_set, valid_set, test_set = self.load_data(source_path)
        train_x , train_y = train_set

        # SAMPLING THE SAMPLE FROM THE DATASETS
        length = train_y.shape[0]
        list_length = range(length)
        samples_num = []
        try:
            samples_num = random.sample(list_length,number)
        except Exception as error:
            print error

        # GET THE NEW DATASETS
        new_train_x = numpy.zeros((number, train_x.shape[1]),dtype = "float32")
        new_train_y = numpy.zeros((number, ),dtype = "int64") 
        for i in xrange(number):
            new_train_x[i][:] = train_x[samples_num[i]][:]
            new_train_y[i]    = train_y[samples_num[i]]    
  
        # SAVA THE SAMPLES INTO THE FILE
        new_datasets = ((new_train_x,new_train_y),valid_set, test_set)
        f = open(save_path,"wb")
        cPickle.dump(new_datasets,f)
        f.close()

    def replace_sample(self, source_path, number ,save_path):
        # LOAD THE DATASETS
        print "sampling with replace .."
        train_set, valid_set, test_set = self.load_data(source_path)
        train_x , train_y = train_set

        # SAMPLING THE SAMPLE FROM THE DATASETS
        length = train_y.shape[0]
        list_num = range(length)
        samples_num = []
        try:
            #random.shuffle(list_num)
            samples_num = numpy.random.randint(0,list_num,number)
        except Exception as error:
            print error

        # GET THE NEW DATASETS
        new_train_x = numpy.zeros((number, train_x.shape[1]),dtype = "float32")
        new_train_y = numpy.zeros((number, ),dtype = "int64") 
        for i in xrange(number):
            new_train_x[i][:] = train_x[samples_num[i]][:]
            new_train_y[i]    = train_y[samples_num[i]]    
  
        # SAVA THE SAMPLES INTO THE FILE
        new_datasets = ((new_train_x,new_train_y),valid_set, test_set)
        f = open(save_path,"wb")
        cPickle.dump(new_datasets,f)
        f.close()


if __name__ == "__main__":
    instance = Samples()
    instance.no_replace_sample("/home/gd/ensemble-nn/data/mnist.pkl.gz", 30000 ,\
                               "/home/gd/ensemble-nn/data/mnist-0.pkl")
    
               
